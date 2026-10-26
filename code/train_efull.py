"""Train one member of the five-fold set-attention ensemble.

Run from this directory (``code/``):

    python train_efull.py --fold 0 --seed 37 --epochs 15 --tag efull_f0_s37

Weights are written to ``runs/efull/``.  ``--data-dir`` defaults to the supplied
``data/`` directory next to this file.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import common as _common  # noqa: E402
from common import load_inputs, parse_answer, read_jsonl  # noqa: E402
from features import hand_features  # noqa: E402
from full_features import quantity_features, repair_features  # noqa: E402
from models import build_model  # noqa: E402
from qctext_metrics import trap_ids  # noqa: E402

MODELS = ROOT / 'runs' / 'efull'
OOF = ROOT / 'runs' / 'efull_oof'
MODELS.mkdir(parents=True, exist_ok=True)
OOF.mkdir(parents=True, exist_ok=True)


class BankData(Dataset):
    def __init__(self, x, y, idx):
        self.x = x
        self.y = y
        self.idx = np.asarray(idx, dtype=np.int64)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = int(self.idx[k])
        return torch.from_numpy(self.x[i]), torch.from_numpy(self.y[i]).float(), i


def collate(batch):
    return (torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            torch.tensor([b[2] for b in batch], dtype=torch.int64))


def mass_loss(logits, correct):
    log_all = torch.logsumexp(logits, dim=1)
    neg = torch.full_like(logits, -float('inf'))
    log_corr = torch.logsumexp(torch.where(correct.bool(), logits, neg), dim=1)
    return log_all - log_corr


def pick_argmax(scores, parsed, i):
    best, best_score = -1, -np.inf
    for j in range(4):
        if parsed[i, j] is None:
            continue
        if scores[i, j] > best_score:
            best_score, best = scores[i, j], j
    return best


def evaluate(logits, parsed, labels, idx):
    idx = list(idx)
    logits = np.asarray(logits)
    if logits.shape[0] == len(idx):
        score_map = {int(i): logits[k] for k, i in enumerate(idx)}

        class View:
            def __getitem__(self, item):
                row, col = item
                return score_map[int(row)][col]

        view = View()
    else:
        view = logits
    picks = [pick_argmax(view, parsed, i) for i in idx]
    correct = sum(labels[i, p] for i, p in zip(idx, picks))
    hard = [i for i in idx if len(set(v for v in parsed[i] if v is not None)) > 1]
    hard_correct = sum(labels[i, pick_argmax(view, parsed, i)] for i in hard)
    return {'n': len(idx), 'correct': int(correct), 'accuracy': 100 * correct / len(idx),
            'hard_n': len(hard), 'hard_correct': int(hard_correct),
            'hard_accuracy': 100 * hard_correct / len(hard) if hard else 0.0}


def label_bank(bank, gold_rows):
    """Align each candidate with the reference answer: (parsed, labels)."""
    n = len(bank)
    labels = np.zeros((n, 4), dtype=bool)
    parsed = np.empty((n, 4), dtype=object)
    for qi, (row, gold) in enumerate(zip(bank, gold_rows)):
        target = parse_answer(gold['answer'])
        for ci, candidate in enumerate(row['candidates']):
            value = parse_answer(candidate['text'])
            parsed[qi, ci] = value
            labels[qi, ci] = value is not None and value == target
    return parsed, labels


def load_data(split):
    bank, arrays = load_inputs(split, _common.DATA)
    gold = read_jsonl(_common.DATA / f'gsm8k_{split}.jsonl')
    parsed, labels = label_bank(bank, gold)
    return bank, parsed, labels, arrays


def build_features(bank, parsed, mode, feature_dim=22, drop_consensus=False):
    if mode == 'embed':
        return None
    hand = (hand_features(bank, parsed, drop_consensus=drop_consensus,
                          feature_dim=feature_dim) if 'hand' in mode else None)
    if mode == 'hand':
        return hand.reshape(len(bank), 4, -1)
    arith = np.concatenate([quantity_features(bank, parsed),
                            repair_features(bank, parsed)], axis=2)
    if mode == 'arith':
        return arith
    return np.concatenate([hand.reshape(len(bank), 4, -1), arith], axis=2)


def build_x(emb, feats):
    base = emb.reshape(-1, emb.shape[-1])
    if feats is None:
        return base
    return np.concatenate([base, feats.reshape(-1, feats.shape[-1])], axis=1)


def pair_loss(logits, correct):
    delta = logits[:, :, None] - logits[:, None, :]
    valid = correct.bool()[:, :, None] & (~correct.bool())[:, None, :]
    n = valid.sum(dim=(1, 2)).clamp_min(1)
    return (F.softplus(-delta) * valid.float()).sum(dim=(1, 2)) / n


def group_masks(parsed):
    n = len(parsed)
    mask = np.zeros((n, 4, 4), dtype=np.float32)
    for i, row in enumerate(parsed):
        for j in range(4):
            if row[j] is None:
                mask[i, j, j] = 1.0
                continue
            for k in range(4):
                if row[j] == row[k]:
                    mask[i, j, k] = 1.0
    return mask


def group_loss(logits, labels, masks):
    score = (logits[:, :, None] * masks).sum(2) / masks.sum(2).clamp_min(1.0)
    return F.binary_cross_entropy_with_logits(
        score, labels, reduction='none').mean(1)


def train_epoch(model, loader, opt, weights, device, loss_name='bce_mass', masks=None):
    model.train()
    total, n = 0.0, 0
    for x, y, idx in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction='none').mean(1)
        if loss_name == 'pair':
            per_q = bce + 0.5 * pair_loss(logits, y)
        elif loss_name == 'listwise' and masks is not None:
            m = torch.from_numpy(masks[idx.numpy()]).to(device)
            per_q = bce + 0.5 * group_loss(logits, y, m)
        elif loss_name == 'bce':
            per_q = bce
        else:
            per_q = bce + 0.5 * mass_loss(logits, y)
        w = torch.from_numpy(weights[idx.numpy()]).to(device)
        loss = (per_q * w).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += float(loss.detach()) * len(idx)
        n += len(idx)
    return total / max(n, 1)


@torch.no_grad()
def predict(model, x, idx, batch=64, device=None):
    model.eval()
    out = []
    for start in range(0, len(idx), batch):
        chunk = idx[start:start + batch]
        xb = torch.from_numpy(x[chunk]).to(device)
        out.append(model(xb).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--seed', type=int, default=37)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--inner-frac', type=float, default=0.1)
    ap.add_argument('--width', type=int, default=128)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--feature-dim', type=int, choices=(18, 22, 29), default=22)
    ap.add_argument('--feature-mode', choices=('hand', 'arith', 'hand_arith', 'embed'),
                    default='hand')
    ap.add_argument('--drop-consensus', action='store_true')
    ap.add_argument('--trap-weight', type=float, default=1.0)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--wd', type=float, default=1e-3)
    ap.add_argument('--hard-weight', type=float, default=2.0)
    ap.add_argument('--error-weight', type=float, default=1.0)
    ap.add_argument('--loss', choices=('bce_mass', 'pair', 'listwise', 'bce'),
                    default='bce_mass')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--data-dir', default=None,
                    help='course data directory (default: ./data)')
    ap.add_argument('--tag', default=None)
    args = ap.parse_args()

    _common.DATA = Path(args.data_dir).expanduser().resolve() if args.data_dir else ROOT / 'data'
    print('data', _common.DATA, flush=True)
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print('device', device, flush=True)

    bank, parsed, labels, arrays = load_data('train')
    feats = build_features(bank, parsed, args.feature_mode, args.feature_dim,
                           args.drop_consensus)
    x = build_x(arrays['embeddings'], feats).astype(np.float32)
    n = len(bank)
    rng = np.random.default_rng(13)
    perm = rng.permutation(n)
    size = n // 5
    folds = [perm[i * size:(i + 1) * size] for i in range(5)]
    outer = folds[args.fold]
    pool = np.concatenate([folds[i] for i in range(5) if i != args.fold])
    weights = np.ones(n, dtype=np.float32)
    for i in range(n):
        if len(set(v for v in parsed[i] if v is not None)) > 1:
            weights[i] = args.hard_weight
    for i in trap_ids(parsed, labels):
        weights[i] = max(weights[i], args.trap_weight)
    if args.error_weight > 1:
        wrong = np.load(ROOT / 'runs/efull_oof/efull_error_mask.npy')
        weights[wrong] = np.maximum(weights[wrong], args.error_weight)
    masks = group_masks(parsed) if args.loss == 'listwise' else None

    pilot_mode = args.pilot
    if pilot_mode:
        r = np.random.default_rng(17 + args.seed)
        p = r.permutation(len(pool))
        dev_n = int(len(pool) * args.inner_frac)
        dev_idx = pool[p[:dev_n]]
        train_idx = pool[p[dev_n:]]
    else:
        dev_idx = None
        train_idx = pool

    mu = x[train_idx].mean(axis=0)
    sd = x[train_idx].std(axis=0) + 1e-6
    xs = (x - mu) / sd
    xs = xs.reshape(n, 4, -1)
    loader = DataLoader(BankData(xs, labels, train_idx), batch_size=args.batch,
                        shuffle=True, num_workers=0, collate_fn=collate, drop_last=False)

    model = build_model('setattn', xs.shape[-1], width=args.width, layers=args.layers,
                        heads=args.heads, dropout=args.dropout)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    tag = args.tag or (f'efull_{"pilot" if pilot_mode else "f" + str(args.fold)}_s{args.seed}')
    best = {'hard': -1.0, 'epoch': -1, 'state': None, 'metrics': None}

    for epoch in range(args.epochs):
        started = time.time()
        loss = train_epoch(model, loader, opt, weights, device, args.loss, masks)
        if pilot_mode:
            logits = predict(model, xs, dev_idx, 64, device)
            metrics = evaluate(logits, parsed, labels, dev_idx)
            print(f'epoch {epoch + 1:02d} loss={loss:.4f} dev_hard={metrics["hard_accuracy"]:.2f} '
                  f'dev_acc={metrics["accuracy"]:.2f} {time.time() - started:.1f}s', flush=True)
            if metrics['hard_accuracy'] > best['hard']:
                best = {'hard': metrics['hard_accuracy'], 'epoch': epoch + 1,
                        'state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        'metrics': metrics}
            if epoch + 1 - best['epoch'] >= args.patience and best['epoch'] > 0:
                print(f'early stop at {epoch + 1}; best {best["epoch"]}', flush=True)
                break
        else:
            print(f'epoch {epoch + 1:02d} loss={loss:.4f} {time.time() - started:.1f}s', flush=True)

    if pilot_mode:
        model.load_state_dict(best['state'])
        epochs = best['epoch']
        metrics = best['metrics']
    else:
        epochs = args.epochs
        logits = predict(model, xs, outer, 64, device)
        metrics = evaluate(logits, parsed, labels, outer)
        np.savez_compressed(OOF / f'{tag}.npz', idx=outer, logits=logits)

    path = MODELS / f'{tag}.pt'
    torch.save({
        'state': {k: v.detach().cpu().half() for k, v in model.state_dict().items()},
        'config': {'width': args.width, 'layers': args.layers, 'heads': args.heads,
                   'dropout': args.dropout, 'in_dim': int(xs.shape[-1]),
                   'feature_dim': args.feature_dim, 'drop_consensus': args.drop_consensus,
                   'feature_mode': args.feature_mode, 'trap_weight': args.trap_weight,
                   'loss': args.loss,
                   'mu': mu, 'sd': sd},
        'epochs': epochs, 'metrics': metrics,
    }, path)
    (MODELS / f'{tag}.json').write_text(json.dumps({
        'tag': tag, 'args': vars(args), 'epochs': epochs, 'metrics': metrics},
        indent=2, default=lambda o: float(o)), encoding='utf-8')
    print('saved', path, flush=True)


if __name__ == '__main__':
    main()
