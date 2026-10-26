"""Frozen MP2 selector: five-fold set-attention ensemble + answer-group aggregation.

Method
------
Each candidate is scored from its 1024-d whole-trace embedding concatenated with 22
reference-free text features computed from the question and that candidate alone.
Five members were trained on disjoint 4/5 splits of the training bank, each
standardising its inputs with its own fold statistics.  The members' sigmoid outputs
are averaged, then candidates that state the same parsed final answer are partially
pooled:

    score_j = (1 - a) * p_j + a * mean(p over the same-answer group)

and the argmax over parseable candidates is returned.  Unparseable candidates can
never be selected.  ``a`` was fixed on the official validation split before the test
split was scored, over the grid {0, 0.1, ..., 1.0}: every value in [0.4, 0.9] ties the
validation maximum (746/798), and the tie-break is the highest validation trap accuracy
(36/60, reached from a >= 0.6), then the smallest a, which gives 0.6.

No external data, no pretrained weights, no reference-answer lookups.  Inference is
pure PyTorch on CPU.  See report/MP2-report.pdf for the experiments behind it.
"""
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import parse_answer
from features import hand_features
from models import build_model

AGGREGATION = 0.6


def _parsed(bank):
    n = len(bank)
    parsed = np.empty((n, 4), dtype=object)
    for i, row in enumerate(bank):
        for j, candidate in enumerate(row['candidates']):
            parsed[i, j] = parse_answer(candidate['text'])
    return parsed


def _probs(bank, arrays, checkpoint):
    import torch

    bundle = Path(checkpoint)
    files = sorted(bundle.glob('*.pt'))
    if not files:
        raise SystemExit(f'no checkpoints in {bundle}')

    parsed = _parsed(bank)
    n = len(bank)
    feats = hand_features(bank, parsed, drop_consensus=False, feature_dim=22)
    emb = arrays['embeddings'].reshape(n * 4, -1)
    base = np.concatenate([emb, feats.reshape(n * 4, -1)], axis=1).astype(np.float32)

    probs = np.zeros((n, 4), dtype=np.float64)
    for path in files:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        cfg = ckpt['config']
        model = build_model('setattn', cfg['in_dim'], width=cfg['width'],
                            layers=cfg['layers'], heads=cfg['heads'],
                            dropout=cfg['dropout'])
        model.load_state_dict({k: v.float() for k, v in ckpt['state'].items()})
        model.eval()
        xs = ((base - np.asarray(cfg['mu'])) / np.asarray(cfg['sd']))
        xs = xs.astype(np.float32).reshape(n, 4, -1)
        with torch.no_grad():
            logits = model(torch.from_numpy(xs)).numpy()
        probs += 1.0 / (1.0 + np.exp(-logits))
    probs /= len(files)
    return probs, parsed


def _choose(probs, parsed, aggregation=AGGREGATION):
    scores = np.asarray(probs, dtype=np.float64).copy()
    for i in range(len(parsed)):
        groups = defaultdict(list)
        for j, value in enumerate(parsed[i]):
            if value is not None:
                groups[value].append(j)
        for members in groups.values():
            mean = scores[i, members].mean()
            for j in members:
                scores[i, j] = (1 - aggregation) * scores[i, j] + aggregation * mean
        for j, value in enumerate(parsed[i]):
            if value is None:
                scores[i, j] = -1e9
    picks = []
    for i in range(len(parsed)):
        best, best_score = -1, -np.inf
        for j in range(4):
            if parsed[i][j] is None:
                continue
            if scores[i, j] > best_score:
                best_score, best = scores[i, j], j
        picks.append(int(best if best >= 0 else 0))
    return picks


def predict(bank, arrays, checkpoint):
    """Return one Python integer in 0..3 per question, in bank order."""
    probs, parsed = _probs(bank, arrays, checkpoint)
    return _choose(probs, parsed)
