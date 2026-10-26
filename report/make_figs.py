"""Figures for the final MP2 report (E-full + answer-group aggregation)."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = HERE / 'figs'
FIG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT / 'qctext'))

from common import DATA, load_inputs, parse_answer, read_jsonl  # noqa: E402
from qctext_metrics import trap_ids  # noqa: E402
from exp_baselines import label_bank  # noqa: E402

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 200, 'savefig.bbox': 'tight'})

AGG = 0.6
PATTERNS = [(3, 1), (2, 1, 1), (2, 2), (1, 1, 1, 1)]


def aggregation(scores, parsed, a):
    out = np.asarray(scores, dtype=np.float64).copy()
    for i in range(len(parsed)):
        groups = defaultdict(list)
        for j, v in enumerate(parsed[i]):
            if v is not None:
                groups[v].append(j)
        for members in groups.values():
            mean = out[i, members].mean()
            for j in members:
                out[i, j] = (1 - a) * out[i, j] + a * mean
        for j, v in enumerate(parsed[i]):
            if v is None:
                out[i, j] = -1e9
    return out


def picks(scores, parsed):
    out = []
    for i in range(len(parsed)):
        best, best_score = -1, -np.inf
        for j in range(4):
            if parsed[i][j] is None:
                continue
            if scores[i, j] > best_score:
                best_score, best = scores[i, j], j
        out.append(best if best >= 0 else 0)
    return np.array(out)


def load_split(split):
    bank, _ = load_inputs(split)
    gold = read_jsonl(DATA / f'gsm8k_{split}.jsonl')
    parsed, labels = label_bank(bank, gold)
    return bank, parsed, labels


def pattern_of(row):
    counts = Counter(v for v in row if v is not None)
    return tuple(sorted(counts.values(), reverse=True))


def main():
    data = {}
    for split, path in (('validation', ROOT / 'qctext/runs/efull_validation-predictions_probs.npy'),
                        ('test', ROOT / 'qctext/runs/efull_test_probs.npy')):
        bank, parsed, labels = load_split(split)
        probs = np.load(path)[:len(bank)]
        chosen = picks(aggregation(probs, parsed, AGG), parsed)
        correct = np.array([bool(labels[i, chosen[i]]) for i in range(len(bank))])
        data[split] = dict(bank=bank, parsed=parsed, labels=labels, probs=probs,
                           chosen=chosen, correct=correct)

    # ---------- Figure 1: pattern composition and accuracy -------------------
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={'width_ratios': [1, 1.25]})
    ax = axs[0]
    vcount = Counter(pattern_of(r) for r in data['validation']['parsed'])
    tcount = Counter(pattern_of(r) for r in data['test']['parsed'])
    labels_ = ['(3,1)', '(2,1,1)', '(2,2)', '(1,1,1,1)']
    vshare = [100 * vcount[p] / 798 for p in PATTERNS]
    tshare = [100 * tcount[p] / 467 for p in PATTERNS]
    x = np.arange(len(PATTERNS)); w = 0.38
    ax.bar(x - w / 2, vshare, w, label='validation (798)', color='#4C72B0')
    ax.bar(x + w / 2, tshare, w, label='test (467)', color='#DD8452')
    for i in range(len(PATTERNS)):
        ax.text(i - w / 2, vshare[i] + 1, f'{vshare[i]:.0f}', ha='center', fontsize=7)
        ax.text(i + w / 2, tshare[i] + 1, f'{tshare[i]:.0f}', ha='center', fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(labels_, fontsize=8)
    ax.set_ylabel('share of questions (%)'); ax.set_ylim(0, 52)
    ax.set_title('(a) answer-pattern mixture', fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc='upper right')

    ax = axs[1]
    vacc, tacc = [], []
    for p in PATTERNS:
        vi = [i for i, r in enumerate(data['validation']['parsed']) if pattern_of(r) == p]
        ti = [i for i, r in enumerate(data['test']['parsed']) if pattern_of(r) == p]
        vacc.append(100 * data['validation']['correct'][vi].mean())
        tacc.append(100 * data['test']['correct'][ti].mean())
    ax.bar(x - w / 2, vacc, w, color='#4C72B0')
    ax.bar(x + w / 2, tacc, w, color='#DD8452')
    for i in range(len(PATTERNS)):
        ax.text(i - w / 2, vacc[i] + 1, f'{vacc[i]:.0f}', ha='center', fontsize=7)
        ax.text(i + w / 2, tacc[i] + 1, f'{tacc[i]:.0f}', ha='center', fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(labels_, fontsize=8)
    ax.set_ylim(0, 108); ax.set_ylabel('selection accuracy (%)')
    ax.set_title('(b) accuracy by pattern, final method', fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / 'fig1_patterns.png'); plt.close(fig)

    # ---------- Figure 2: aggregation strength ------------------------------
    grid = np.round(np.arange(0, 1.001, 0.05), 2)
    vcurve, tcurve, vtrap, ttrap = [], [], [], []
    for a in grid:
        vp = picks(aggregation(data['validation']['probs'], data['validation']['parsed'], a),
                   data['validation']['parsed'])
        tp = picks(aggregation(data['test']['probs'], data['test']['parsed'], a),
                   data['test']['parsed'])
        vcurve.append(100 * np.mean([data['validation']['labels'][i, vp[i]]
                                     for i in range(798)]))
        tcurve.append(100 * np.mean([data['test']['labels'][i, tp[i]] for i in range(467)]))
        vtr = trap_ids(data['validation']['parsed'], data['validation']['labels'])
        ttr = trap_ids(data['test']['parsed'], data['test']['labels'])
        vtrap.append(100 * np.mean([data['validation']['labels'][i, vp[i]] for i in vtr]))
        ttrap.append(100 * np.mean([data['test']['labels'][i, tp[i]] for i in ttr]))

    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ax = axs[0]
    ax.plot(grid, vcurve, 'o-', ms=2.5, lw=1.2, color='#4C72B0', label='validation')
    ax.plot(grid, tcurve, 's-', ms=2.5, lw=1.2, color='#DD8452', label='test')
    ax.axvspan(0.4, 0.9, color='0.9', zorder=0)
    ax.axvline(AGG, ls='--', lw=0.9, color='0.35')
    ax.annotate('selected $a$=0.6', xy=(AGG, tcurve[12]), xytext=(0.08, 85.6),
                fontsize=7.5, arrowprops=dict(arrowstyle='->', lw=0.7, color='0.35'))
    ax.set_xlabel('answer-group aggregation strength $a$')
    ax.set_ylabel('overall accuracy (%)')
    ax.set_title('(a) validation plateau 0.4-0.9 (shaded)', fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc='center right')
    ax = axs[1]
    ax.plot(grid, vtrap, 'o-', ms=2.5, lw=1.2, color='#4C72B0', label='validation')
    ax.plot(grid, ttrap, 's-', ms=2.5, lw=1.2, color='#DD8452', label='test')
    ax.axvline(AGG, ls='--', lw=0.9, color='0.35')
    ax.set_xlabel('$a$'); ax.set_ylabel('trap accuracy (%)')
    ax.set_title('(b) the gain is concentrated in traps', fontsize=9)
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / 'fig2_aggregation.png'); plt.close(fig)

    # ---------- Figure 3: input ablation ------------------------------------
    ab = ROOT / 'qctext/runs/ablation'
    _, vparsed, vlabels = load_split('validation')
    vhard = [i for i in range(len(vparsed))
             if len(set(v for v in vparsed[i] if v is not None)) > 1]
    vtraps = trap_ids(vparsed, vlabels)
    table = []
    for kind in ('embed', 'surface', 'noarith', 'nocons', 'hand22', 'hand29'):
        path = ab / f'{kind}_setattn_validation_probs.npy'
        if not path.exists():
            continue
        probs = np.load(path)[:len(vparsed)]
        logits = np.log(np.clip(probs, 1e-9, 1.0))
        chosen = picks(aggregation(probs, vparsed, 0.0), vparsed)
        overall = 100 * np.mean([vlabels[i, chosen[i]] for i in range(len(vparsed))])
        hard = 100 * np.mean([vlabels[i, chosen[i]] for i in vhard])
        tr = 100 * np.mean([vlabels[i, chosen[i]] for i in vtraps])
        table.append(dict(kind=kind, overall=overall, hard=hard, trap=tr,
                          trap_correct=int(sum(vlabels[i, chosen[i]] for i in vtraps)),
                          trap_n=len(vtraps), dim=int(probs.shape[0] and 0)))
    resmlp_path = ab / 'hand22_resmlp_validation_probs.npy'
    if resmlp_path.exists():
        probs = np.load(resmlp_path)[:len(vparsed)]
        chosen = picks(aggregation(probs, vparsed, 0.0), vparsed)
        table.append(dict(kind='hand22_resmlp',
                          overall=100 * np.mean([vlabels[i, chosen[i]] for i in range(len(vparsed))]),
                          hard=100 * np.mean([vlabels[i, chosen[i]] for i in vhard]),
                          trap=100 * np.mean([vlabels[i, chosen[i]] for i in vtraps]),
                          trap_correct=int(sum(vlabels[i, chosen[i]] for i in vtraps)),
                          trap_n=len(vtraps), dim=0))
    (HERE / 'ablation_table.json').write_text(json.dumps(table, indent=2), encoding='utf-8')

    label = {'embed': 'embedding only', 'surface': 'embedding + surface (5)',
             'noarith': 'embedding + hand, no arithmetic',
             'nocons': 'embedding + hand, consensus zeroed',
             'hand22': 'embedding + hand (22) [control]',
             'hand29': 'embedding + hand + symbolic (29)',
             'hand22_resmlp': 'independent scorer (same features)'}
    y = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    h = 0.26
    ax.barh(y + h, [r['overall'] for r in table], h, color='#8EAADB', label='overall')
    ax.barh(y, [r['hard'] for r in table], h, color='#4C72B0', label='hard')
    ax.barh(y - h, [r['trap'] for r in table], h, color='#DD8452', label='trap')
    for i, r in enumerate(table):
        ax.text(max(r['overall'] + 0.6, 36.5), i + h, f"{r['overall']:.1f}", va='center', fontsize=7)
        ax.text(max(r['hard'] + 0.6, 36.5), i, f"{r['hard']:.1f}", va='center', fontsize=7)
        ax.text(max(r['trap'] + 0.6, 36.5), i - h, f"{r['trap']:.1f}", va='center', fontsize=7)
    ax.set_yticks(y); ax.set_yticklabels([label[r['kind']] for r in table], fontsize=7.5)
    ax.invert_yaxis(); ax.set_xlim(35, 108)
    ax.set_xlabel('validation accuracy (%)')
    ax.set_title('Input and architecture ablation (validation)', fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc='lower center',
              bbox_to_anchor=(0.5, -0.42))
    fig.tight_layout()
    fig.savefig(FIG / 'fig3_ablation.png'); plt.close(fig)

    # ---------- Figure 4: calibration ---------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.7))
    edges = [0.30, 0.50, 0.70, 0.80, 0.90, 1.001]
    calib = {}
    for ax, split, colour, tag in ((axs[0], 'validation', '#4C72B0', 'a'),
                                   (axs[1], 'test', '#DD8452', 'b')):
        scores = aggregation(data[split]['probs'], data[split]['parsed'], AGG)
        conf = scores.max(axis=1)
        correct = data[split]['correct']
        xs, ys, ns, ece = [], [], [], 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (conf >= lo) & (conf < hi)
            if sel.sum():
                acc = 100 * correct[sel].mean()
                xs.append(100 * conf[sel].mean()); ys.append(acc); ns.append(int(sel.sum()))
                ece += sel.mean() * abs(acc - 100 * conf[sel].mean()) / 100
        ax.plot([30, 100], [30, 100], ls=':', color='0.6', lw=1)
        ax.plot(xs, ys, 'o-', ms=4, lw=1.2, color=colour)
        for xv, yv, nv in zip(xs, ys, ns):
            ax.text(xv, yv - 7, str(nv), ha='center', fontsize=6.5, color='0.35')
        ax.set_xlabel('top score after aggregation (%)')
        ax.set_ylabel('selection accuracy (%)')
        ax.set_xlim(30, 100); ax.set_ylim(40, 105)
        ax.set_title(f'({tag}) {split}  (ECE {ece:.2f}, n={len(conf)})', fontsize=9)
        calib[split] = dict(x=xs, y=ys, n=ns, ece=ece)
    fig.tight_layout()
    fig.savefig(FIG / 'fig4_calibration.png'); plt.close(fig)

    summary = {
        'validation': dict(overall=100 * data['validation']['correct'].mean()),
        'test': dict(overall=100 * data['test']['correct'].mean()),
        'aggregation_curve': {'a': grid.tolist(), 'validation': vcurve, 'test': tcurve,
                              'validation_trap': vtrap, 'test_trap': ttrap},
    }
    (HERE / 'fig_data.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('figures written', sorted(p.name for p in FIG.glob('*.png')))


if __name__ == '__main__':
    main()
