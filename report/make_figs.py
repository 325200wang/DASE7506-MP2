"""Rebuild the four report figures from the records shipped in this repository.

    cd report && python make_figs.py

Inputs: records/probs/*.npy, records/aggregation_curve.json,
records/ablation_table.json. Output: figs/*.png
"""
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

from common import DATA, load_inputs, parse_answer, read_jsonl  # noqa: E402
from qctext_metrics import trap_ids  # noqa: E402
import submission as S  # noqa: E402

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 200, 'savefig.bbox': 'tight'})

AGG = 0.6
PATTERNS = [(3, 1), (2, 1, 1), (2, 2), (1, 1, 1, 1)]


def load_split(split):
    bank, _ = load_inputs(split)
    gold = read_jsonl(DATA / f'gsm8k_{split}.jsonl')
    parsed = S._parsed(bank)
    labels = np.zeros((len(bank), 4), dtype=bool)
    for i, g in enumerate(gold):
        ga = parse_answer(g['answer'])
        for j in range(4):
            labels[i, j] = parsed[i, j] is not None and parsed[i, j] == ga
    probs = np.load(ROOT / 'records' / 'probs' / f'efull_{split}_probs.npy')
    picks = S._choose(probs, parsed, AGG)
    correct = np.array([bool(labels[i, picks[i]]) for i in range(len(bank))])
    return dict(bank=bank, parsed=parsed, labels=labels, probs=probs,
                picks=picks, correct=correct)


def pattern_of(row):
    counts = Counter(v for v in row if v is not None)
    return tuple(sorted(counts.values(), reverse=True))


def main():
    data = {s: load_split(s) for s in ('validation', 'test')}
    curve = json.loads((ROOT / 'records' / 'aggregation_curve.json').read_text())

    # ---------- Figure 1: pattern mixture and accuracy -----------------------
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

    # ---------- Figure 2: aggregation strength (0.1 grid) -------------------
    grid = np.array(curve['a'])
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ax = axs[0]
    ax.plot(grid, curve['validation'], 'o-', ms=3.5, lw=1.2, color='#4C72B0', label='validation')
    ax.plot(grid, curve['test'], 's-', ms=3.5, lw=1.2, color='#DD8452', label='test')
    ax.axvspan(0.4, 0.9, color='0.9', zorder=0)
    ax.axvline(AGG, ls='--', lw=0.9, color='0.35')
    ax.annotate('selected $a$=0.6', xy=(AGG, curve['test'][6]), xytext=(0.05, 85.6),
                fontsize=7.5, arrowprops=dict(arrowstyle='->', lw=0.7, color='0.35'))
    ax.set_xlabel('answer-group aggregation strength $a$')
    ax.set_ylabel('overall accuracy (%)')
    ax.set_title('(a) validation ties over $a\\in[0.4,0.9]$ (shaded)', fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc='center right')
    ax = axs[1]
    ax.plot(grid, curve['validation_trap'], 'o-', ms=3.5, lw=1.2, color='#4C72B0', label='validation')
    ax.plot(grid, curve['test_trap'], 's-', ms=3.5, lw=1.2, color='#DD8452', label='test')
    ax.axvspan(0.4, 0.9, color='0.9', zorder=0)
    ax.axvline(AGG, ls='--', lw=0.9, color='0.35')
    ax.set_xlabel('$a$'); ax.set_ylabel('trap accuracy (%)')
    ax.set_title('(b) the gain is concentrated in traps', fontsize=9)
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / 'fig2_aggregation.png'); plt.close(fig)

    # ---------- Figure 3: input and architecture ablation -------------------
    rows = json.loads((ROOT / 'records' / 'ablation_table.json').read_text())
    label = {'embed': 'embedding only',
             'surface': 'embedding + surface (5)',
             'noarith': 'embedding + hand, no arithmetic',
             'nocons': 'embedding + hand, consensus zeroed',
             'hand22': 'embedding + hand (22) [control]',
             'hand29': 'embedding + hand + symbolic (29)',
             'hand22_resmlp': 'independent scorer (same features)'}
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    h = 0.26
    ax.barh(y + h, [r['overall'] for r in rows], h, color='#8EAADB', label='overall')
    ax.barh(y, [r['hard'] for r in rows], h, color='#4C72B0', label='hard')
    ax.barh(y - h, [r['trap'] for r in rows], h, color='#DD8452', label='trap')
    for i, r in enumerate(rows):
        ax.text(max(r['overall'] + 0.6, 36.5), i + h, f"{r['overall']:.1f}", va='center', fontsize=7)
        ax.text(max(r['hard'] + 0.6, 36.5), i, f"{r['hard']:.1f}", va='center', fontsize=7)
        ax.text(max(r['trap'] + 0.6, 36.5), i - h, f"{r['trap']:.1f}", va='center', fontsize=7)
    ax.set_yticks(y); ax.set_yticklabels([label[r['kind']] for r in rows], fontsize=7.5)
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
    for ax, split, colour, tag in ((axs[0], 'validation', '#4C72B0', 'a'),
                                   (axs[1], 'test', '#DD8452', 'b')):
        scores = np.asarray(data[split]['probs'], dtype=np.float64).copy()
        for i in range(len(data[split]['parsed'])):
            groups = defaultdict(list)
            for j, v in enumerate(data[split]['parsed'][i]):
                if v is not None:
                    groups[v].append(j)
            for members in groups.values():
                mean = scores[i, members].mean()
                for j in members:
                    scores[i, j] = (1 - AGG) * scores[i, j] + AGG * mean
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
    fig.tight_layout()
    fig.savefig(FIG / 'fig4_calibration.png'); plt.close(fig)
    print('figures written', sorted(p.name for p in FIG.glob('*.png')))


if __name__ == '__main__':
    main()
