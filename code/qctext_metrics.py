"""MP2 metrics and answer pooling for QCText."""
from collections import Counter, defaultdict

import numpy as np

COMP4 = {
    (3, 1): 196 / 467,
    (2, 1, 1): 102 / 467,
    (2, 2): 39 / 467,
    (1, 1, 1, 1): 38 / 467,
}


def pattern(parsed_row):
    return tuple(sorted(Counter(v for v in parsed_row if v is not None).values(), reverse=True))


def pick_argmax(scores, parsed, i):
    best, best_score = -1, -np.inf
    for j in range(4):
        if parsed[i, j] is None:
            continue
        if scores[i, j] > best_score:
            best_score, best = scores[i, j], j
    return best


def group_logits(logits, parsed):
    logits = np.asarray(logits)
    out = logits.copy()
    for i in range(len(parsed)):
        groups = defaultdict(list)
        for j, value in enumerate(parsed[i]):
            if value is not None:
                groups[value].append(j)
        for members in groups.values():
            out[i, members] = logits[i, members].mean()
        for j, value in enumerate(parsed[i]):
            if value is None:
                out[i, j] = -1e9
    return out


def evaluate_group_logits(logits, data, idx=None):
    return evaluate(group_logits(logits, data['parsed']), data, idx)


def pick_pooled(probs, parsed, i):
    groups = defaultdict(list)
    for j, value in enumerate(parsed[i]):
        if value is not None:
            groups[value].append(j)
    best, best_score = -1, -np.inf
    for members in groups.values():
        score = float(np.mean(probs[i, members]))
        if score > best_score:
            best_score = score
            best = members[int(np.argmax(probs[i, members]))]
    return best


def trap_ids(parsed, labels):
    out = []
    for i in range(len(parsed)):
        counts = Counter(v for v in parsed[i] if v is not None)
        if not counts:
            continue
        top = max(counts.values())
        winners = [v for v, c in counts.items() if c == top]
        if len(winners) != 1 or top < 2:
            continue
        if int(labels[i].sum()) != 1:
            continue
        correct = [j for j in range(4) if labels[i, j]]
        if correct and parsed[i, correct[0]] != winners[0]:
            out.append(i)
    return out


def evaluate(scores, data, idx=None, pooled=False):
    parsed, labels = data['parsed'], data['labels']
    ids = range(len(parsed)) if idx is None else idx
    ids = list(ids)
    scores = np.asarray(scores)
    local = scores.shape[0] == len(ids)
    pick = pick_pooled if pooled else pick_argmax

    if local:
        score_map = {int(i): scores[k] for k, i in enumerate(ids)}

        class LocalScores:
            def __getitem__(self, item):
                row, col = item
                return score_map[int(row)][col]

        view = LocalScores()
    else:
        view = scores
    picks = [pick(view, parsed, i) for i in ids]
    correct = sum(labels[i, p] for i, p in zip(ids, picks))
    hard = [i for i in ids if len(set(v for v in parsed[i] if v is not None)) > 1]
    traps = trap_ids(parsed, labels)
    traps = [i for i in traps if i in set(ids)]
    hard_correct = sum(labels[i, pick(view, parsed, i)] for i in hard)
    trap_correct = sum(labels[i, pick(view, parsed, i)] for i in traps)
    by_pattern = {}
    for i in ids:
        p = pattern(parsed[i])
        by_pattern.setdefault(p, [0, 0])
        by_pattern[p][0] += 1
        by_pattern[p][1] += int(labels[i, pick(view, parsed, i)])
    num = den = 0.0
    for p, weight in COMP4.items():
        if p in by_pattern and by_pattern[p][0]:
            num += weight * by_pattern[p][1] / by_pattern[p][0]
            den += weight
    comp4 = 100 * num / den if den else 0.0
    return {
        'n': len(ids),
        'correct': int(correct),
        'accuracy': 100 * correct / len(ids),
        'hard_n': len(hard),
        'hard_correct': int(hard_correct),
        'hard_accuracy': 100 * hard_correct / len(hard) if hard else 0.0,
        'trap_n': len(traps),
        'trap_correct': int(trap_correct),
        'trap_accuracy': 100 * trap_correct / len(traps) if traps else 0.0,
        'comp4': comp4,
    }
