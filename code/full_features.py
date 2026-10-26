"""Feature families for the full feature-rich expert.

B1 linear embedding views, B2 hand/group, B3 quantity/repair, B5 PCA.
B4 lexical is produced by lexical_pair.py.
"""
import ast
import math
import re
from collections import Counter, defaultdict
from fractions import Fraction

import numpy as np

from features import (_EQ_RE, _fraction, _safe_eval_lhs, chain_features,
                      check_equations)

NUM_RE = re.compile(r'-?\d[\d,]*(?:\.\d+)?(?:/\d+)?')
WORD_NUM = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
    'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12,
    'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16,
    'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
    'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
    'eighty': 80, 'ninety': 90, 'dozen': 12, 'half': .5, 'quarter': .25,
    'double': 2, 'twice': 2, 'triple': 3,
}
TOPICS = ('each', 'per', 'total', 'left', 'remaining', 'more', 'less', 'twice',
          'double', 'percent', 'profit', 'cost', 'spent', 'week', 'hour',
          'minute', 'day', 'average', 'difference', 'together', 'every')


def numbers(text, word_numbers=False):
    out = []
    for match in NUM_RE.finditer(text):
        try:
            value = float(Fraction(match.group().replace(',', '')))
            if np.isfinite(value) and abs(value) < 1e12:
                out.append((match.start(), value))
        except (ValueError, ZeroDivisionError, OverflowError):
            pass
    if word_numbers:
        for match in re.finditer(r'\b[a-z]+\b', text.lower()):
            if match.group() in WORD_NUM:
                out.append((match.start(), float(WORD_NUM[match.group()])))
    return np.asarray([v for _, v in sorted(out)], dtype=np.float64)


def close(values, target):
    return np.abs(values - target) <= 1e-5 * np.maximum(1, np.abs(target))


def pair_values(nums):
    if not len(nums):
        return np.zeros(0), np.zeros(0, dtype=np.uint32)
    n = len(nums)
    a = np.repeat(nums, n)
    b = np.tile(nums, n)
    ia = np.repeat(np.arange(n), n)
    ib = np.tile(np.arange(n), n)
    valid = ia != ib
    mask = (np.left_shift(np.uint32(1), ia) | np.left_shift(np.uint32(1), ib))[valid]
    a, b = a[valid], b[valid]
    with np.errstate(divide='ignore', invalid='ignore'):
        vals = np.concatenate([a + b, a - b, a * b, a / b])
    masks = np.tile(mask, 4)
    keep = np.isfinite(vals) & (np.abs(vals) < 1e12)
    return vals[keep], masks[keep]


def program_summary(nums, values, masks, target):
    exact = close(values, target)
    one = bool(exact.any())
    maxcov = 2 if one else 0
    two = False
    if len(values) and len(nums):
        a = values[:, None]
        b = nums[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            candidates = (a + b, a - b, b - a, a * b, a / b, b / a)
        for c in candidates:
            hit = close(c, target) & np.isfinite(c)
            if hit.any():
                two = True
                i, j = np.nonzero(hit)
                cov = [int(int(masks[ii]) | (1 << int(jj))).bit_count()
                       for ii, jj in zip(i[:512], j[:512])]
                maxcov = max(maxcov, max(cov))
    three = False
    if len(values):
        order = np.argsort(values)
        sv = values[order]
        with np.errstate(divide='ignore', invalid='ignore'):
            needed = (target - values, values - target, target + values,
                      target / values, values / target, target * values)
        for req in needed:
            idx = np.searchsorted(sv, req)
            for off in (0, -1):
                at = np.clip(idx + off, 0, len(values) - 1)
                hit = np.isfinite(req) & close(sv[at], req)
                if hit.any():
                    three = True
                    left = np.flatnonzero(hit)
                    right = order[at[hit]]
                    cov = [int(int(masks[i]) | int(masks[j])).bit_count()
                           for i, j in zip(left[:512], right[:512])]
                    maxcov = max(maxcov, max(cov))
    return [float(one), float(two), float(three), maxcov / max(1, len(nums))]


def residual(values, target):
    if not len(values):
        return 1e6
    return float(np.min(np.abs(values - target) / np.maximum(1, np.abs(target))))


def quantity_features(bank, parsed):
    n = len(bank)
    out = np.zeros((n, 4, 62), dtype=np.float32)
    for i, row in enumerate(bank):
        q = row['question']
        qnums = numbers(q, True)[:12]
        values, masks = pair_values(qnums)
        total = qnums.sum() if len(qnums) else 0.
        product = np.prod(qnums) if len(qnums) and len(qnums) <= 8 else 0.
        whole = np.asarray([total, product, qnums.max() if len(qnums) else 0.,
                            qnums.min() if len(qnums) else 0.])
        ql = q.lower()
        schema = None
        if len(qnums) >= 5:
            a, b, c, d, e = qnums[:5]
            schema = np.asarray([a + b * c - d * e, a - b * c + d * e,
                                 a * b + c * d - e, a * b - c * d + e,
                                 a - b * c - d * e, a + b * c + d * e,
                                 (a - b) * c * d, a * (b - c) * d,
                                 a * b * c * d, a * b - c * d, a * b + c * d,
                                 (a - b - c) * d, a * b / max(c * d, 1e-9),
                                 a / max(b, 1e-9) * c * d, (a + b + c) * d,
                                 a * (b + c + d)])
        for j, candidate in enumerate(row['candidates']):
            text = candidate['text']
            answer = parsed[i, j]
            try:
                target = float(Fraction(answer)) if answer is not None else 0.
            except (ValueError, OverflowError):
                target = 0.
            if not np.isfinite(target) or abs(target) > 1e15:
                target = 0.
            one, two, three, coverage = program_summary(qnums, values, masks, target)
            unit_vals = target * np.asarray([60, 1 / 60, 100, .01, 12, 1 / 12, 7, 1 / 7, 1000, .001])
            if len(values):
                unit = [float(close(values, v).any()) for v in unit_vals]
            else:
                unit = [0.0] * 10
            if schema is not None:
                hit = (np.isfinite(schema) & close(schema, target)).astype(float)
                schema_res = float(np.nanmin(np.abs(schema - target) /
                                             np.maximum(1, np.abs(target))))
            else:
                hit = np.zeros(16)
                schema_res = 1e6
            eq = check_equations(text)
            nums_text = numbers(text)
            qcov = (sum(1 for v in qnums if close(nums_text, v).any()) /
                    max(1, len(qnums)))
            answer_in_text = float(any(close(nums_text, target))) if len(nums_text) else 0.
            chain = chain_features(text, _fraction(answer) if answer is not None else None)
            out[i, j] = [
                one, two, three, coverage,
                1.0 / (1.0 + residual(values, target)),
                1.0 / (1.0 + residual(whole, target)),
                *unit,
                *hit,
                1.0 / (1.0 + schema_res),
                *[float(w in ql) for w in TOPICS],
                eq[0] / 10., eq[1] / max(eq[0], 1), eq[2] / max(eq[0], 1),
                qcov, answer_in_text,
                *chain[:3],
            ]
    return out


def repair_features(bank, parsed):
    n = len(bank)
    out = np.zeros((n, 4, 8), dtype=np.float32)
    for i, row in enumerate(bank):
        repairs = []
        targets = []
        for candidate in row['candidates']:
            found = []
            for match in _EQ_RE.finditer(candidate['text']):
                left = _safe_eval_lhs(match.group(1))
                right = _fraction(match.group(2))
                if left is None or right is None:
                    continue
                if left != right:
                    found.append((float(right), float(left)))
            targets.append(found)
        all_corr = [c for found in targets for _, c in found]
        for j, candidate in enumerate(row['candidates']):
            answer = parsed[i, j]
            try:
                target = float(Fraction(answer)) if answer is not None else 0.
            except (ValueError, OverflowError):
                target = 0.
            found = targets[j]
            support = sum(1 for k, other in enumerate(targets) if k != j
                          for wrong, corr in other if abs(corr - target) <= 1e-5 * max(1, abs(target)))
            repaired = any(abs(wrong - target) <= 1e-5 * max(1, abs(target)) for wrong, _ in found)
            corrected = any(abs(corr - target) <= 1e-5 * max(1, abs(target)) for _, corr in found)
            cross = sum(1 for v in all_corr if abs(v - target) <= 1e-5 * max(1, abs(target)))
            out[i, j] = [len(found), float(repaired), float(corrected),
                         support / 4., cross / 16., float(len(found) > 0 and not repaired),
                         float(repaired and corrected), float(len(found) > 2)]
    return out


def group_features(parsed):
    n = len(parsed)
    out = np.zeros((n, 4, 4), dtype=np.float32)
    for i, row in enumerate(parsed):
        counts = Counter(v for v in row if v is not None)
        max_group = max(counts.values()) if counts else 0
        n_distinct = len(counts)
        for j, value in enumerate(row):
            g = counts.get(value, 0) if value is not None else 0
            out[i, j] = [g / 4., n_distinct / 4., max_group / 4.,
                         float(g == max_group and g > 0)]
    return out


def embedding_views(emb, parsed):
    mean = emb.mean(axis=1, keepdims=True)
    centered = emb - mean
    cluster = np.zeros_like(emb)
    for i, row in enumerate(parsed):
        groups = defaultdict(list)
        for j, value in enumerate(row):
            if value is not None:
                groups[value].append(j)
        for members in groups.values():
            cluster[i, members] = emb[i, members].mean(axis=0)
    spread = centered.std(axis=1, keepdims=True) + 0.01
    return {'raw': emb, 'centered': centered, 'cluster': cluster,
            'contrast': centered / spread}
