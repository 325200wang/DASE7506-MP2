"""Hand-crafted features visible at inference time, incl. arithmetic re-verification.

Everything here is computed from the current question's inputs only.
"""
import ast
import re
from collections import Counter, defaultdict
from fractions import Fraction

import numpy as np

from common import parse_answer
from symverify import verify_trace

_QNUM_RE = re.compile(r'-?\d[\d,]*(?:\.\d+)?')

# ---- arithmetic re-verification -------------------------------------------------

_NUM = r'-?\d[\d,]*(?:\.\d+)?'
_EQ_RE = re.compile(
    r'(?<![\w.])('                    # LHS start, not preceded by word char or dot
    r'\(?\s*' + _NUM + r'\s*\)?'
    r'(?:\s*[-+*/×÷]\s*\(?\s*' + _NUM + r'\s*\)?)+'
    r')\s*=\s*(' + _NUM + r'(?:\s*/\s*\d+)?)'   # RHS number or fraction a/b
)


def _fraction(tok):
    tok = tok.replace(',', '').strip()
    try:
        return Fraction(tok)
    except (ValueError, ZeroDivisionError):
        return None


def _safe_eval_lhs(expr):
    """Evaluate a restricted arithmetic expression with Fraction arithmetic."""
    expr = expr.replace('×', '*').replace('÷', '/').strip()
    try:
        tree = ast.parse(expr, mode='eval')
    except (SyntaxError, ValueError):
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            return None
        if isinstance(node, ast.UnaryOp) and not isinstance(node.op, ast.USub):
            return None
        if not isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Constant, ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub)):
            return None
    try:
        val = _eval_node(tree.body)
        return val if isinstance(val, Fraction) else None
    except (ValueError, ZeroDivisionError, AttributeError, SyntaxError):
        return None


def _eval_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return Fraction(str(node.value))
        return None
    if isinstance(node, ast.Num):  # py<3.8 compat
        return Fraction(str(node.n))
    if isinstance(node, ast.BinOp):
        l, r = _eval_node(node.left), _eval_node(node.right)
        if l is None or r is None:
            return None
        if isinstance(node.op, ast.Add):
            return l + r
        if isinstance(node.op, ast.Sub):
            return l - r
        if isinstance(node.op, ast.Mult):
            return l * r
        if isinstance(node.op, ast.Div):
            return l / r
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _eval_node(node.operand)
        return -v if v is not None else None
    return None


def chain_features(text, answer_fraction=None):
    """Chain-level arithmetic verification (Prove/EVoSS lightweight version).

    Builds the dependency graph of checkable equations (an edge j->k when
    equation k's LHS uses the verified result of equation j) and checks the
    path that produces the final answer. Returns 4 scalars; neutral defaults
    when nothing is checkable.
    """
    first_error_rel, dep_usage, path_clean, orphan_frac = 1.0, 0.0, 0.5, 0.0
    try:
        eqs = []
        for m in _EQ_RE.finditer(text):
            if '%' in text[max(0, m.start() - 1):m.end() + 1]:
                continue
            lv = _safe_eval_lhs(m.group(1))
            rv = _fraction(m.group(2))
            if lv is None or rv is None:
                continue
            lhs_tokens = [_fraction(t) for t in _QNUM_RE.findall(m.group(1))]
            lhs_tokens = [t for t in lhs_tokens if t is not None]
            eqs.append({'lv': lv, 'rv': rv, 'ok': lv == rv, 'uses': set(lhs_tokens)})
        if not eqs:
            return [first_error_rel, dep_usage, path_clean, orphan_frac]
        n = len(eqs)
        for idx, e in enumerate(eqs):
            if not e['ok']:
                first_error_rel = idx / n
                break
        # dependency edges: later equation uses an earlier equation's result value,
        # regardless of correctness (so wrong provenance reaches the answer path)
        rev_edges = defaultdict(set)   # k -> {j: k uses j's result}
        used = set()
        for k in range(n):
            for j in range(k):
                if eqs[j]['rv'] in eqs[k]['uses']:
                    rev_edges[k].add(j)
                    used.add(j)
        if n > 1:
            dep_usage = len(used) / (n - 1)
        n_correct = sum(1 for e in eqs if e['ok'])
        correct_used = used & {j for j in range(n) if eqs[j]['ok']}
        if n_correct > 0:
            orphan_frac = (n_correct - len(correct_used)) / n_correct
        # answer path: last equation whose RHS equals the final answer
        if answer_fraction is not None:
            ans_idx = [i for i, e in enumerate(eqs) if e['rv'] == answer_fraction]
            if ans_idx:
                root = ans_idx[-1]
                seen, stack = set(), [root]
                while stack:
                    u = stack.pop()
                    if u in seen:
                        continue
                    seen.add(u)
                    stack.extend(rev_edges.get(u, ()))
                bad_on_path = any(not eqs[i]['ok'] for i in seen)
                path_clean = 0.0 if bad_on_path else 1.0
            # no equation producing the parsed answer: keep 0.5
    except Exception:
        pass
    return [first_error_rel, dep_usage, path_clean, orphan_frac]


def sym_features(text):
    """Candidate-level aggregates from the symbolic verification engine."""
    d = verify_trace(text)
    steps = d['steps']
    n = len(steps)
    if n == 0:
        return [0.0] * 7
    s = steps
    n_bind_err = sum(x['n_bind_err'] for x in s)
    n_bind = sum(x['n_bind'] for x in s)
    n_nameval = sum(x['n_nameval'] + x['defines_name'] for x in s)
    uses = sum(x['uses_names'] for x in s)
    ok_frac = sum(1.0 for x in s if x['ok']) / n
    n_eq = sum(x['n_eq'] for x in s)
    n_wrong = sum(x['n_wrong'] for x in s)
    return [
        float(n_bind_err > 0),
        np.log1p(n_bind_err),
        np.log1p(n_bind),
        np.log1p(n_nameval),
        np.log1p(uses),
        np.log1p(n),
        ok_frac,
    ]


def check_equations(text):
    """Return (n_checked, n_correct, n_wrong, n_unknown)."""
    n_checked = n_correct = n_wrong = n_unknown = 0
    for m in _EQ_RE.finditer(text):
        lhs, rhs = m.group(1), m.group(2)
        if '%' in text[max(0, m.start() - 1):m.end() + 1]:
            continue  # skip percent contexts (semantics ambiguous)
        lv = _safe_eval_lhs(lhs)
        rv = _fraction(rhs)
        if lv is None or rv is None:
            n_unknown += 1
            continue
        n_checked += 1
        if lv == rv:
            n_correct += 1
        else:
            n_wrong += 1
    return n_checked, n_correct, n_wrong, n_unknown


# ---- main feature extractor ------------------------------------------------------

def limit_feature_dim(feats, feature_dim):
    x = np.asarray(feats)
    if feature_dim not in (18, 22, 29):
        raise ValueError(f'unsupported historical feature dimension: {feature_dim}')
    if x.ndim < 1 or x.shape[-1] < feature_dim:
        raise ValueError(f'cannot select {feature_dim} features from {x.shape}')
    return x[..., :feature_dim]


def hand_features(bank, parsed, drop_consensus=False, feature_dim=None):
    """Per-candidate numeric features visible at inference time. [n, 4, D] -> [n*4, D].

    drop_consensus: zero the vote-related features (group size, distinct count,
    max group) for the E08 ablation.
    """
    n = len(bank)
    feats = np.zeros((n, 4, 29), dtype=np.float32)
    qnums_cache = {}
    for qi, row in enumerate(bank):
        counts = Counter(v for v in parsed[qi] if v is not None)
        n_distinct = len(counts)
        max_group = max(counts.values()) if counts else 0
        if qi not in qnums_cache:
            qnums_cache[qi] = set(_QNUM_RE.findall(row['question']))
        qnums = qnums_cache[qi]
        for ci, c in enumerate(row['candidates']):
            text = c['text']
            a = parsed[qi, ci]
            has_parse = float(a is not None)
            group_size = counts.get(a, 0) if a is not None else 0
            boxed = '\\boxed' in text
            h4 = '####' in text
            consistent = 0.0
            if boxed:
                m = re.findall(r'\\boxed\{([^{}]+)\}', text)
                if m and a is not None:
                    consistent = float(parse_answer(m[-1]) == a)
            n_checked, n_corr, n_wrong, n_unk = check_equations(text)
            checked = max(n_checked, 1)
            # does the final equation's RHS equal the parsed final answer?
            final_eq_matches = 0.0
            if a is not None and n_checked > 0:
                last = None
                for m in _EQ_RE.finditer(text):
                    lv = _safe_eval_lhs(m.group(1))
                    rv = _fraction(m.group(2))
                    if lv is not None and rv is not None:
                        last = rv
                if last is not None:
                    final_eq_matches = float(last == _fraction(a))
            af = _fraction(a) if a is not None else None
            qcov = (sum(1 for qn in qnums if qn in text) / len(qnums)) if qnums else 0.0
            chain = chain_features(text, af)
            feats[qi, ci] = [
                has_parse,
                np.log1p(group_size),
                n_distinct,
                np.log1p(max_group),
                np.log1p(len(text)),
                sum(ch.isdigit() for ch in text) / max(len(text), 1),
                float(boxed or h4),
                float(boxed and h4 and consistent),
                np.log1p(sum(1 for ch in text if ch in '+-*/=÷×')),
                float(n_checked > 0),
                n_checked / 10.0,
                n_corr / checked if n_checked else 0.0,
                n_wrong / checked if n_checked else 0.0,
                n_unk / 10.0,
                final_eq_matches,
                qcov,
                float(af is not None and af.denominator == 1),
                np.log1p(float(abs(af))) if af is not None else 0.0,
            ] + chain + sym_features(text)
    if drop_consensus:
        feats[:, :, 1:4] = 0.0
    feats = feats.reshape(n * 4, -1)
    return limit_feature_dim(feats, 29 if feature_dim is None else feature_dim)
