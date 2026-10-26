"""Symbolic verification engine for candidate traces (Prove/EVoSS-style).

Pipeline per candidate text:
  1. LaTeX normalization: \\frac{a}{b} -> ((a)/(b)), \\text{Name} -> Name
     (captured as a name token), \\times/\\div, strip $ and markup.
  2. Step segmentation: split on '###' headers, '---' rules, blank-line blocks.
  3. Per step: extract (name, expr, value) equations and plain "name = value"
     bindings; substitute known bindings into expressions and re-execute.
  4. Symbol table persists across steps.
Outputs per-step correctness labels (pseudo-labels for the PRM) plus
step-level features. Pure rule-based; no model involved.
"""
import ast
import re
from collections import defaultdict
from fractions import Fraction

_NUM = r'-?\d[\d,]*(?:\.\d+)?'
_FRAC_RE = re.compile(r'\\d?frac\{([^{}]+)\}\{([^{}]+)\}')
_TEXT_RE = re.compile(r'\\text\{([^{}]+)\}')
_BOXED_RE = re.compile(r'\\boxed\{([^{}]+)\}')
_NAMEVAL_RE = re.compile(
    r'^\s*(?:[-*•]\s*)?(?:\*\*)?([A-Za-z][A-Za-z \'\-]{1,40}?(?: [a-z]+)?)(?:\*\*)?\s*[:=]\s*(' + _NUM + r')\s*$',
    re.M)


def _fraction(tok):
    tok = tok.replace(',', '').strip().rstrip('.').strip('$ \\')
    if tok.endswith('%'):
        try:
            return Fraction(tok[:-1]) / 100
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return Fraction(tok)
    except (ValueError, ZeroDivisionError):
        m = re.fullmatch(r'(' + _NUM + r')\s*/\s*(\d+)', tok)
        if m:
            try:
                return Fraction(m.group(1).replace(',', '')) / Fraction(m.group(2))
            except (ValueError, ZeroDivisionError):
                return None
    return None


def _eval_arith(expr):
    """Evaluate a restricted arithmetic expression to Fraction."""
    expr = expr.strip()
    if not expr or not re.fullmatch(r'[\d\s.+\-*/(),]+', expr):
        return None
    try:
        tree = ast.parse(expr, mode='eval')
    except (SyntaxError, ValueError):
        return None
    try:
        val = _eval_node(tree.body)
        return val if isinstance(val, Fraction) else None
    except (ValueError, ZeroDivisionError, AttributeError):
        return None


def _eval_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return Fraction(str(node.value))
        return None
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


def normalize_latex(text):
    """Return (plain_text, boxed_contents). \\text{Name} becomes Name."""
    boxed = _BOXED_RE.findall(text)
    text = _BOXED_RE.sub(lambda m: ' ' + m.group(1) + ' ', text)

    def frac_sub(m):
        return ' (( ' + m.group(1) + ' ) / ( ' + m.group(2) + ' )) '

    prev = None
    while prev != text:
        prev = text
        text = _FRAC_RE.sub(frac_sub, text)
    text = text.replace('\\times', '*').replace('\\div', '/').replace('×', '*').replace('÷', '/')
    text = text.replace('\\%', ' % ')
    text = _TEXT_RE.sub(lambda m: ' ' + m.group(1) + ' ', text)
    text = re.sub(r'\\[a-zA-Z]+', ' ', text)           # remaining latex commands
    text = text.replace('$', ' ')
    text = re.sub(r'[{}]', ' ', text)
    return text, boxed


def _norm_name(name):
    return re.sub(r'[^a-z0-9 ]', '', name.lower()).strip()


def split_steps(plain_text):
    """Segment normalized text into reasoning steps."""
    text = re.sub(r'(?m)^#{1,4}\s*', '\n@@@ ', plain_text)   # headers start steps
    text = re.sub(r'(?m)^\s*---+\s*$', '\n@@@ ', text)
    chunks = re.split(r'\n\s*\n|@@@', text)
    steps = []
    for ch in chunks:
        ch = ch.strip()
        if not ch:
            continue
        # drop pure-answer/noise chunks later via feature filters
        steps.append(ch)
    return steps


def extract_equations(step_text):
    """Extract (name, expr, value) checks from one step.

    Handles chains 'Name = expr1 = expr2 = value': every expression part is
    checked against the final value. Returns a list of checks; a check with
    expr=None is a plain 'Name = value' binding.
    """
    out = []
    for raw_line in step_text.split('\n'):
        line = re.sub(r'^\s*[-*•]\s*', '', raw_line.strip())
        if '=' not in line:
            continue
        parts = [p.strip() for p in line.split('=') if p.strip() != '']
        if len(parts) < 2:
            continue
        tail_val = _fraction(parts[-1])
        if tail_val is None:
            continue
        head = parts[0]
        is_name = re.fullmatch(r"[A-Za-z][A-Za-z '\-]{1,40}", head) is not None
        if len(parts) == 2 and not re.search(r'[-+*/]', head):
            # 'Name = value' plain binding (or prose with '='; harmless if wrong)
            if is_name:
                out.append({'name': head, 'expr': None, 'value': tail_val, 'raw': raw_line})
            continue
        # one check per expression part
        expr_idx = [i for i, p in enumerate(parts[:-1]) if re.search(r'[-+*/]', p)]
        for i in expr_idx:
            nm = head if is_name else None
            out.append({'name': nm, 'expr': parts[i], 'value': tail_val, 'raw': raw_line})
    return out


def substitute_names(expr, bindings):
    """Replace known quantity names in expr with their bound numeric values.
    Returns (new_expr, n_substitutions) or (None, 0) if ambiguous."""
    work = ' ' + expr + ' '
    n_sub = 0
    # longest names first; word-boundary match to avoid partial replacements
    for name in sorted(bindings, key=len, reverse=True):
        pat = re.compile(r'\b' + re.escape(name) + r'\b', re.I)
        if pat.search(work):
            work = pat.sub(' ( ' + str(float(bindings[name])) + ' ) ', work)
            n_sub += 1
    return work, n_sub


def verify_trace(text):
    """Full symbolic verification. Returns per-step records."""
    plain, boxed = normalize_latex(text)
    steps = split_steps(plain)
    bindings = {}
    records = []
    for si, step in enumerate(steps):
        rec = {'step': si, 'n_eq': 0, 'n_correct': 0, 'n_wrong': 0,
               'n_bind_err': 0, 'n_bind': 0, 'n_nameval': 0, 'defines_name': 0,
               'uses_names': 0, 'n_tokens': len(step.split())}
        eqs = extract_equations(step)
        # plain "Name = value" bindings (bullets like '- Total slices = 16')
        for m in _NAMEVAL_RE.finditer(step):
            nm = _norm_name(m.group(1))
            v = _fraction(m.group(2))
            if nm and v is not None and len(nm) >= 3:
                bindings[nm] = v
                rec['n_nameval'] += 1
        for eq in eqs:
            rec['n_eq'] += 1
            nm = _norm_name(eq['name']) if eq['name'] else None
            if eq['expr'] is None:
                if nm:
                    bindings[nm] = eq['value']
                    rec['defines_name'] += 1
                continue
            sub_expr, n_sub = substitute_names(eq['expr'], bindings)
            val = _eval_arith(sub_expr) if sub_expr else None
            rec['uses_names'] += n_sub
            if nm:
                rec['n_bind'] += 1
            if val is None:
                continue
            if val == eq['value']:
                rec['n_correct'] += 1
                if nm:
                    bindings[nm] = eq['value']
                    rec['defines_name'] += 1
            else:
                if n_sub > 0 and nm:
                    rec['n_bind_err'] += 1
                else:
                    rec['n_wrong'] += 1
        rec['ok'] = (rec['n_wrong'] == 0 and rec['n_bind_err'] == 0)
        records.append(rec)
    return {'steps': records, 'boxed': boxed, 'n_steps': len(records)}
