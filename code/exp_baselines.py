"""Baseline scan: majority vote vs linear models (text / embedding / features / combined).

Development-only experiment. Trains on the official train split, evaluates on
validation. C is selected on an internal 80/20 question-level split of train;
validation is only reported on, not searched.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, load_inputs, read_jsonl, parse_answer  # noqa: E402
from features import hand_features  # noqa: E402

SEED = 13


def label_bank(bank, gold_rows):
    n = len(bank)
    labels = np.zeros((n, 4), dtype=bool)
    parsed = np.empty((n, 4), dtype=object)
    for qi, (row, g) in enumerate(zip(bank, gold_rows)):
        ga = parse_answer(g['answer'])
        for ci, c in enumerate(row['candidates']):
            ca = parse_answer(c['text'])
            parsed[qi, ci] = ca
            labels[qi, ci] = ca is not None and ca == ga
    return parsed, labels


def pattern_of(parsed_row):
    groups = Counter(v for v in parsed_row)
    return tuple(sorted(groups.values(), reverse=True))


def majority_pick(parsed_row):
    counts = Counter(v for v in parsed_row if v is not None)
    if not counts:
        return 0
    best = max(counts.values())
    for i, v in enumerate(parsed_row):
        if v is not None and counts[v] == best:
            return i
    return 0


def pick_by_scores(scores, parsed, idxs=None):
    """Argmax per question, skipping unparseable candidates. scores: [len(idxs)*4]."""
    if idxs is None:
        idxs = range(len(parsed))
    picks = {}
    k = 0
    for i in idxs:
        best, best_s = -1, -np.inf
        for ci in range(4):
            s = scores[k * 4 + ci]
            if parsed[i, ci] is None:
                s = -np.inf
            if s > best_s:
                best_s, best = s, ci
        picks[i] = best
        k += 1
    return picks


def eval_picks(picks, labels, parsed, idxs=None):
    if idxs is None:
        idxs = list(range(len(labels)))
    acc = 100 * np.mean([labels[i, picks[i]] for i in idxs])
    per_pattern = {}
    for i in idxs:
        pat = str(pattern_of(parsed[i]))
        d = per_pattern.setdefault(pat, [0, 0])
        d[1] += 1
        d[0] += int(labels[i, picks[i]])
    per_pattern = {k: round(100 * v[0] / v[1], 1) for k, v in sorted(per_pattern.items())}
    maj_ok = {i: labels[i, majority_pick(parsed[i])] for i in idxs}
    n_maj_ok = sum(maj_ok.values())
    on_maj_correct = np.mean([labels[i, picks[i]] for i in idxs if maj_ok[i]]) if n_maj_ok else None
    on_maj_wrong = np.mean([labels[i, picks[i]] for i in idxs if not maj_ok[i]]) if n_maj_ok < len(idxs) else None
    n_fix = sum(1 for i in idxs if (not maj_ok[i]) and labels[i, picks[i]])
    n_break = sum(1 for i in idxs if maj_ok[i] and not labels[i, picks[i]])
    return {
        'accuracy_pct': round(acc, 2),
        'per_pattern': per_pattern,
        'on_majority_correct_pct': None if on_maj_correct is None else round(100 * on_maj_correct, 2),
        'on_majority_wrong_pct': None if on_maj_wrong is None else round(100 * on_maj_wrong, 2),
        'fixed_vs_majority': n_fix,
        'broke_vs_majority': n_break,
    }


def main():
    splits = {}
    for split in ('train', 'validation'):
        bank, arrays = load_inputs(split)
        gold = read_jsonl(DATA / f'gsm8k_{split}.jsonl')
        parsed, labels = label_bank(bank, gold)
        splits[split] = dict(bank=bank, emb=arrays['embeddings'], parsed=parsed, labels=labels)

    tr, va = splits['train'], splits['validation']
    n_tr, n_va = len(tr['labels']), len(va['labels'])
    y_tr = tr['labels'].reshape(-1).astype(int)

    results = {}
    results['majority'] = eval_picks(
        {i: majority_pick(va['parsed'][i]) for i in range(n_va)}, va['labels'], va['parsed'])

    # question-level internal split of train for hyperparameter selection
    rng = np.random.default_rng(SEED)
    dev_idx = rng.permutation(n_tr)[: n_tr // 5]
    q_dev_mask = np.zeros(n_tr, dtype=bool)
    q_dev_mask[dev_idx] = True
    cand_dev_mask = np.repeat(q_dev_mask, 4)
    train_rows = ~cand_dev_mask
    dev_qs = np.where(q_dev_mask)[0]

    def choose_C(fit_predict_proba, Cs):
        best = None
        for C in Cs:
            s_dev = fit_predict_proba(C)
            picks = pick_by_scores(s_dev, tr['parsed'], dev_qs)
            acc = np.mean([tr['labels'][i, picks[i]] for i in dev_qs])
            if best is None or acc > best[0]:
                best = (acc, C)
        return best[1]

    def report(name, picks, **extra):
        r = eval_picks(picks, va['labels'], va['parsed'])
        r.update(extra)
        results[name] = r

    # --- TF-IDF (word 1-2 + char_wb 3-5) logistic ---
    def texts_of(split_d):
        return [row['question'] + ' ' + c['text']
                for row in split_d['bank'] for c in row['candidates']]

    vec_w = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=3,
                            max_features=2**16, sublinear_tf=True)
    vec_c = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=3,
                            max_features=2**16, sublinear_tf=True)
    X_tr_text = hstack([vec_w.fit_transform(texts_of(tr)), vec_c.fit_transform(texts_of(tr))]).tocsr()
    X_va_text = hstack([vec_w.transform(texts_of(va)), vec_c.transform(texts_of(va))]).tocsr()

    def tfidf_proba(C):
        clf = LogisticRegression(C=C, max_iter=2000)
        clf.fit(X_tr_text[train_rows], y_tr[train_rows])
        return clf.predict_proba(X_tr_text[cand_dev_mask])[:, 1]

    C_tfidf = choose_C(tfidf_proba, (0.1, 1.0, 10.0))
    clf_tfidf = LogisticRegression(C=C_tfidf, max_iter=2000)
    clf_tfidf.fit(X_tr_text, y_tr)
    report('tfidf_logistic',
           pick_by_scores(clf_tfidf.predict_proba(X_va_text)[:, 1], va['parsed']),
           C=C_tfidf)

    # --- dense-input models (embedding / features / both) ---
    X_tr_emb = tr['emb'].reshape(n_tr * 4, -1)
    X_va_emb = va['emb'].reshape(n_va * 4, -1)
    F_tr = hand_features(tr['bank'], tr['parsed'])
    F_va = hand_features(va['bank'], va['parsed'])

    def dense_run(name, X_tr, X_va, Cs):
        sc = StandardScaler().fit(X_tr[train_rows])
        Xs_tr, Xs_va = sc.transform(X_tr), sc.transform(X_va)

        def proba(C):
            clf = LogisticRegression(C=C, max_iter=2000)
            clf.fit(Xs_tr[train_rows], y_tr[train_rows])
            return clf.predict_proba(Xs_tr[cand_dev_mask])[:, 1]

        C = choose_C(proba, Cs)
        clf = LogisticRegression(C=C, max_iter=2000)
        clf.fit(Xs_tr, y_tr)
        report(name, pick_by_scores(clf.predict_proba(Xs_va)[:, 1], va['parsed']), C=C)

    dense_run('embedding_logistic', X_tr_emb, X_va_emb, (0.01, 0.1, 1.0))
    dense_run('features_logistic', F_tr, F_va, (0.01, 0.1, 1.0))
    dense_run('emb_plus_features_logistic', np.hstack([X_tr_emb, F_tr]), np.hstack([X_va_emb, F_va]),
              (0.01, 0.1, 1.0))

    out = Path(__file__).resolve().parent / 'runs' / 'baselines.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
