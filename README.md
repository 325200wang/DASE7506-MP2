# MP2 — Selecting Reliable LLM Reasoning

**Frozen method:** five-fold set-attention ensemble over frozen trace embeddings, with
answer-group aggregation at decision time.

| Split      | Correct       | Accuracy              |
| ---------- | -------------:| ---------------------:|
| **test**   | **396 / 467** | **84.79657387580299** |
| validation | 746 / 798     | 93.48370927318295     |

**Inference assets:** 4,104,689 bytes = **3.91 MiB** uncompressed (limit 16 MiB).
**Prediction + scoring:** **4.0–5.7 s** for the complete 467-question test bank with four
threads (limit 30 s), measured on an AMD Ryzen 7 8845HS.

Report: [`report/MP2-report.pdf`](report/MP2-report.pdf) (7 pages).

## Repository layout

This is the supplied course package with the method added; `guide/`, `code/common.py`,
`code/predict.py`, `code/score_predictions.py`, `code/check_data.py` and `code/data/` are
unchanged.

```
.
├── guide/GUIDE.md                  supplied
├── code/
│   ├── submission.py               our method — predict(bank, arrays, checkpoint)
│   ├── features.py                 our 29 reference-free text features
│   ├── models.py                   our set-attention / ResMLP / DeepSets scorers
│   ├── symverify.py                our lightweight symbolic trace checker
│   ├── train_efull.py              our training script (numpy + torch only)
│   ├── full_features.py            our quantity / repair features (ablation)
│   ├── qctext_metrics.py           our evaluation metrics (trap definition, comp4)
│   ├── artifacts/efull/*.pt        the five trained members (3.91 MiB)
│   ├── data/                       supplied data, unchanged
│   └── …                           supplied runner, scorer and data tools
├── report/                         report PDFs, sources, figures, figure script
└── records/                        official scores, ablations, sweeps, analysis
```

## Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r code/requirements.txt      # numpy (supplied pin)
pip install torch==2.13.0                 # CPU build is sufficient
```

## Reproduce the score without retraining

```bash
cd code
python check_data.py

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python predict.py --method submission.py --checkpoint artifacts/efull \
  --split test --output runs/test-predictions.json
python score_predictions.py --split test \
  --predictions runs/test-predictions.json --output runs/test-score.json
```

Expected `"correct": 396, "n": 467, "accuracy": 84.79657387580299`; use
`--split validation` for `746 / 798 = 93.48370927318295`. The recorded outputs are in
`records/official/`.

`submission.py` uses only the `bank` and `arrays` passed to `predict()`, opens no reference
file and makes no assumption about the number of questions, so the same frozen method runs
unchanged on an unseen audit bank.

## Train

```bash
cd code
for f in 0 1 2 3 4; do
  python train_efull.py --fold $f --seed 37 --epochs 15 --batch 64 --lr 1e-4 \
    --dropout 0.2 --width 128 --layers 2 --heads 4 \
    --feature-dim 22 --feature-mode hand --tag efull_f${f}_s37
done
```

Weights land in `code/runs/efull/`. About 8 s per member on an RTX 3070 Laptop GPU, about
80 s per member on the CPU used for the ablations. Fold splits come from
`numpy.random.default_rng(13)`; the model seed is `torch.manual_seed(37)`.

Retraining reproduces the recipe but **not the shipped weights bit for bit**: the members were
trained on a GPU, and a CPU run differs in floating-point reduction order. A CPU retrain of
fold 0 scores 863/951 on its held-out fold (shipped member: 860/951) and 392/467 on test
(shipped ensemble: 396/467). Reproduce the reported score from `code/artifacts/efull/`, not
from a retrain.

**Checkpoint identity** (`code/artifacts/efull/`):

| File              | Bytes   | SHA-256                                                            |
| ----------------- | -------:| ------------------------------------------------------------------ |
| `efull_f0_s37.pt` | 820,925 | `810201776e0ca0b7c90f9b074b5093beade0436a23948c938f97b043be861cd0` |
| `efull_f1_s37.pt` | 820,925 | `bad849712771ccba8df5e011a8a419a257b19559724f7978c3a758ca0f254643` |
| `efull_f2_s37.pt` | 820,989 | `b08ad13384729a33315bb01776ebf9f7cbde2decb4fac56e9d3df0373c827b70` |
| `efull_f3_s37.pt` | 820,925 | `c1dadccaf63fd3efb6f78c00a7ebb35869d64edee3740c6d863340e9391ea840` |
| `efull_f4_s37.pt` | 820,925 | `6f7585613221dc7e8e9d52b1b4a2cefa9c4c75f530aec5feb5db4b02b37e1193` |

## Method in one paragraph

Each candidate is represented by its 1024-d frozen trace embedding concatenated with 22
reference-free text features (parseability, three consensus statistics, length and digit
density, formatting flags, equation re-checking counts, question-number coverage, four
chain-consistency scalars). A permutation-equivariant set-attention network (width 128, two
layers, four heads, GELU, dropout 0.2) maps the four candidates to four logits. Five members
with disjoint held-out folds are averaged in probability space. Candidates stating the same
parsed final answer are then partially pooled, `score_j = (1-a) p_j + a * mean(group)`, with
`a = 0.6`, and the argmax over parseable candidates is returned. The pooling preserves
within-group order and changes only the comparison between answer groups. `a` was fixed on
validation before test was scored, over the grid `{0, 0.1, ..., 1.0}`: every value in
`[0.4, 0.9]` ties the validation maximum (746/798), and the tie-break is highest validation
trap accuracy (36/60, from `a >= 0.6`), then the smallest `a`, giving 0.6. The full curve is in
`records/aggregation_curve.json`.

**Submitted code version.** Tag `v1`. The checkpoint in `code/artifacts/efull/` is the one that
produced the reported scores; its hashes are in the table above.

## Records

| Path                           | Contents                                                          |
| ------------------------------ | ----------------------------------------------------------------- |
| `records/official/`            | official scorer output for both splits, plus the prediction files |
| `records/ablation_table.json`  | input and architecture ablation (Table 3 of the report)           |
| `records/member_analysis.json` | per-member scores, pairwise agreement, oracle, calibration        |
| `records/text_baselines.json`  | TF-IDF text-only baselines (full trace vs final answer)           |
| `records/sweep/`               | 18-configuration hyper-parameter sweep with per-epoch curves      |
| `records/fig_data.json`        | numbers behind Figures 1–4                                        |

## AI and reuse

See [`AI_DISCLOSURE.md`](AI_DISCLOSURE.md). No external datasets, no pretrained weights and
no reference-answer lookups are used at inference. Questions and references derive from GSM8K
(MIT); candidate traces and fixed embeddings were prepared with Qwen3-0.6B. The supplied
`data/LICENSE.txt` and `data/NOTICE.txt` are kept.
