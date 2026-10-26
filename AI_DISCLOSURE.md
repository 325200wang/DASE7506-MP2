# AI and reuse disclosure

## What was used

AI coding assistance (an agentic coding assistant) was used throughout this project for:

- implementing the training, evaluation and analysis scripts in `code/`;
- orchestrating experiments on a local GPU and a remote RTX 3070 machine;
- drafting the report text and the figures.

## What was checked by the author

- Every number in the report was recomputed from recorded outputs and cross-checked against
  the official scorer's JSON in `records/official/`.
- The method-selection protocol was fixed before the test split was scored, and the report
  states explicitly which choices were made on validation and which could not be.
- The frozen checkpoint in `code/artifacts/efull/` is the one that produced the reported
  scores; SHA-256 hashes are listed in `README.md`.

## Reuse

- Course-supplied files (`guide/GUIDE.md`, `code/common.py`, `code/predict.py`,
  `code/score_predictions.py`, `code/check_data.py`, `code/data/`) are unchanged.
- `code/features.py`, `code/models.py`, `code/symverify.py`, `code/exp_baselines.py`,
  `code/full_features.py`, `code/qctext_metrics.py`, `code/train_efull.py` and
  `code/submission.py` were written for this project.
- No external datasets and no pretrained model weights are used. All learned parameters are
  trained from the supplied training split only.
- Questions and references derive from GSM8K (MIT licence). Candidate solutions and the fixed
  1024-d representations were prepared by the course with Qwen3-0.6B. `data/LICENSE.txt` and
  `data/NOTICE.txt` are retained.

## Negative results that were influenced by prior exploration

The report lists sixteen rejected routes. Two of them were built by importing feature code
from an earlier, independent run of the same assignment on another machine, purely as a
measurement of how much those feature families are worth; the numbers are reported in
Table 5 and Table 3 of the report, and **none of that code or any weight from it is part of
the submitted method**. The submitted checkpoint, its inputs and its decision rule are
entirely this project's own.
