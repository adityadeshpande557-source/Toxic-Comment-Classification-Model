# Toxic Comment Classification: COEP DSAI Club Induction 2026

## Name & Branch
- **Name:** Aditya Deshpande
- **Branch:** CSE

## Model Architecture & Core Idea
- **Task:** multi-label prediction of `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`; metric is the mean of the six ROC AUCs.
- **Data used:** only `competition_train.csv` for training; test comments were used only to produce predictions.
- **Preprocessing:**
  - Lowercasing and accent stripping only. Punctuation, capitals and repeated characters are kept because they carry toxicity signal.
  - Missing comments are treated as empty text. No rows are dropped.
- **Features (TF-IDF, sparse, float32):**
  - Word 1-2-grams: `min_df=3`, 50,000 features, `sublinear_tf=True`.
  - Character n-grams: `char_wb` 2-5, `min_df=3`, 50,000 features, `sublinear_tf=True`.
  - Both blocks are stacked into one 100,000-feature vector per comment.
- **Model:** logistic regression (`liblinear`, L2) wrapped in `OneVsRestClassifier`.
  - One independent binary classifier per label, so one comment can carry several labels at once.
  - `predict_proba` gives six probabilities per comment, which is what ROC AUC needs (no 0/1 thresholding).
- **Validation:** stratified 80/20 split, seed 42, stratified on "any toxic label".
  - TF-IDF is fitted on the training split only, so no leakage.
  - Models are compared by mean validation ROC AUC on the same split.
- **Experiments run:**
  - Word TF-IDF baseline (`C=4`).
  - Word + char TF-IDF (`C=4`).
  - Word + char TF-IDF with `class_weight="balanced"`.
  - Word + char TF-IDF with `C=1` and `C=10`.
- **Final model:** the best validation configuration, refit on all labelled training data, then used to predict the test set.
  - Recorded best validation mean ROC AUC (Colab run): about 0.98550 with word + char TF-IDF, `C=1`.
  - No test ROC AUC is claimed, since test labels are hidden.
- **Parameter limit (< 1,000,000):**
  - 6 x 100,000 coefficients + 6 intercepts = 600,006 classifier parameters.
  - The 100,000 fitted IDF weights are also counted, giving 700,006 in total.
  - The notebook asserts this is below 1,000,000 on the fitted model.

## Unique / Differentiating Ideas
- **Word + character n-grams together:** `char_wb` n-grams catch misspellings and obfuscation (e.g. `idi0t`) that word features miss.
- **Light cleaning on purpose:** I kept punctuation and capitalisation instead of stripping them, since they signal toxicity.
- **Measured, not assumed:** the notebook tests class weighting and a `C` sweep, and keeps a change only if validation AUC supports it.
- **Conservative parameter accounting:** IDF weights are counted as learned parameters, and the 1M limit is asserted in code.
- **Programmatic reporting:** comparison tables, per-label AUCs, AUC differences and the README numbers are generated from the actual run.
- **Strict format checks:** the submission is validated for column order, row count, matching IDs, finite values and range [0, 1], with no index column.
- **Rule compliance:** no external datasets, pretrained models, LLM calls, or answer keys.
