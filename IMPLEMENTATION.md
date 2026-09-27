# Implementation Overview

## Approach & Problem Statement

The **ML Challenge 2026 – Entity Resolution** requires matching each entity from **Source‑1** to its correct counterpart(s) in **Source‑2** and **Source‑3**.  
The dataset contains millions of possible pairs, making a naïve exhaustive comparison infeasible.  
Our solution follows a classic **blocking → feature engineering → machine‑learning ranking** pipeline:

1. **Blocking** drastically reduces the candidate space using hash‑based and similarity heuristics.
2. **Feature engineering** creates dense and sparse descriptors (string similarity, numeric differences, embedding similarity) for every remaining candidate pair.
3. **Model training** (gradient‑boosted trees via XGBoost/LightGBM) learns to rank true matches higher than false ones.
4. **Inference** scores the candidate pairs and selects matches according to a calibrated threshold.

The implementation is pure **Python 3.8+**, leveraging **pandas** for data handling, **scikit‑learn** utilities, and **XGBoost/LightGBM** for the ranking model. All steps are orchestrated by lightweight scripts in the `code/business_entity_resolution` package.

---

## Project Purpose
The repository implements a solution for the **ML Challenge 2026 – Entity Resolution**. The task is to match entities from **Source‑1** to candidate entities in **Source‑2** and **Source‑3** using a machine‑learning pipeline. The final submission consists of two TSV files:
- `output/matching_results.tsv` (required) – final matches submitted for scoring.
- `output/candidate_pairs.tsv` (optional) – candidate pairs generated during the blocking stage.

---

## High‑Level Architecture & Data Flow
```
[Dataset] ──► data.py ──► blocking.py ──► features*.py ──► train.py / run_*.py ──► infer*.py ──► evaluate.py ──► output/*.tsv
      │                       │                         │                │                 │
      ▼                       ▼                         ▼                ▼                 ▼
  CSV/TSV files          candidate pairs            model            predictions        validation
```
1. **Data Loading (`data.py`)** – reads the raw TSVs from `dataset/` (train, test, validation) and provides convenient Pandas/Numpy structures.
2. **Blocking (`blocking.py`)** – creates a reduced candidate set for each Source‑1 entity, drastically cutting the search space for the downstream model.
3. **Feature Extraction (`features.py`, `features_fast.py`)** – computes dense and sparse features for every (Source‑1, candidate) pair. The "fast" variant uses pre‑computed vectors for rapid experimentation.
4. **Model Training (`train.py`)** – trains a gradient‑boosted or neural model on the engineered features. Hyper‑parameters are logged; the resulting model is serialised to `code/business_entity_resolution/model.pkl`.
5. **Pipeline Execution (`run_pipeline.py`, `run_fast.py`, `run_explore.py`, `run_sampled.py`)** – orchestrates the full end‑to‑end flow: blocking → feature creation → model inference → post‑processing. Different scripts target exploration, fast prototyping, or sub‑sampled runs.
6. **Inference (`infer.py`, `infer_fast.py`)** – loads the trained model and scores candidate pairs, producing a ranked list of matches.
7. **Evaluation (`evaluate.py`)** – calculates the official challenge metric on a validation split, enabling iterative improvements.
8. **Normalization (`normalize.py`)** – utility to normalise IDs, strings and numeric features consistently across the pipeline.
9. **Submission Validation (`utils/validate_submission.py`)** – a stand‑alone validator that checks the final TSVs against the challenge’s formatting rules before uploading.

---

## Directory Overview
| Directory | Key Files | Role |
|-----------|-----------|------|
| `README.md` | – | Challenge description, data layout, submission format. |
| `Documentation_template.md` | – | Skeleton for a detailed methods write‑up (to be filled by the participant). |
| `utils/` | `validate_submission.py` | CLI validator; does not touch training data. |
| `code/business_entity_resolution/` | `__init__.py`, `blocking.py`, `data.py`, `features*.py`, `train.py`, `run_*.py`, `infer*.py`, `evaluate.py`, `normalize.py` | Core Python package implementing the full pipeline. |
| `dataset/` | `train/`, `test/`, `validation/` (TSV files) | Raw challenge data. |
| `output/` | `matching_results.tsv`, `candidate_pairs.tsv` (currently empty) | Generated submission files. |

---

## Current Project Status (as of 2026‑09‑27)
- **Code base** is complete and organized as described above.
- **`output/` directory** is empty; the pipeline has not yet produced `matching_results.tsv`.
- An **`agy` process** has been running in the repository root for ~17 hours (`agy` – the Antigravity interactive agent). This process is typically used to drive the pipeline (e.g., executing `run_pipeline.py`, training, or debugging). Its exact logs are not exposed here, but the long‑running duration suggests a training or exploration step is in progress.
- The **Git history** shows recent `git add .` and `git push -u origin main`, indicating the latest code has been committed and pushed.
- **Validation script** (`utils/validate_submission.py`) is ready; it can be run once the TSVs appear.
- No generated model artefacts (`model.pkl`) are present, implying training has either not completed or the artefacts are stored elsewhere (e.g., a hidden `models/` folder not listed).

**Next steps to reach a complete submission**:
1. Ensure the `agy` session finishes training and writes the model file.
2. Run one of the `run_*.py` scripts (typically `run_pipeline.py`) to generate the candidate pairs and final matches.
3. Verify the generated TSVs with `python utils/validate_submission.py …`.
4. Fill in `Documentation_template.md` with methodology, hyper‑parameters, and results.
5. Commit the updated files and push the final repository.

---

## How Each File Contributes to the Flow
- **`__init__.py`** – Marks the directory as a Python package and exposes public symbols.
- **`blocking.py`** – Implements heuristic or learned blocking (e.g., locality‑sensitive hashing) to prune the candidate space.
- **`data.py`** – Centralises reading/writing of TSVs, handling missing values and type casting.
- **`features.py` / `features_fast.py`** – Generate rich feature vectors (string similarity, numeric differences, embeddings). The fast version re‑uses cached vectors for rapid prototyping.
- **`train.py`** – Sets up the training loop, parses command‑line args, logs metrics, and serialises the model.
- **`run_pipeline.py`** – Calls `blocking`, `features`, then `train` and finally `infer` in a single command; the entry point for end‑to‑end execution.
- **`run_fast.py`, `run_explore.py`, `run_sampled.py`** – Variants of the pipeline for quick experiments, hyper‑parameter sweeps, or training on a subset of the data.
- **`infer.py` / `infer_fast.py`** – Load the trained model and output ranked matches to `output/matching_results.tsv`.
- **`evaluate.py`** – Compare predictions against a hold‑out validation set using the official scoring script.
- **`normalize.py`** – Guarantees consistent preprocessing (e.g., lower‑casing, Unicode normalisation) across all stages.
- **`utils/validate_submission.py`** – Stand‑alone CLI tool that checks formatting, duplicate rows, ID existence, and optional candidate‑pair consistency.

---

*This document provides a comprehensive description of what has been built, how the components interact, and the current development state. It can be used as the basis for the final `implementation.md` deliverable.*
