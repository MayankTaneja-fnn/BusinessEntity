# Business Entity Resolution — Amazon ML Challenge 2026

## Overview

This solution implements a multi-stage entity resolution pipeline to match business records across three independent data sources.

## Architecture

```
Stage 0: Data exploration & validation
Stage 1: Text normalization (name + address normalizers)
Stage 2: Blocking / candidate generation (TF-IDF + Token + Sorted-Neighborhood)
Stage 3: Feature engineering (string similarity suite)
Stage 4: Model training (LightGBM gradient-boosted trees)
Stage 5: Conflict resolution & threshold optimization (F_0.5 sweep)
Stage 6: Inference & output generation
```

## Quick Start — Reproduce End-to-End

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full pipeline (from student_resource/ directory)
cd <path_to_student_resource>
python code/business_entity_resolution/src/run_pipeline.py \
    --base-dir . \
    --output-dir output \
    --top-k 20 \
    --neg-ratio 3 \
    --max-pairs 5000000

# 3. Output files will be in output/
#    - output/matching_results.tsv  (final matches, scored on leaderboard)
#    - output/candidate_pairs.tsv   (blocking candidate set)
```

## Running Individual Stages

```bash
# Blocking only (train)
python src/blocking.py --mode train --base-dir . --output-dir output --top-k 20

# Training only (requires blocking output)
python src/train.py --base-dir . --output-dir output --neg-ratio 3

# Inference only (requires trained model)
python src/infer.py --base-dir . --output-dir output --top-k 20

# Evaluate predictions against ground truth
python src/evaluate.py output/matching_results.tsv dataset/train/train_ground_truth.tsv

# Validate submission format
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Code Structure

```
src/
├── run_pipeline.py   # Main orchestrator — runs all stages
├── normalize.py      # Text normalization (names + addresses)
├── blocking.py       # Candidate generation (TF-IDF + Token + Sorted-Neighborhood)
├── features.py       # Pairwise feature engineering (string similarities)
├── train.py          # Model training (LightGBM) + threshold optimization
├── infer.py          # Test set inference + conflict resolution + output
└── evaluate.py       # F_0.5 evaluation metric
```

## Key Design Decisions

1. **Blocking**: Union of 3 strategies for high recall ceiling (>98%)
   - TF-IDF char n-gram (3-5) nearest neighbors (captures fuzzy matches)
   - Token blocking with inverted index (captures exact token overlaps)
   - Sorted-neighborhood on name prefix (catches near-identical names)

2. **Features**: 18 string similarity features covering name, address, and country
   - All distance functions implemented from scratch (no external deps)
   - Country treated as open-ended string (same_country boolean only)

3. **Model**: LightGBM GBDT classifier (MIT licensed, <1M params)
   - Calibrated probabilities for meaningful threshold sweep
   - F_0.5 optimized threshold (not default 0.5)

4. **Conflict Resolution**: Each S2/S3 record assigned to at most one S1 entity
   - Prevents the same record from being claimed by multiple entities
   - Keeps only the highest-scoring link per S2/S3 record

## Requirements

- Python 3.8+
- See requirements.txt for pinned dependencies
- No external API calls or internet access required
