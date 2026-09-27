"""
infer.py — Inference pipeline for test set.

Stages:
1. Load test data and normalize
2. Run blocking to generate candidates
3. Compute features for all candidate pairs
4. Score with trained model
5. Apply conflict resolution (one S2/S3 record -> at most one S1)
6. Apply threshold
7. Write output files (matching_results.tsv, candidate_pairs.tsv)
"""

import os
import sys
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from normalize import normalize_name, normalize_address
from features import compute_features_batch
from blocking import (
    load_source, add_normalized_fields,
    tfidf_blocking, token_blocking, sorted_neighborhood_blocking,
    union_candidates, save_candidates_tsv,
)


def load_model(model_path: str) -> dict:
    """Load trained model and config."""
    with open(model_path, "rb") as f:
        config = pickle.load(f)
    print(f"Loaded model from {model_path}")
    print(f"  Threshold: {config['threshold']:.3f}")
    print(f"  Val F_0.5: {config['val_f05']:.4f}")
    print(f"  Features: {len(config['feature_cols'])} columns")
    return config


def create_inference_pairs(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    candidates: dict,
) -> pd.DataFrame:
    """
    Create pair DataFrame for all candidate pairs for scoring.
    """
    print("Creating inference pairs...", flush=True)
    t0 = time.time()

    s1_lookup = s1_df.set_index("entity_id").to_dict("index")
    s23_lookup = s23_df.set_index("entity_id").to_dict("index")

    pairs = []
    skipped = 0
    for s1_id, cand_ids in candidates.items():
        if s1_id not in s1_lookup:
            continue
        s1_info = s1_lookup[s1_id]
        for cand_id in cand_ids:
            if cand_id not in s23_lookup:
                skipped += 1
                continue
            s23_info = s23_lookup[cand_id]
            pairs.append({
                "s1_id": s1_id,
                "s23_id": cand_id,
                "name1": s1_info.get("business_name", ""),
                "name2": s23_info.get("business_name", ""),
                "addr1": s1_info.get("business_address", ""),
                "addr2": s23_info.get("business_address", ""),
                "country1": s1_info.get("country", ""),
                "country2": s23_info.get("country", ""),
                "name1_norm": s1_info.get("name_norm", ""),
                "name2_norm": s23_info.get("name_norm", ""),
                "addr1_norm": s1_info.get("addr_norm", ""),
                "addr2_norm": s23_info.get("addr_norm", ""),
            })

    pairs_df = pd.DataFrame(pairs)
    print(f"  Created {len(pairs_df)} inference pairs (skipped {skipped} unknown candidates)")
    print(f"  Done in {time.time()-t0:.1f}s", flush=True)
    return pairs_df


def score_pairs(pairs_df: pd.DataFrame, config: dict, batch_size: int = 100000) -> np.ndarray:
    """
    Compute features and score all pairs with the trained model.
    """
    print(f"Scoring {len(pairs_df)} pairs...", flush=True)
    t0 = time.time()

    model = config["model"]
    feature_cols = config["feature_cols"]

    all_scores = []
    n_batches = (len(pairs_df) + batch_size - 1) // batch_size

    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, len(pairs_df))
        batch = pairs_df.iloc[start:end]

        # Compute features
        features = compute_features_batch(batch)

        # Ensure all feature columns exist
        for col in feature_cols:
            if col not in features.columns:
                features[col] = 0.0

        X = features[feature_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)

        # Score with model
        try:
            # LightGBM model
            scores = model.predict(X)
        except Exception:
            try:
                # XGBoost model
                import xgboost as xgb
                dtest = xgb.DMatrix(X, feature_names=feature_cols)
                scores = model.predict(dtest)
            except Exception:
                # sklearn model
                scores = model.predict_proba(X)[:, 1]

        all_scores.append(scores)

        if (i + 1) % 10 == 0 or i == n_batches - 1:
            print(f"  Batch {i+1}/{n_batches}, elapsed {time.time()-t0:.1f}s", flush=True)

    scores = np.concatenate(all_scores)
    print(f"  Scoring done in {time.time()-t0:.1f}s", flush=True)
    return scores


def conflict_resolution(
    pairs_df: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    all_s1_ids: list,
) -> dict:
    """
    Apply conflict resolution:
    1. For each S2/S3 candidate, keep only the highest-scoring S1 link
    2. Apply threshold
    
    This prevents the same S2/S3 record from being claimed by multiple S1 entities.
    
    Returns: {s1_id: set(matched_ids)}
    """
    print(f"Applying conflict resolution (threshold={threshold:.3f})...", flush=True)

    s1_ids = pairs_df["s1_id"].values
    s23_ids = pairs_df["s23_id"].values

    # Step 1: For each S2/S3 record, find the best S1 match
    s23_best = {}  # s23_id -> (best_s1_id, best_score)
    for idx in range(len(scores)):
        score = scores[idx]
        if score < threshold:
            continue
        s1_id = s1_ids[idx]
        s23_id = s23_ids[idx]

        if s23_id not in s23_best or score > s23_best[s23_id][1]:
            s23_best[s23_id] = (s1_id, score)

    # Step 2: Build final predictions
    predictions = defaultdict(set)
    for s23_id, (s1_id, score) in s23_best.items():
        predictions[s1_id].add(s23_id)

    # Step 3: Ensure all S1 entities have a row
    for s1_id in all_s1_ids:
        if s1_id not in predictions:
            predictions[s1_id] = set()

    n_matched = sum(1 for v in predictions.values() if v)
    n_singleton = sum(1 for v in predictions.values() if not v)
    total_matches = sum(len(v) for v in predictions.values())

    print(f"  {n_matched} entities with matches, {n_singleton} singletons")
    print(f"  Total match links: {total_matches}")
    print(f"  Avg matches per matched entity: {total_matches/max(n_matched,1):.2f}")
    return dict(predictions)


def write_matching_results(predictions: dict, path: str):
    """Write matching_results.tsv in submission format."""
    rows = []
    for s1_id in sorted(predictions.keys()):
        matched = predictions[s1_id]
        matched_str = ",".join(sorted(matched)) if matched else ""
        rows.append(f"{s1_id}\t{matched_str}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        f.write("\n".join(rows) + "\n")
    print(f"  Wrote {len(rows)} rows to {path}")


def run_inference(
    base_dir: str = ".",
    output_dir: str = "output",
    top_k: int = 20,
):
    """
    Full inference pipeline on test set.
    """
    print(f"\n{'='*60}")
    print("INFERENCE PIPELINE")
    print(f"{'='*60}\n")

    t0_total = time.time()

    # Load test data
    s1_path = os.path.join(base_dir, "dataset/test/test_source1.tsv")
    s2_path = os.path.join(base_dir, "dataset/test/test_source2.tsv")
    s3_path = os.path.join(base_dir, "dataset/test/test_source3.tsv")

    print("Loading test data...", flush=True)
    s1 = load_source(s1_path)
    s2 = load_source(s2_path)
    s3 = load_source(s3_path)
    print(f"  S1: {len(s1)}, S2: {len(s2)}, S3: {len(s3)}")

    s23 = pd.concat([s2, s3], ignore_index=True)
    print(f"  S2+S3 combined: {len(s23)}")

    # Normalize
    print("\nNormalizing...", flush=True)
    s1 = add_normalized_fields(s1)
    s23 = add_normalized_fields(s23)

    # Save normalized data
    norm_path = os.path.join(output_dir, "test_normalized.pkl")
    with open(norm_path, "wb") as f:
        pickle.dump({"s1": s1, "s23": s23}, f, protocol=4)

    # Blocking
    print("\nRunning blocking...", flush=True)
    cands_tfidf = tfidf_blocking(s1, s23, top_k=top_k)
    cands_token = token_blocking(s1, s23, max_block_size=1000, min_token_len=4)
    cands_sorted = sorted_neighborhood_blocking(s1, s23, window_size=3)
    candidates = union_candidates(cands_tfidf, cands_token, cands_sorted)

    avg_cands = np.mean([len(v) for v in candidates.values()])
    print(f"  Combined: {len(candidates)} S1 entities, avg {avg_cands:.1f} candidates each")

    all_s1_ids = s1["entity_id"].tolist()

    # Ensure all S1 entities have candidates entry
    for s1_id in all_s1_ids:
        if s1_id not in candidates:
            candidates[s1_id] = set()

    # Save candidate_pairs.tsv
    os.makedirs(output_dir, exist_ok=True)
    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")
    save_candidates_tsv(candidates, cand_path, all_s1_ids)

    # Load model
    model_path = os.path.join(output_dir, "model.pkl")
    config = load_model(model_path)

    # Create inference pairs
    pairs_df = create_inference_pairs(s1, s23, candidates)

    if len(pairs_df) == 0:
        print("WARNING: No candidate pairs to score!")
        predictions = {s1_id: set() for s1_id in all_s1_ids}
    else:
        # Score pairs
        scores = score_pairs(pairs_df, config)

        # Conflict resolution + thresholding
        predictions = conflict_resolution(pairs_df, scores, config["threshold"], all_s1_ids)

    # Write matching_results.tsv
    match_path = os.path.join(output_dir, "matching_results.tsv")
    write_matching_results(predictions, match_path)

    print(f"\nTotal inference time: {time.time()-t0_total:.1f}s")
    print(f"\nOutput files:")
    print(f"  {match_path}")
    print(f"  {cand_path}")

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on test set")
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    run_inference(args.base_dir, args.output_dir, args.top_k)
