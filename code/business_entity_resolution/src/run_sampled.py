"""
run_sampled.py — Run the pipeline on a representative sample first.

Given the massive scale (2.2M S1 x 10M S2+S3 = 22 trillion potential pairs),
we first validate on a sample, then scale up.

This script:
1. Samples N S1 entities and their corresponding S2/S3 candidates
2. Runs the full pipeline on this sample
3. Reports metrics
4. Then runs full inference on the complete test set
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
from normalize import normalize_name, normalize_address, normalize_combined
from blocking import (
    load_source, add_normalized_fields,
    tfidf_blocking, token_blocking, sorted_neighborhood_blocking,
    union_candidates, measure_recall_ceiling, save_candidates_tsv,
)
from features import compute_features_batch
from evaluate import compute_f05_per_entity, compute_macro_f05


def sample_data(s1_df, s23_df, gt_dict, n_s1=50000, seed=42):
    """Sample n_s1 S1 entities and all their potential S2/S3 candidates."""
    rng = np.random.RandomState(seed)
    
    # Sample S1 entities, ensuring we get a mix of singletons and non-singletons
    s1_ids = s1_df["entity_id"].values
    rng.shuffle(s1_ids)
    sampled_s1 = set(s1_ids[:n_s1])
    
    s1_sample = s1_df[s1_df["entity_id"].isin(sampled_s1)].copy()
    
    # Get all true match IDs for sampled S1 entities
    true_s23_ids = set()
    for s1_id in sampled_s1:
        if s1_id in gt_dict:
            true_s23_ids.update(gt_dict[s1_id])
    
    # Sample S2+S3: include all true matches plus random others
    s23_ids_set = set(s23_df["entity_id"].values)
    other_ids = list(s23_ids_set - true_s23_ids)
    rng.shuffle(other_ids)
    # Take 5x the true matches as "noise" to make blocking realistic
    n_noise = min(len(other_ids), max(5 * len(true_s23_ids), 200000))
    sampled_s23_ids = true_s23_ids | set(other_ids[:n_noise])
    
    s23_sample = s23_df[s23_df["entity_id"].isin(sampled_s23_ids)].copy()
    
    # Sample ground truth
    gt_sample = {s1_id: gt_dict.get(s1_id, set()) for s1_id in sampled_s1}
    
    print(f"  Sampled S1: {len(s1_sample)}")
    print(f"  Sampled S23: {len(s23_sample)} ({len(true_s23_ids)} true matches + {n_noise} noise)")
    print(f"  Singleton rate in sample: {sum(1 for v in gt_sample.values() if not v)/len(gt_sample):.4f}")
    
    return s1_sample, s23_sample, gt_sample


def run_sampled_pipeline(
    base_dir=".",
    output_dir="output",
    n_s1_sample=50000,
    top_k=20,
    neg_ratio=3,
    max_pairs=2000000,
    seed=42,
):
    """Run the full pipeline on a sample for fast iteration."""
    t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("SAMPLED PIPELINE — FAST ITERATION MODE")
    print(f"{'='*60}\n")
    
    # ============================================================
    # Load data
    # ============================================================
    print("Loading training data...", flush=True)
    s1 = load_source(os.path.join(base_dir, "dataset/train/train_source1.tsv"))
    s2 = load_source(os.path.join(base_dir, "dataset/train/train_source2.tsv"))
    s3 = load_source(os.path.join(base_dir, "dataset/train/train_source3.tsv"))
    s23 = pd.concat([s2, s3], ignore_index=True)
    print(f"  Full data: S1={len(s1)}, S2={len(s2)}, S3={len(s3)}, S23={len(s23)}")
    
    # Load ground truth
    gt = pd.read_csv(os.path.join(base_dir, "dataset/train/train_ground_truth.tsv"), 
                     sep="\t", dtype=str, keep_default_na=False)
    gt_dict = {}
    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        matched = row["matched_entity_ids"]
        if not matched or matched.strip() == "":
            gt_dict[s1_id] = set()
        else:
            gt_dict[s1_id] = set(matched.strip().split(","))
    print(f"  Ground truth: {len(gt_dict)} S1 entities")
    
    # ============================================================
    # Sample
    # ============================================================
    print("\nSampling data for fast iteration...", flush=True)
    s1_sample, s23_sample, gt_sample = sample_data(s1, s23, gt_dict, n_s1=n_s1_sample, seed=seed)
    
    # ============================================================
    # Normalize
    # ============================================================
    print("\nNormalizing...", flush=True)
    s1_sample = add_normalized_fields(s1_sample)
    s23_sample = add_normalized_fields(s23_sample)
    
    # ============================================================
    # Blocking
    # ============================================================
    print("\nRunning blocking on sample...", flush=True)
    
    cands_tfidf = tfidf_blocking(s1_sample, s23_sample, top_k=top_k)
    cands_token = token_blocking(s1_sample, s23_sample, max_block_size=500, min_token_len=4)
    cands_sorted = sorted_neighborhood_blocking(s1_sample, s23_sample, window_size=3)
    
    candidates = union_candidates(cands_tfidf, cands_token, cands_sorted)
    
    avg_cands = np.mean([len(v) for v in candidates.values()])
    print(f"  Combined: {len(candidates)} S1 entities, avg {avg_cands:.1f} candidates")
    
    # Measure recall ceiling
    recall_ceiling = measure_recall_ceiling(candidates, gt_sample)
    
    if recall_ceiling < 0.95:
        print("⚠️  WARNING: Blocking recall ceiling is below 95%!")
        print("  Consider increasing top_k or adding more blocking strategies.")
    
    # ============================================================
    # Create training pairs
    # ============================================================
    print("\nCreating training pairs...", flush=True)
    rng = np.random.RandomState(seed)
    
    s1_lookup = s1_sample.set_index("entity_id").to_dict("index")
    s23_lookup = s23_sample.set_index("entity_id").to_dict("index")
    
    # Split S1 entities into train/val
    s1_ids = list(gt_sample.keys())
    rng.shuffle(s1_ids)
    n_val = int(len(s1_ids) * 0.15)
    val_s1 = set(s1_ids[:n_val])
    train_s1 = set(s1_ids[n_val:])
    
    print(f"  Train S1: {len(train_s1)}, Val S1: {len(val_s1)}")
    
    def make_pairs(s1_set, label_prefix=""):
        pairs = []
        total_pos = 0
        total_neg = 0
        for s1_id in s1_set:
            if s1_id not in s1_lookup:
                continue
            true_ids = gt_sample.get(s1_id, set())
            cand_ids = candidates.get(s1_id, set())
            if not cand_ids:
                continue
            s1_info = s1_lookup[s1_id]
            
            # Positives
            pos_ids = true_ids & cand_ids
            for match_id in pos_ids:
                if match_id not in s23_lookup:
                    continue
                s23_info = s23_lookup[match_id]
                pairs.append({
                    "s1_id": s1_id, "s23_id": match_id, "label": 1,
                    "name1_norm": s1_info.get("name_norm", ""),
                    "name2_norm": s23_info.get("name_norm", ""),
                    "addr1_norm": s1_info.get("addr_norm", ""),
                    "addr2_norm": s23_info.get("addr_norm", ""),
                    "country1": s1_info.get("country", ""),
                    "country2": s23_info.get("country", ""),
                })
                total_pos += 1
            
            # Negatives
            neg_ids = cand_ids - true_ids
            if neg_ids:
                n_neg = min(len(neg_ids), max(neg_ratio * len(pos_ids), 1))
                neg_sample = rng.choice(list(neg_ids), size=n_neg, replace=False)
                for neg_id in neg_sample:
                    if neg_id not in s23_lookup:
                        continue
                    s23_info = s23_lookup[neg_id]
                    pairs.append({
                        "s1_id": s1_id, "s23_id": neg_id, "label": 0,
                        "name1_norm": s1_info.get("name_norm", ""),
                        "name2_norm": s23_info.get("name_norm", ""),
                        "addr1_norm": s1_info.get("addr_norm", ""),
                        "addr2_norm": s23_info.get("addr_norm", ""),
                        "country1": s1_info.get("country", ""),
                        "country2": s23_info.get("country", ""),
                    })
                    total_neg += 1
        
        df = pd.DataFrame(pairs)
        print(f"  {label_prefix} pairs: {len(df)} ({total_pos} pos, {total_neg} neg)")
        return df
    
    train_pairs = make_pairs(train_s1, "Train")
    val_pairs = make_pairs(val_s1, "Val")
    
    # ============================================================
    # Feature engineering
    # ============================================================
    print("\nComputing features...", flush=True)
    t_feat = time.time()
    train_features = compute_features_batch(train_pairs)
    val_features = compute_features_batch(val_pairs)
    print(f"  Feature computation: {time.time()-t_feat:.1f}s")
    
    feature_cols = [c for c in train_features.columns]
    print(f"  Feature columns ({len(feature_cols)}): {feature_cols}")
    
    X_train = train_features.values.astype(np.float32)
    y_train = train_pairs["label"].values
    X_val = val_features.values.astype(np.float32)
    y_val = val_pairs["label"].values
    
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=1.0, neginf=0.0)
    
    print(f"\n  X_train: {X_train.shape}, pos rate: {y_train.mean():.3f}")
    print(f"  X_val: {X_val.shape}, pos rate: {y_val.mean():.3f}")
    
    # ============================================================
    # Train model
    # ============================================================
    print("\nTraining LightGBM...", flush=True)
    t_train = time.time()
    
    try:
        import lightgbm as lgb
        
        lgb_train = lgb.Dataset(X_train, y_train, feature_name=feature_cols)
        lgb_val_ds = lgb.Dataset(X_val, y_val, reference=lgb_train, feature_name=feature_cols)
        
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 50,
            "scale_pos_weight": (len(y_train) - y_train.sum()) / max(y_train.sum(), 1),
            "verbose": -1,
            "n_jobs": -1,
            "seed": seed,
        }
        
        model = lgb.train(
            params, lgb_train,
            num_boost_round=500,
            valid_sets=[lgb_val_ds],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30),
                lgb.log_evaluation(period=50),
            ],
        )
        
        val_scores = model.predict(X_val)
        
        # Feature importance
        importance = model.feature_importance(importance_type="gain")
        feat_imp = sorted(zip(feature_cols, importance), key=lambda x: -x[1])
        print("\nFeature importance (top 10):")
        for fname, imp in feat_imp[:10]:
            print(f"  {fname}: {imp:.1f}")
        
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        print("  LightGBM not available, using sklearn GBT...")
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=seed,
        )
        model.fit(X_train, y_train)
        val_scores = model.predict_proba(X_val)[:, 1]
    
    print(f"  Training done in {time.time()-t_train:.1f}s")
    
    # ============================================================
    # Threshold sweep
    # ============================================================
    print("\nSweeping threshold for F_0.5...", flush=True)
    
    val_ground_truth = {s1_id: gt_sample.get(s1_id, set()) for s1_id in val_s1}
    thresholds = np.arange(0.05, 0.95, 0.02)
    
    best_f05 = -1
    best_threshold = 0.5
    
    print(f"\n{'Threshold':>10} {'F_0.5':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 40)
    
    for thresh in thresholds:
        predictions = defaultdict(set)
        
        val_s1_ids = val_pairs["s1_id"].values
        val_s23_ids = val_pairs["s23_id"].values
        
        # Apply conflict resolution: each S23 gets assigned to highest-scoring S1
        s23_best = {}
        for idx in range(len(val_scores)):
            score = val_scores[idx]
            if score < thresh:
                continue
            s1_id = val_s1_ids[idx]
            s23_id = val_s23_ids[idx]
            if s23_id not in s23_best or score > s23_best[s23_id][1]:
                s23_best[s23_id] = (s1_id, score)
        
        for s23_id, (s1_id, score) in s23_best.items():
            predictions[s1_id].add(s23_id)
        
        for s1_id in val_s1:
            if s1_id not in predictions:
                predictions[s1_id] = set()
        
        f05, prec, rec, _ = compute_macro_f05(dict(predictions), val_ground_truth)
        
        if f05 > best_f05:
            best_f05 = f05
            best_threshold = thresh
            # Print only notable thresholds
            print(f"{thresh:>10.3f} {f05:>8.4f} {prec:>10.4f} {rec:>8.4f} <-- BEST")
        elif abs(thresh - 0.5) < 0.01 or thresh % 0.1 < 0.02:
            print(f"{thresh:>10.3f} {f05:>8.4f} {prec:>10.4f} {rec:>8.4f}")
    
    print(f"\n{'='*40}")
    print(f"BEST THRESHOLD: {best_threshold:.3f}")
    print(f"BEST F_0.5:     {best_f05:.4f}")
    print(f"{'='*40}")
    
    # ============================================================
    # Save model
    # ============================================================
    config = {
        "model": model,
        "feature_cols": feature_cols,
        "threshold": best_threshold,
        "val_f05": best_f05,
    }
    model_path = os.path.join(output_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(config, f, protocol=4)
    print(f"\nSaved model to {model_path}")
    
    # ============================================================
    # Print summary
    # ============================================================
    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print("METHODOLOGY SUMMARY")
    print(f"{'='*60}")
    print(f"Blocking strategy: TF-IDF (char 3-5gram, top-{top_k}) + Token blocking + Sorted-neighborhood")
    print(f"Blocking recall ceiling: {recall_ceiling:.4f}")
    print(f"Features used: {', '.join(feature_cols)}")
    print(f"Model: LightGBM GBDT (MIT license, <1M params)")
    print(f"Threshold: {best_threshold:.3f} (optimized for F_0.5)")
    print(f"Validation F_0.5: {best_f05:.4f}")
    print(f"Sample size: {n_s1_sample} S1 entities")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sampled pipeline")
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--n-s1", type=int, default=50000, help="Number of S1 entities to sample")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--neg-ratio", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    run_sampled_pipeline(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        n_s1_sample=args.n_s1,
        top_k=args.top_k,
        neg_ratio=args.neg_ratio,
        seed=args.seed,
    )
