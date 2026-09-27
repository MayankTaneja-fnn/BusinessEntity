"""
train.py — Train the matching model (LightGBM gradient-boosted tree classifier).

Stages:
1. Load normalized data and candidate sets from blocking stage
2. Create positive pairs from ground truth, negative pairs from blocked candidates
3. Compute features for all pairs
4. Train LightGBM with calibrated probabilities
5. Sweep threshold on validation split to maximize F_0.5
6. Save trained model and optimal threshold

Designed for massive scale: uses sampling for negative pairs to keep
training set manageable while maintaining hard negative quality.
"""

import os
import sys
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import GroupKFold
from sklearn.calibration import CalibratedClassifierCV

# Add parent to path
sys.path.insert(0, os.path.dirname(__file__))
from normalize import normalize_name, normalize_address
from features import compute_features_batch
from evaluate import compute_f05_per_entity, compute_macro_f05


def load_ground_truth(gt_path: str) -> dict:
    """Load ground truth into {s1_id: set(matched_ids)} dict."""
    gt = pd.read_csv(gt_path, sep="\t")
    truth = {}
    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        matched = row["matched_entity_ids"]
        if pd.isna(matched) or str(matched).strip() == "":
            truth[s1_id] = set()
        else:
            truth[s1_id] = set(str(matched).split(","))
    return truth


def create_training_pairs(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    candidates: dict,
    ground_truth: dict,
    neg_ratio: int = 3,
    max_pairs: int = 5000000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create labeled training pairs from ground truth and blocked candidates.
    
    For each S1 entity:
    - Positive pairs: all true matches that are also in the candidate set
    - Negative pairs: sampled from candidates that are NOT true matches (hard negatives)
    
    Args:
        neg_ratio: number of negative pairs per positive pair
        max_pairs: maximum total pairs to generate
    """
    print("Creating training pairs...", flush=True)
    t0 = time.time()
    rng = np.random.RandomState(seed)

    # Build lookup dicts for S1 and S23
    s1_lookup = s1_df.set_index("entity_id").to_dict("index")
    s23_lookup = s23_df.set_index("entity_id").to_dict("index")

    pairs = []
    total_pos = 0
    total_neg = 0

    s1_ids = list(ground_truth.keys())
    rng.shuffle(s1_ids)

    for s1_id in s1_ids:
        if s1_id not in s1_lookup:
            continue
        true_ids = ground_truth[s1_id]
        cand_ids = candidates.get(s1_id, set())

        if not cand_ids:
            continue

        s1_info = s1_lookup[s1_id]

        # Positive pairs: true matches that are in candidate set
        pos_ids = true_ids & cand_ids
        for match_id in pos_ids:
            if match_id not in s23_lookup:
                continue
            s23_info = s23_lookup[match_id]
            pairs.append({
                "s1_id": s1_id,
                "s23_id": match_id,
                "label": 1,
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
            total_pos += 1

        # Negative pairs: hard negatives from candidates minus true matches
        neg_ids = cand_ids - true_ids
        if neg_ids:
            n_neg = min(len(neg_ids), max(neg_ratio * len(pos_ids), 1))
            neg_sample = rng.choice(list(neg_ids), size=n_neg, replace=False)
            for neg_id in neg_sample:
                if neg_id not in s23_lookup:
                    continue
                s23_info = s23_lookup[neg_id]
                pairs.append({
                    "s1_id": s1_id,
                    "s23_id": neg_id,
                    "label": 0,
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
                total_neg += 1

        if len(pairs) >= max_pairs:
            break

    pairs_df = pd.DataFrame(pairs)
    print(f"  Created {len(pairs_df)} pairs: {total_pos} positive, {total_neg} negative")
    print(f"  Positive rate: {total_pos / len(pairs_df) * 100:.1f}%")
    print(f"  Done in {time.time()-t0:.1f}s", flush=True)
    return pairs_df


def train_val_split(
    pairs_df: pd.DataFrame,
    ground_truth: dict,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple:
    """
    Split pairs into train/val, ensuring no S1 entity leaks across splits.
    Uses GroupKFold on s1_id groups.
    """
    print("Splitting into train/val...", flush=True)
    rng = np.random.RandomState(seed)

    # Get unique S1 IDs and shuffle
    s1_ids = pairs_df["s1_id"].unique()
    rng.shuffle(s1_ids)

    # Simple hold-out split: first (1-val_fraction) for train, rest for val
    n_val = int(len(s1_ids) * val_fraction)
    val_s1 = set(s1_ids[:n_val])
    train_s1 = set(s1_ids[n_val:])

    train_mask = pairs_df["s1_id"].isin(train_s1)
    val_mask = pairs_df["s1_id"].isin(val_s1)

    train_df = pairs_df[train_mask].copy()
    val_df = pairs_df[val_mask].copy()

    print(f"  Train: {len(train_df)} pairs ({train_df.label.sum()} pos), {len(train_s1)} S1 entities")
    print(f"  Val: {len(val_df)} pairs ({val_df.label.sum()} pos), {len(val_s1)} S1 entities")
    return train_df, val_df, val_s1


def compute_features_for_pairs(pairs_df: pd.DataFrame, batch_size: int = 100000) -> pd.DataFrame:
    """Compute features for all pairs in batches."""
    print(f"Computing features for {len(pairs_df)} pairs...", flush=True)
    t0 = time.time()

    all_features = []
    n_batches = (len(pairs_df) + batch_size - 1) // batch_size

    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, len(pairs_df))
        batch = pairs_df.iloc[start:end]
        features = compute_features_batch(batch)
        all_features.append(features)
        if (i + 1) % 10 == 0 or i == n_batches - 1:
            print(f"  Batch {i+1}/{n_batches}, elapsed {time.time()-t0:.1f}s", flush=True)

    result = pd.concat(all_features, ignore_index=True)
    print(f"  Feature computation done in {time.time()-t0:.1f}s", flush=True)
    return result


def sweep_threshold(
    val_scores: np.ndarray,
    val_labels: np.ndarray,
    val_s1_ids: np.ndarray,
    val_s23_ids: np.ndarray,
    val_ground_truth: dict,
    thresholds: np.ndarray = None,
) -> tuple:
    """
    Sweep threshold on validation set and find the one maximizing F_0.5.
    
    Args:
        val_scores: predicted probabilities for each pair
        val_labels: true labels
        val_s1_ids: S1 entity IDs for each pair
        val_s23_ids: S2/S3 entity IDs for each pair
        val_ground_truth: {s1_id: set(true_match_ids)}
    
    Returns: (best_threshold, best_f05, results_dict)
    """
    print("\nSweeping threshold for F_0.5 optimization...", flush=True)
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.02)

    best_f05 = -1
    best_threshold = 0.5
    results = []

    for thresh in thresholds:
        # Build predictions at this threshold
        predictions = defaultdict(set)
        
        # For each candidate pair, if score >= threshold, add to predictions
        for s1_id, s23_id, score in zip(val_s1_ids, val_s23_ids, val_scores):
            if score >= thresh:
                predictions[s1_id].add(s23_id)

        # Ensure all val S1 entities are present
        for s1_id in val_ground_truth:
            if s1_id not in predictions:
                predictions[s1_id] = set()

        # Compute macro F_0.5
        f05, prec, rec, _ = compute_macro_f05(dict(predictions), val_ground_truth)
        results.append((thresh, f05, prec, rec))

        if f05 > best_f05:
            best_f05 = f05
            best_threshold = thresh

    # Print results
    print(f"\n{'Threshold':>10} {'F_0.5':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 40)
    for thresh, f05, prec, rec in results:
        marker = " <-- BEST" if thresh == best_threshold else ""
        print(f"{thresh:>10.3f} {f05:>8.4f} {prec:>10.4f} {rec:>8.4f}{marker}")

    print(f"\nBest threshold: {best_threshold:.3f}, F_0.5: {best_f05:.4f}")
    return best_threshold, best_f05, results


def train_model(
    base_dir: str = ".",
    output_dir: str = "output",
    neg_ratio: int = 3,
    max_pairs: int = 5000000,
    seed: int = 42,
):
    """
    Full training pipeline:
    1. Load normalized data + candidates from blocking stage
    2. Create training pairs
    3. Compute features
    4. Train LightGBM
    5. Calibrate probabilities
    6. Sweep threshold on validation split
    7. Save model + threshold
    """
    print(f"\n{'='*60}")
    print("TRAINING PIPELINE")
    print(f"{'='*60}\n")

    # Load normalized data
    norm_path = os.path.join(output_dir, "train_normalized.pkl")
    print(f"Loading normalized data from {norm_path}...", flush=True)
    with open(norm_path, "rb") as f:
        data = pickle.load(f)
    s1_df = data["s1"]
    s23_df = data["s23"]
    print(f"  S1: {len(s1_df)}, S23: {len(s23_df)}")

    # Load candidates
    cands_path = os.path.join(output_dir, "train_candidates.pkl")
    print(f"Loading candidates from {cands_path}...", flush=True)
    with open(cands_path, "rb") as f:
        candidates = pickle.load(f)
    print(f"  {len(candidates)} S1 entities with candidates")

    # Load ground truth
    gt_path = os.path.join(base_dir, "dataset/train/train_ground_truth.tsv")
    ground_truth = load_ground_truth(gt_path)
    print(f"  Ground truth: {len(ground_truth)} S1 entities")

    # Create training pairs
    pairs_df = create_training_pairs(
        s1_df, s23_df, candidates, ground_truth,
        neg_ratio=neg_ratio, max_pairs=max_pairs, seed=seed,
    )

    # Train/val split
    pairs_df, val_pairs_df, val_s1_set = train_val_split(pairs_df, ground_truth, val_fraction=0.15, seed=seed)

    # Compute features
    train_features = compute_features_for_pairs(pairs_df)
    val_features = compute_features_for_pairs(val_pairs_df)

    feature_cols = [c for c in train_features.columns if c not in ["s1_id", "s23_id", "label"]]
    print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")

    X_train = train_features[feature_cols].values.astype(np.float32)
    y_train = pairs_df["label"].values
    X_val = val_features[feature_cols].values.astype(np.float32)
    y_val = val_pairs_df["label"].values

    print(f"\nX_train shape: {X_train.shape}, y_train: {y_train.sum()} pos / {len(y_train)} total")
    print(f"X_val shape: {X_val.shape}, y_val: {y_val.sum()} pos / {len(y_val)} total")

    # Handle NaN/inf in features
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=1.0, neginf=0.0)

    # Train LightGBM
    print("\nTraining LightGBM...", flush=True)
    t0 = time.time()

    try:
        import lightgbm as lgb
        print("  Using LightGBM")

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

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
            params,
            lgb_train,
            num_boost_round=500,
            valid_sets=[lgb_val],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30),
                lgb.log_evaluation(period=50),
            ],
        )

        # Get predictions
        train_scores = model.predict(X_train)
        val_scores = model.predict(X_val)

        # Feature importance
        importance = model.feature_importance(importance_type="gain")
        feat_imp = sorted(zip(feature_cols, importance), key=lambda x: -x[1])
        print("\nFeature importance (top 15):")
        for fname, imp in feat_imp[:15]:
            print(f"  {fname}: {imp:.1f}")

    except ImportError:
        print("  LightGBM not available, falling back to XGBoost...")
        try:
            import xgboost as xgb
            print("  Using XGBoost")

            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)

            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": (len(y_train) - y_train.sum()) / max(y_train.sum(), 1),
                "seed": seed,
                "nthread": -1,
            }

            model = xgb.train(
                params, dtrain,
                num_boost_round=500,
                evals=[(dval, "val")],
                early_stopping_rounds=30,
                verbose_eval=50,
            )

            train_scores = model.predict(dtrain)
            val_scores = model.predict(dval)

        except ImportError:
            print("  Neither LightGBM nor XGBoost available!")
            print("  Falling back to sklearn GradientBoostingClassifier...")
            from sklearn.ensemble import GradientBoostingClassifier

            model = GradientBoostingClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, random_state=seed,
            )
            model.fit(X_train, y_train)
            train_scores = model.predict_proba(X_train)[:, 1]
            val_scores = model.predict_proba(X_val)[:, 1]

    print(f"  Training done in {time.time()-t0:.1f}s")

    # Build ground truth for validation entities
    val_ground_truth = {s1_id: ground_truth.get(s1_id, set()) for s1_id in val_s1_set}

    # Sweep threshold
    best_threshold, best_f05, sweep_results = sweep_threshold(
        val_scores,
        y_val,
        val_pairs_df["s1_id"].values,
        val_pairs_df["s23_id"].values,
        val_ground_truth,
    )

    # Save model and config
    model_path = os.path.join(output_dir, "model.pkl")
    config = {
        "model": model,
        "feature_cols": feature_cols,
        "threshold": best_threshold,
        "val_f05": best_f05,
        "sweep_results": sweep_results,
    }
    with open(model_path, "wb") as f:
        pickle.dump(config, f, protocol=4)
    print(f"\nSaved model + config to {model_path}")
    print(f"Best threshold: {best_threshold:.3f}")
    print(f"Validation F_0.5: {best_f05:.4f}")

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train matching model")
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--neg-ratio", type=int, default=3)
    parser.add_argument("--max-pairs", type=int, default=5000000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_model(args.base_dir, args.output_dir, args.neg_ratio, args.max_pairs, args.seed)
