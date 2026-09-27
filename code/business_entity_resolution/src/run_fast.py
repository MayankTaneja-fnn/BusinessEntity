"""
run_fast.py — Memory-efficient pipeline using chunked reading.

Avoids loading all 10M+ S2/S3 rows into memory. Instead:
1. Reads ground truth first, samples S1 entities
2. Collects true S23 IDs needed from ground truth  
3. Reads S23 files in chunks, keeping only needed records
4. Runs blocking, feature engineering, training, and inference

For test inference: uses the trained model on the full test set in batches.
"""

import os
import sys
import time
import gc
import pickle
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from normalize import normalize_name, normalize_address, normalize_combined, normalize_df_fast
from features import compute_features_batch
from evaluate import compute_f05_per_entity, compute_macro_f05


def normalize_df(df, label=""):
    t0 = time.time()
    if label:
        print(f"  Normalizing {len(df):,} {label}...", end=" ", flush=True)
    df = normalize_df_fast(df)
    if label:
        print(f"done in {time.time()-t0:.1f}s", flush=True)
    return df


def load_source_chunked(path, entity_filter=None, chunksize=500000):
    """Load a source TSV, optionally filtering to specific entity_ids."""
    chunks = []
    for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                             na_values=[], chunksize=chunksize):
        chunk["business_address"] = chunk["business_address"].fillna("")
        chunk["business_name"] = chunk["business_name"].fillna("")
        chunk["country"] = chunk["country"].fillna("")
        chunk = chunk.fillna("")
        if entity_filter is not None:
            chunk = chunk[chunk["entity_id"].isin(entity_filter)]
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()




def tfidf_block(s1_df, s23_df, top_k=15):
    """TF-IDF blocking with char n-grams."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    print("  TF-IDF blocking...", flush=True)
    t0 = time.time()
    
    tfidf = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5),
        max_features=300000, sublinear_tf=True, dtype=np.float32)
    
    tfidf_s23 = tfidf.fit_transform(s23_df["combined_norm"].values)
    nn = NearestNeighbors(n_neighbors=min(top_k, len(s23_df)),
                         metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(tfidf_s23)
    
    tfidf_s1 = tfidf.transform(s1_df["combined_norm"].values)
    distances, indices = nn.kneighbors(tfidf_s1)
    
    s1_ids = s1_df["entity_id"].values
    s23_ids = s23_df["entity_id"].values
    
    candidates = {}
    for i, s1_id in enumerate(s1_ids):
        cand_set = set()
        for dist, idx in zip(distances[i], indices[i]):
            if dist < 0.85:
                cand_set.add(s23_ids[idx])
        candidates[s1_id] = cand_set
    
    avg = np.mean([len(v) for v in candidates.values()])
    print(f"    TF-IDF: {len(candidates)} S1, avg {avg:.1f} candidates, {time.time()-t0:.1f}s", flush=True)
    return candidates


def token_block(s1_df, s23_df, max_block=500, min_tok_len=4):
    """Token-based inverted index blocking."""
    print("  Token blocking...", flush=True)
    t0 = time.time()
    
    inv_idx = defaultdict(set)
    for eid, name, country in zip(s23_df["entity_id"].values, 
                                   s23_df["name_norm"].values,
                                   s23_df["country"].values):
        for tok in name.split():
            if len(tok) >= min_tok_len:
                inv_idx[tok + "_" + country.strip().lower()].add(eid)
    
    # Remove large blocks
    inv_idx = {k: v for k, v in inv_idx.items() if len(v) <= max_block}
    
    candidates = {}
    for s1_id, name, country in zip(s1_df["entity_id"].values,
                                     s1_df["name_norm"].values,
                                     s1_df["country"].values):
        cands = set()
        for tok in name.split():
            if len(tok) >= min_tok_len:
                key = tok + "_" + country.strip().lower()
                if key in inv_idx:
                    cands.update(inv_idx[key])
        candidates[s1_id] = cands
    
    avg = np.mean([len(v) for v in candidates.values()])
    print(f"    Token: {len(candidates)} S1, avg {avg:.1f} candidates, {time.time()-t0:.1f}s", flush=True)
    return candidates


def union_cands(*dicts):
    merged = defaultdict(set)
    for d in dicts:
        for k, v in d.items():
            merged[k].update(v)
    return dict(merged)


def measure_recall(candidates, ground_truth):
    total = found = 0
    for s1_id, true_ids in ground_truth.items():
        if not true_ids: continue
        total += len(true_ids)
        found += len(true_ids & candidates.get(s1_id, set()))
    rc = found / total if total > 0 else 1.0
    print(f"\n  BLOCKING RECALL CEILING: {rc:.6f} ({rc*100:.4f}%)")
    print(f"  True: {total}, Found: {found}, Missed: {total - found}")
    return rc


def run_pipeline(base_dir=".", output_dir="output", n_s1=20000, top_k=15, seed=42):
    """Run the complete pipeline efficiently."""
    t_start = time.time()
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("STAGE 0: LOAD AND SAMPLE TRAINING DATA")
    print(f"{'='*60}\n")
    
    # Load ground truth first (small)
    gt_path = os.path.join(base_dir, "dataset/train/train_ground_truth.tsv")
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    gt_dict = {}
    for _, row in gt_df.iterrows():
        s1_id = row["source1_entity_id"]
        matched = row["matched_entity_ids"]
        gt_dict[s1_id] = set(matched.split(",")) if matched.strip() else set()
    print(f"Ground truth: {len(gt_dict)} S1 entities")
    
    # Print distribution
    match_counts = [len(v) for v in gt_dict.values()]
    print(f"Match count stats: min={min(match_counts)}, max={max(match_counts)}, "
          f"mean={np.mean(match_counts):.2f}")
    print(f"Singletons: {sum(1 for c in match_counts if c==0)} "
          f"({sum(1 for c in match_counts if c==0)/len(match_counts)*100:.1f}%)")
    del gt_df; gc.collect()
    
    # Sample S1 IDs
    rng = np.random.RandomState(seed)
    all_s1_ids = list(gt_dict.keys())
    rng.shuffle(all_s1_ids)
    sampled_s1 = set(all_s1_ids[:n_s1])
    
    # Get all true match IDs for sampled S1
    needed_s23 = set()
    for s1_id in sampled_s1:
        needed_s23.update(gt_dict[s1_id])
    print(f"\nSampled {n_s1} S1 entities, need {len(needed_s23)} true S23 matches")
    
    # Also get some random S23 IDs for negative examples
    # Read S2/S3 entity IDs only (first column)
    print("Scanning S2/S3 entity IDs...", flush=True)
    s23_all_ids = set()
    for fname in ["train_source2.tsv", "train_source3.tsv"]:
        fpath = os.path.join(base_dir, "dataset/train", fname)
        with open(fpath, encoding="utf-8") as f:
            next(f)  # skip header
            for line in f:
                eid = line.split("\t", 1)[0]
                s23_all_ids.add(eid)
    print(f"  Total S23 IDs: {len(s23_all_ids)}")
    
    # Sample additional S23 IDs for noise
    other_ids = list(s23_all_ids - needed_s23)
    rng.shuffle(other_ids)
    n_noise = min(len(other_ids), max(5 * len(needed_s23), 100000))
    needed_s23_full = needed_s23 | set(other_ids[:n_noise])
    print(f"  Including {n_noise} noise S23 records")
    print(f"  Total S23 to load: {len(needed_s23_full)}")
    
    # Load S1 (sampled)
    print("\nLoading sampled S1...", flush=True)
    s1_df = load_source_chunked(
        os.path.join(base_dir, "dataset/train/train_source1.tsv"),
        entity_filter=sampled_s1)
    print(f"  S1 loaded: {len(s1_df)}")
    
    # Load S23 (filtered)
    print("Loading filtered S23...", flush=True)
    s2_df = load_source_chunked(
        os.path.join(base_dir, "dataset/train/train_source2.tsv"),
        entity_filter=needed_s23_full)
    s3_df = load_source_chunked(
        os.path.join(base_dir, "dataset/train/train_source3.tsv"),
        entity_filter=needed_s23_full)
    s23_df = pd.concat([s2_df, s3_df], ignore_index=True)
    del s2_df, s3_df; gc.collect()
    print(f"  S23 loaded: {len(s23_df)}")
    
    print(f"\nData loading took {time.time()-t_start:.1f}s")
    
    # ============================================================
    print(f"\n{'='*60}")
    print("STAGE 1: NORMALIZATION")
    print(f"{'='*60}\n")
    
    s1_df = normalize_df(s1_df)
    s23_df = normalize_df(s23_df)
    
    # ============================================================
    print(f"\n{'='*60}")
    print("STAGE 2: BLOCKING")
    print(f"{'='*60}\n")
    
    cands_tfidf = tfidf_block(s1_df, s23_df, top_k=top_k)
    cands_token = token_block(s1_df, s23_df, max_block=500, min_tok_len=4)
    candidates = union_cands(cands_tfidf, cands_token)
    del cands_tfidf, cands_token; gc.collect()
    
    avg = np.mean([len(v) for v in candidates.values()])
    print(f"\n  Combined: {len(candidates)} S1, avg {avg:.1f} candidates")
    
    gt_sample = {s1_id: gt_dict[s1_id] for s1_id in sampled_s1}
    recall_ceiling = measure_recall(candidates, gt_sample)
    
    # ============================================================
    print(f"\n{'='*60}")
    print("STAGE 3+4: FEATURES & TRAINING")
    print(f"{'='*60}\n")
    
    # Split S1 into train/val
    s1_ids_list = list(sampled_s1)
    rng.shuffle(s1_ids_list)
    n_val = int(len(s1_ids_list) * 0.15)
    val_s1 = set(s1_ids_list[:n_val])
    train_s1 = set(s1_ids_list[n_val:])
    print(f"Train S1: {len(train_s1)}, Val S1: {len(val_s1)}")
    
    # Create pairs
    s1_lookup = s1_df.set_index("entity_id").to_dict("index")
    s23_lookup = s23_df.set_index("entity_id").to_dict("index")
    
    def make_pairs(s1_set, neg_ratio=3):
        pairs = []
        for s1_id in s1_set:
            if s1_id not in s1_lookup: continue
            true_ids = gt_sample.get(s1_id, set())
            cand_ids = candidates.get(s1_id, set())
            if not cand_ids: continue
            s1_info = s1_lookup[s1_id]
            
            pos_ids = true_ids & cand_ids
            for mid in pos_ids:
                if mid not in s23_lookup: continue
                info = s23_lookup[mid]
                pairs.append({
                    "s1_id": s1_id, "s23_id": mid, "label": 1,
                    "name1_norm": s1_info.get("name_norm",""),
                    "name2_norm": info.get("name_norm",""),
                    "addr1_norm": s1_info.get("addr_norm",""),
                    "addr2_norm": info.get("addr_norm",""),
                    "country1": s1_info.get("country",""),
                    "country2": info.get("country",""),
                })
            
            neg_ids = list(cand_ids - true_ids)
            if neg_ids:
                n = min(len(neg_ids), max(neg_ratio * len(pos_ids), 1))
                for nid in rng.choice(neg_ids, size=n, replace=False):
                    if nid not in s23_lookup: continue
                    info = s23_lookup[nid]
                    pairs.append({
                        "s1_id": s1_id, "s23_id": nid, "label": 0,
                        "name1_norm": s1_info.get("name_norm",""),
                        "name2_norm": info.get("name_norm",""),
                        "addr1_norm": s1_info.get("addr_norm",""),
                        "addr2_norm": info.get("addr_norm",""),
                        "country1": s1_info.get("country",""),
                        "country2": info.get("country",""),
                    })
        return pd.DataFrame(pairs)
    
    train_pairs = make_pairs(train_s1, neg_ratio=3)
    val_pairs = make_pairs(val_s1, neg_ratio=3)
    print(f"Train pairs: {len(train_pairs)} ({train_pairs['label'].sum()} pos)")
    print(f"Val pairs: {len(val_pairs)} ({val_pairs['label'].sum()} pos)")
    
    # Compute features
    print("\nComputing features...", flush=True)
    t_feat = time.time()
    train_feats = compute_features_batch(train_pairs)
    val_feats = compute_features_batch(val_pairs)
    print(f"  Done in {time.time()-t_feat:.1f}s")
    
    feature_cols = list(train_feats.columns)
    X_train = np.nan_to_num(train_feats.values.astype(np.float32))
    y_train = train_pairs["label"].values
    X_val = np.nan_to_num(val_feats.values.astype(np.float32))
    y_val = val_pairs["label"].values
    
    print(f"X_train: {X_train.shape}, pos rate: {y_train.mean():.3f}")
    print(f"X_val: {X_val.shape}, pos rate: {y_val.mean():.3f}")
    
    # Train LightGBM
    print("\nTraining LightGBM...", flush=True)
    t_train = time.time()
    
    import lightgbm as lgb
    
    lgb_train = lgb.Dataset(X_train, y_train, feature_name=feature_cols)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train, feature_name=feature_cols)
    
    params = {
        "objective": "binary", "metric": "binary_logloss",
        "boosting_type": "gbdt", "num_leaves": 63,
        "learning_rate": 0.05, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 5,
        "min_child_samples": 50,
        "scale_pos_weight": (len(y_train) - y_train.sum()) / max(y_train.sum(), 1),
        "verbose": -1, "n_jobs": -1, "seed": seed,
    }
    
    model = lgb.train(
        params, lgb_train, num_boost_round=500,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)])
    
    val_scores = model.predict(X_val)
    print(f"  Training: {time.time()-t_train:.1f}s")
    
    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: -x[1])
    print("\nFeature importance:")
    for fn, imp in feat_imp[:12]:
        print(f"  {fn}: {imp:.1f}")
    
    # ============================================================
    print(f"\n{'='*60}")
    print("STAGE 5: THRESHOLD OPTIMIZATION")
    print(f"{'='*60}\n")
    
    val_gt = {s1_id: gt_sample.get(s1_id, set()) for s1_id in val_s1}
    val_s1_ids = val_pairs["s1_id"].values
    val_s23_ids = val_pairs["s23_id"].values
    
    best_f05 = -1
    best_thresh = 0.5
    
    for thresh in np.arange(0.05, 0.95, 0.02):
        preds = defaultdict(set)
        s23_best = {}
        for idx in range(len(val_scores)):
            if val_scores[idx] < thresh: continue
            s1_id = val_s1_ids[idx]
            s23_id = val_s23_ids[idx]
            if s23_id not in s23_best or val_scores[idx] > s23_best[s23_id][1]:
                s23_best[s23_id] = (s1_id, val_scores[idx])
        
        for s23_id, (s1_id, _) in s23_best.items():
            preds[s1_id].add(s23_id)
        for s1_id in val_s1:
            if s1_id not in preds:
                preds[s1_id] = set()
        
        f05, prec, rec, _ = compute_macro_f05(dict(preds), val_gt)
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
            print(f"  thresh={thresh:.3f} F0.5={f05:.4f} P={prec:.4f} R={rec:.4f} <-- BEST")
    
    print(f"\n  BEST: threshold={best_thresh:.3f}, F_0.5={best_f05:.4f}")
    
    # Save model
    config = {
        "model": model, "feature_cols": feature_cols,
        "threshold": best_thresh, "val_f05": best_f05,
    }
    with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
        pickle.dump(config, f, protocol=4)
    print(f"  Model saved to {output_dir}/model.pkl")
    
    # ============================================================
    print(f"\n{'='*60}")
    print("STAGE 6: TEST INFERENCE")
    print(f"{'='*60}\n")
    
    # Load test S1
    print("Loading test S1...", flush=True)
    test_s1 = load_source_chunked(
        os.path.join(base_dir, "dataset/test/test_source1.tsv"))
    print(f"  Test S1: {len(test_s1)}")
    test_s1 = normalize_df(test_s1)
    all_test_s1_ids = test_s1["entity_id"].tolist()
    
    # Process test in country batches for memory efficiency
    test_predictions = {}
    test_candidates = {}
    
    # Get all test S1 IDs per country for batched processing  
    countries = test_s1["country"].unique()
    print(f"  Countries in test: {list(countries)}")
    
    for country in countries:
        print(f"\n  Processing country: {country}")
        s1_country = test_s1[test_s1["country"] == country].copy()
        print(f"    S1 entities: {len(s1_country)}")
        
        # Load S2+S3 for this country
        s23_country_dfs = []
        for fname in ["test_source2.tsv", "test_source3.tsv"]:
            fpath = os.path.join(base_dir, "dataset/test", fname)
            for chunk in pd.read_csv(fpath, sep="\t", dtype=str,
                                     keep_default_na=False, na_values=[],
                                     chunksize=500000):
                chunk = chunk.fillna("")
                chunk = chunk[chunk["country"] == country]
                if len(chunk) > 0:
                    s23_country_dfs.append(chunk)
        
        if not s23_country_dfs:
            print(f"    No S23 records for {country}")
            for s1_id in s1_country["entity_id"]:
                test_predictions[s1_id] = set()
                test_candidates[s1_id] = set()
            continue
        
        s23_country = pd.concat(s23_country_dfs, ignore_index=True)
        del s23_country_dfs; gc.collect()
        print(f"    S23 entities: {len(s23_country)}")
        
        s23_country = normalize_df(s23_country)
        
        # Blocking
        cands_t = tfidf_block(s1_country, s23_country, top_k=top_k)
        cands_tok = token_block(s1_country, s23_country, max_block=500, min_tok_len=4)
        country_cands = union_cands(cands_t, cands_tok)
        
        # Store candidates
        for s1_id in s1_country["entity_id"]:
            test_candidates[s1_id] = country_cands.get(s1_id, set())
        
        # Create inference pairs
        s1_lkp = s1_country.set_index("entity_id").to_dict("index")
        s23_lkp = s23_country.set_index("entity_id").to_dict("index")
        
        inf_pairs = []
        for s1_id, cand_ids in country_cands.items():
            if s1_id not in s1_lkp: continue
            s1_info = s1_lkp[s1_id]
            for cid in cand_ids:
                if cid not in s23_lkp: continue
                info = s23_lkp[cid]
                inf_pairs.append({
                    "s1_id": s1_id, "s23_id": cid,
                    "name1_norm": s1_info.get("name_norm",""),
                    "name2_norm": info.get("name_norm",""),
                    "addr1_norm": s1_info.get("addr_norm",""),
                    "addr2_norm": info.get("addr_norm",""),
                    "country1": s1_info.get("country",""),
                    "country2": info.get("country",""),
                })
        
        if not inf_pairs:
            for s1_id in s1_country["entity_id"]:
                test_predictions[s1_id] = set()
            del s23_country; gc.collect()
            continue
        
        inf_df = pd.DataFrame(inf_pairs)
        print(f"    Inference pairs: {len(inf_df)}")
        
        # Compute features and score in batches
        batch_size = 200000
        all_scores = []
        for b_start in range(0, len(inf_df), batch_size):
            b_end = min(b_start + batch_size, len(inf_df))
            batch = inf_df.iloc[b_start:b_end]
            feats = compute_features_batch(batch)
            for col in feature_cols:
                if col not in feats.columns:
                    feats[col] = 0.0
            X = np.nan_to_num(feats[feature_cols].values.astype(np.float32))
            scores = model.predict(X)
            all_scores.append(scores)
            print(f"      Scored batch {b_start}-{b_end}", flush=True)
        
        scores = np.concatenate(all_scores)
        
        # Conflict resolution
        s23_best = {}
        for idx in range(len(scores)):
            if scores[idx] < best_thresh: continue
            s1_id = inf_df.iloc[idx]["s1_id"]
            s23_id = inf_df.iloc[idx]["s23_id"]
            if s23_id not in s23_best or scores[idx] > s23_best[s23_id][1]:
                s23_best[s23_id] = (s1_id, scores[idx])
        
        country_preds = defaultdict(set)
        for s23_id, (s1_id, _) in s23_best.items():
            country_preds[s1_id].add(s23_id)
        
        for s1_id in s1_country["entity_id"]:
            test_predictions[s1_id] = country_preds.get(s1_id, set())
        
        n_matched = sum(1 for v in country_preds.values() if v)
        total_links = sum(len(v) for v in country_preds.values())
        print(f"    Matched: {n_matched} entities, {total_links} total links")
        
        del s23_country, inf_df; gc.collect()
    
    # Ensure ALL test S1 entities have entries
    for s1_id in all_test_s1_ids:
        if s1_id not in test_predictions:
            test_predictions[s1_id] = set()
        if s1_id not in test_candidates:
            test_candidates[s1_id] = set()
    
    # Write outputs
    print("\nWriting output files...", flush=True)
    
    # matching_results.tsv
    match_path = os.path.join(output_dir, "matching_results.tsv")
    with open(match_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in sorted(test_predictions.keys()):
            matched = test_predictions[s1_id]
            f.write(f"{s1_id}\t{','.join(sorted(matched)) if matched else ''}\n")
    print(f"  {match_path}: {len(test_predictions)} rows")
    
    # candidate_pairs.tsv
    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")
    with open(cand_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in sorted(test_candidates.keys()):
            cands = test_candidates[s1_id]
            f.write(f"{s1_id}\t{','.join(sorted(cands)) if cands else ''}\n")
    print(f"  {cand_path}: {len(test_candidates)} rows")
    
    # Validate
    print(f"\n{'='*60}")
    print("VALIDATION")
    print(f"{'='*60}\n")
    
    import subprocess
    validate_script = os.path.join(base_dir, "utils/validate_submission.py")
    test_dir = os.path.join(base_dir, "dataset/test")
    result = subprocess.run(
        [sys.executable, validate_script,
         "--matching", match_path,
         "--candidate", cand_path,
         "--test-dir", test_dir],
        capture_output=True, text=True, encoding="utf-8")
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print("METHODOLOGY SUMMARY")
    print(f"{'='*60}")
    print(f"Blocking: TF-IDF (char 3-5gram, top-{top_k}) + Token blocking (union)")
    print(f"Blocking recall ceiling: {recall_ceiling:.4f}")
    print(f"Features: {', '.join(feature_cols)}")
    print(f"Model: LightGBM GBDT (MIT license, <1M params)")
    print(f"Threshold: {best_thresh:.3f} (optimized for F_0.5)")
    print(f"Validation F_0.5: {best_f05:.4f}")
    print(f"Train sample: {n_s1} S1 entities")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--n-s1", type=int, default=20000)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    run_pipeline(args.base_dir, args.output_dir, args.n_s1, args.top_k, args.seed)
