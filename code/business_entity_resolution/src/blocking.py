"""
blocking.py — Candidate generation via TF-IDF + sorted-neighborhood blocking.

We UNION two complementary blocking strategies:
1) TF-IDF (word + char n-gram) vectorization of normalized name+address,
   with approximate nearest-neighbor retrieval (top-k per S1 entity).
2) Sorted-neighborhood blocking on a phonetic/normalized name sort key.

The blocking stage optimizes for RECALL — we want to capture as many true
matches as possible. Precision is left to the downstream matching model.

Designed for massive scale: ~2M S1 entities vs ~10M S2+S3 entities.
Uses batched processing to keep memory bounded.
"""

import os
import sys
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.sparse import vstack as sparse_vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from normalize import normalize_name, normalize_address, normalize_combined


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_source(path: str) -> pd.DataFrame:
    """Load a source TSV, verify tab-separation, fill NaN addresses."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[])
    assert "entity_id" in df.columns, f"Missing entity_id column in {path}"
    assert len(df.columns) >= 4, f"Tab separation may have failed: only {len(df.columns)} columns"
    # Replace literal 'nan' or empty with empty string
    df["business_address"] = df["business_address"].fillna("")
    df["business_name"] = df["business_name"].fillna("")
    df["country"] = df["country"].fillna("")
    # Replace pandas NaN
    df = df.fillna("")
    return df


def add_normalized_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized name, address, and combined fields to a dataframe."""
    print(f"  Normalizing {len(df)} records...", flush=True)
    t0 = time.time()
    df["name_norm"] = df["business_name"].apply(normalize_name)
    df["addr_norm"] = df["business_address"].apply(normalize_address)
    df["combined_norm"] = df.apply(
        lambda r: normalize_combined(r["business_name"], r["business_address"]),
        axis=1,
    )
    print(f"  Done in {time.time() - t0:.1f}s", flush=True)
    return df


# ---------------------------------------------------------------------------
# Strategy 1: TF-IDF + nearest-neighbor blocking
# ---------------------------------------------------------------------------

def tfidf_blocking(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    top_k: int = 20,
    batch_size: int = 50000,
) -> dict:
    """
    Build TF-IDF vectors for all entities, then for each S1 entity find
    the top-k most similar S2/S3 entities by cosine similarity.
    
    Returns: dict of {s1_entity_id: set(candidate_ids)}
    """
    print("=== TF-IDF Blocking ===", flush=True)
    t0 = time.time()

    # Build TF-IDF on all combined (name+address) text
    # Use both word and character n-grams for robustness against typos
    all_text_s23 = s23_df["combined_norm"].values
    all_text_s1 = s1_df["combined_norm"].values

    print(f"  Fitting TF-IDF on {len(all_text_s23)} S2/S3 records...", flush=True)
    tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=500000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    # Fit on S2/S3, then transform both
    tfidf_s23 = tfidf.fit_transform(all_text_s23)
    print(f"  TF-IDF matrix shape: {tfidf_s23.shape}, fitting took {time.time()-t0:.1f}s", flush=True)

    # Build NearestNeighbors index on S2/S3
    print(f"  Building NearestNeighbors index (k={top_k})...", flush=True)
    nn = NearestNeighbors(n_neighbors=min(top_k, tfidf_s23.shape[0]), metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(tfidf_s23)
    print(f"  Index built in {time.time()-t0:.1f}s", flush=True)

    # Query in batches
    candidates = {}
    s1_ids = s1_df["entity_id"].values
    s23_ids = s23_df["entity_id"].values
    n_batches = (len(s1_ids) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(s1_ids))
        batch_text = all_text_s1[start:end]
        batch_s1_ids = s1_ids[start:end]

        # Transform batch
        batch_vectors = tfidf.transform(batch_text)

        # Find neighbors
        distances, indices = nn.kneighbors(batch_vectors)

        for i, s1_id in enumerate(batch_s1_ids):
            # Filter by a generous distance threshold (cosine distance < 0.85)
            cand_set = set()
            for j, (dist, idx) in enumerate(zip(distances[i], indices[i])):
                if dist < 0.85:  # cosine distance; lower = more similar
                    cand_set.add(s23_ids[idx])
            candidates[s1_id] = cand_set

        if (batch_idx + 1) % 5 == 0 or batch_idx == n_batches - 1:
            print(f"  Batch {batch_idx+1}/{n_batches} done, elapsed {time.time()-t0:.1f}s", flush=True)

    avg_cands = np.mean([len(v) for v in candidates.values()])
    print(f"  TF-IDF blocking: {len(candidates)} S1 entities, avg {avg_cands:.1f} candidates each", flush=True)
    return candidates


# ---------------------------------------------------------------------------
# Strategy 2: Sorted-neighborhood blocking
# ---------------------------------------------------------------------------

def make_sort_key(name_norm: str, country: str) -> str:
    """Create a sort key from normalized name + country for sorted-neighborhood."""
    # Take first 8 chars of normalized name + country
    # This groups similar-named businesses near each other
    key = name_norm.replace(" ", "")[:8] + "_" + country.lower().strip()
    return key


def sorted_neighborhood_blocking(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    window_size: int = 5,
) -> dict:
    """
    Sorted-neighborhood blocking: sort all records by a sort key, then 
    compare each S1 record with S2/S3 records within a sliding window.
    
    Returns: dict of {s1_entity_id: set(candidate_ids)}
    """
    print("=== Sorted-Neighborhood Blocking ===", flush=True)
    t0 = time.time()

    # Create sort keys
    s1_keys = s1_df.apply(lambda r: make_sort_key(r["name_norm"], r["country"]), axis=1).values
    s23_keys = s23_df.apply(lambda r: make_sort_key(r["name_norm"], r["country"]), axis=1).values

    s1_ids = s1_df["entity_id"].values
    s23_ids = s23_df["entity_id"].values

    # Combine and sort
    all_keys = np.concatenate([s1_keys, s23_keys])
    all_ids = np.concatenate([s1_ids, s23_ids])
    is_s1 = np.concatenate([np.ones(len(s1_ids), dtype=bool), np.zeros(len(s23_ids), dtype=bool)])

    sort_order = np.argsort(all_keys)
    all_keys_sorted = all_keys[sort_order]
    all_ids_sorted = all_ids[sort_order]
    is_s1_sorted = is_s1[sort_order]

    # Sliding window
    candidates = defaultdict(set)
    n = len(all_keys_sorted)

    for i in range(n):
        if not is_s1_sorted[i]:
            continue
        s1_id = all_ids_sorted[i]
        # Look at neighbors within the window
        lo = max(0, i - window_size)
        hi = min(n, i + window_size + 1)
        for j in range(lo, hi):
            if j == i:
                continue
            if not is_s1_sorted[j]:  # It's an S2/S3 record
                candidates[s1_id].add(all_ids_sorted[j])

    # Ensure all S1 entities are present
    for s1_id in s1_ids:
        if s1_id not in candidates:
            candidates[s1_id] = set()

    avg_cands = np.mean([len(v) for v in candidates.values()])
    print(f"  Sorted-neighborhood: {len(candidates)} S1 entities, avg {avg_cands:.1f} candidates each", flush=True)
    print(f"  Done in {time.time()-t0:.1f}s", flush=True)
    return dict(candidates)


# ---------------------------------------------------------------------------
# Strategy 3: Token-based blocking (for higher recall)
# ---------------------------------------------------------------------------

def token_blocking(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    max_block_size: int = 1000,
    min_token_len: int = 4,
) -> dict:
    """
    Token blocking: create inverted index on name tokens, then for each S1
    entity find S2/S3 entities sharing at least one discriminating token.
    
    Skip very common tokens (block size > max_block_size) to avoid quadratic blowup.
    """
    print("=== Token Blocking ===", flush=True)
    t0 = time.time()

    # Build inverted index from S2/S3
    inverted_index = defaultdict(set)
    s23_ids = s23_df["entity_id"].values
    s23_names = s23_df["name_norm"].values
    s23_countries = s23_df["country"].values

    for i, (eid, name, country) in enumerate(zip(s23_ids, s23_names, s23_countries)):
        tokens = name.split()
        for tok in tokens:
            if len(tok) >= min_token_len:
                key = tok + "_" + country.lower().strip()
                inverted_index[key].add(eid)

    # Remove overly large blocks
    keys_to_remove = [k for k, v in inverted_index.items() if len(v) > max_block_size]
    for k in keys_to_remove:
        del inverted_index[k]
    print(f"  Inverted index: {len(inverted_index)} blocks (removed {len(keys_to_remove)} large blocks)", flush=True)

    # Query S1
    candidates = {}
    s1_ids = s1_df["entity_id"].values
    s1_names = s1_df["name_norm"].values
    s1_countries = s1_df["country"].values

    for s1_id, name, country in zip(s1_ids, s1_names, s1_countries):
        cand_set = set()
        tokens = name.split()
        for tok in tokens:
            if len(tok) >= min_token_len:
                key = tok + "_" + country.lower().strip()
                if key in inverted_index:
                    cand_set.update(inverted_index[key])
        candidates[s1_id] = cand_set

    avg_cands = np.mean([len(v) for v in candidates.values()])
    print(f"  Token blocking: {len(candidates)} S1 entities, avg {avg_cands:.1f} candidates each", flush=True)
    print(f"  Done in {time.time()-t0:.1f}s", flush=True)
    return candidates


# ---------------------------------------------------------------------------
# Union blocking strategies
# ---------------------------------------------------------------------------

def union_candidates(*candidate_dicts) -> dict:
    """Merge multiple candidate dicts by unioning their candidate sets per S1 entity."""
    merged = defaultdict(set)
    for cands in candidate_dicts:
        for s1_id, cand_set in cands.items():
            merged[s1_id].update(cand_set)
    return dict(merged)


# ---------------------------------------------------------------------------
# Measure blocking recall ceiling
# ---------------------------------------------------------------------------

def measure_recall_ceiling(candidates: dict, ground_truth: dict) -> float:
    """
    Measure what fraction of true matches appear in the candidate set.
    This is the recall ceiling — it caps everything downstream.
    """
    total_true = 0
    found_true = 0

    for s1_id, true_ids in ground_truth.items():
        if not true_ids:
            continue
        total_true += len(true_ids)
        cand_set = candidates.get(s1_id, set())
        found_true += len(true_ids & cand_set)

    recall_ceiling = found_true / total_true if total_true > 0 else 1.0
    print(f"\n=== BLOCKING RECALL CEILING ===")
    print(f"  True matches: {total_true}")
    print(f"  Found in candidates: {found_true}")
    print(f"  Recall ceiling: {recall_ceiling:.6f} ({recall_ceiling*100:.4f}%)")
    print(f"  Missed: {total_true - found_true}")
    return recall_ceiling


# ---------------------------------------------------------------------------
# Save / load candidates
# ---------------------------------------------------------------------------

def save_candidates(candidates: dict, path: str):
    """Save candidates dict to a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(candidates, f, protocol=4)
    print(f"  Saved candidates to {path}")


def load_candidates(path: str) -> dict:
    """Load candidates dict from a pickle file."""
    with open(path, "rb") as f:
        candidates = pickle.load(f)
    print(f"  Loaded candidates from {path} ({len(candidates)} entities)")
    return candidates


def save_candidates_tsv(candidates: dict, path: str, all_s1_ids: list = None):
    """
    Save candidates as a TSV file in the submission format.
    Ensures every S1 entity gets a row even if no candidates.
    """
    rows = []
    all_ids = set(candidates.keys())
    if all_s1_ids:
        all_ids = all_ids | set(all_s1_ids)

    for s1_id in sorted(all_ids):
        cand_set = candidates.get(s1_id, set())
        cand_str = ",".join(sorted(cand_set)) if cand_set else ""
        rows.append(f"{s1_id}\t{cand_str}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        f.write("\n".join(rows) + "\n")
    print(f"  Saved {len(rows)} rows to {path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_blocking(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    gt_path: str = None,
    output_dir: str = "output",
    top_k: int = 20,
    mode: str = "train",
):
    """
    Run the full blocking pipeline.
    
    Args:
        s1_path: Path to source1 TSV
        s2_path: Path to source2 TSV
        s3_path: Path to source3 TSV
        gt_path: Path to ground truth TSV (for measuring recall ceiling; None for test)
        output_dir: Where to save output files
        top_k: Number of nearest neighbors for TF-IDF blocking
        mode: 'train' or 'test'
    """
    print(f"\n{'='*60}")
    print(f"BLOCKING PIPELINE — {mode.upper()}")
    print(f"{'='*60}\n")

    # Load data
    print("Loading source files...", flush=True)
    s1 = load_source(s1_path)
    s2 = load_source(s2_path)
    s3 = load_source(s3_path)
    print(f"  S1: {len(s1)}, S2: {len(s2)}, S3: {len(s3)}")

    # Combine S2 and S3
    s23 = pd.concat([s2, s3], ignore_index=True)
    print(f"  S2+S3 combined: {len(s23)}")

    # Normalize
    print("\nNormalizing...", flush=True)
    s1 = add_normalized_fields(s1)
    s23 = add_normalized_fields(s23)

    # Save normalized data for later use
    os.makedirs(output_dir, exist_ok=True)
    norm_path = os.path.join(output_dir, f"{mode}_normalized.pkl")
    with open(norm_path, "wb") as f:
        pickle.dump({"s1": s1, "s23": s23}, f, protocol=4)
    print(f"  Saved normalized data to {norm_path}")

    # Run blocking strategies
    print("\nRunning blocking strategies...", flush=True)

    # Strategy 1: TF-IDF
    cands_tfidf = tfidf_blocking(s1, s23, top_k=top_k)

    # Strategy 2: Token blocking (more recall than sorted-neighborhood for this scale)
    cands_token = token_blocking(s1, s23, max_block_size=1000, min_token_len=4)

    # Strategy 3: Sorted-neighborhood (additional recall)
    cands_sorted = sorted_neighborhood_blocking(s1, s23, window_size=3)

    # Union all
    print("\nMerging candidate sets...", flush=True)
    candidates = union_candidates(cands_tfidf, cands_token, cands_sorted)

    avg_cands = np.mean([len(v) for v in candidates.values()])
    print(f"  Combined: {len(candidates)} S1 entities, avg {avg_cands:.1f} candidates each")

    # Measure recall ceiling if ground truth available
    if gt_path:
        gt = pd.read_csv(gt_path, sep="\t")
        ground_truth = {}
        for _, row in gt.iterrows():
            s1_id = row["source1_entity_id"]
            matched = row["matched_entity_ids"]
            if pd.isna(matched) or str(matched).strip() == "":
                ground_truth[s1_id] = set()
            else:
                ground_truth[s1_id] = set(str(matched).split(","))
        
        recall_ceiling = measure_recall_ceiling(candidates, ground_truth)

    # Save candidates
    cands_path = os.path.join(output_dir, f"{mode}_candidates.pkl")
    save_candidates(candidates, cands_path)

    return candidates, s1, s23


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run blocking pipeline")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    if args.mode == "train":
        s1_path = os.path.join(args.base_dir, "dataset/train/train_source1.tsv")
        s2_path = os.path.join(args.base_dir, "dataset/train/train_source2.tsv")
        s3_path = os.path.join(args.base_dir, "dataset/train/train_source3.tsv")
        gt_path = os.path.join(args.base_dir, "dataset/train/train_ground_truth.tsv")
    else:
        s1_path = os.path.join(args.base_dir, "dataset/test/test_source1.tsv")
        s2_path = os.path.join(args.base_dir, "dataset/test/test_source2.tsv")
        s3_path = os.path.join(args.base_dir, "dataset/test/test_source3.tsv")
        gt_path = None

    run_blocking(s1_path, s2_path, s3_path, gt_path, args.output_dir, args.top_k, args.mode)
