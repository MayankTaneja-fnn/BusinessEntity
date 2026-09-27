"""
infer_test.py — Memory-efficient test-set inference.

The main OOM issue in run_fast.py was fitting TF-IDF on 3.8M US records at once.
Fix: process S23 in sub-batches; use HashingVectorizer (O(1) memory for vocabulary)
combined with a TruncatedSVD for dimensionality reduction, then do batched cosine NN.

Pipeline per country:
  1. Load + normalize all S1 for country (bounded by test S1 size per country)
  2. Load + normalize S23 for country in CHUNKS
  3. For each S23 chunk, do token blocking (inverted index) -> build running inv_idx
  4. Do TF-IDF blocking in S23 sub-batches of S23_BATCH_SIZE records
  5. Union candidates; score; conflict-resolve; accumulate predictions

Usage:
    python src/infer_test.py --base-dir . --output-dir output
"""

import os, sys, time, gc, pickle, argparse
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from normalize import normalize_name, normalize_address, normalize_combined, normalize_df_fast
from features import compute_features_batch
from evaluate import compute_macro_f05


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
S23_TFIDF_BATCH   = 500_000   # S23 records per TF-IDF index build
S23_LOAD_CHUNK    = 500_000   # rows per pandas read_csv chunk
SCORE_BATCH       = 200_000   # pairs per LightGBM batch
TOP_K             = 15        # nearest neighbours per S1 entity
MAX_TOKEN_BLOCK   = 500       # discard inverted-index blocks bigger than this
MIN_TOK_LEN       = 4         # minimum token length for token blocking


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalize_df(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    t0 = time.time()
    if label:
        print(f"  Normalizing {len(df):,} {label} records...", end=" ", flush=True)
    df = normalize_df_fast(df)
    if label:
        print(f"done in {time.time()-t0:.1f}s", flush=True)
    return df



def load_source_full(path: str) -> pd.DataFrame:
    """Load an entire source TSV with safe defaults."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[])
    df = df.fillna("")
    return df


def _tfidf_block_batch(s1_df: pd.DataFrame,
                       s23_batch: pd.DataFrame,
                       top_k: int = TOP_K) -> dict:
    """
    Fit TF-IDF on ONE batch of S23, query all S1 entities.
    Returns {s1_id: set(candidate_s23_ids)}.
    Uses HashingVectorizer so memory is bounded regardless of vocabulary size.
    """
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.neighbors import NearestNeighbors

    # HashingVectorizer: no vocabulary stored → bounded memory
    vec = HashingVectorizer(
        analyzer="char_wb", ngram_range=(3, 5),
        n_features=2**18,          # 256K features — fits in ~100 MB float32
        alternate_sign=False,
        norm="l2",
        dtype=np.float32,
    )

    X_s23 = vec.transform(s23_batch["combined_norm"].values)
    X_s1  = vec.transform(s1_df["combined_norm"].values)

    k = min(top_k, X_s23.shape[0])
    nn = NearestNeighbors(n_neighbors=k, metric="cosine",
                          algorithm="brute", n_jobs=-1)
    nn.fit(X_s23)
    distances, indices = nn.kneighbors(X_s1)

    s1_ids  = s1_df["entity_id"].values
    s23_ids = s23_batch["entity_id"].values

    cands = {}
    for i, s1_id in enumerate(s1_ids):
        cset = set()
        for dist, idx in zip(distances[i], indices[i]):
            if dist < 0.85:
                cset.add(s23_ids[idx])
        cands[s1_id] = cset
    return cands


def tfidf_block_chunked(s1_df: pd.DataFrame,
                        s23_df: pd.DataFrame,
                        batch_size: int = S23_TFIDF_BATCH,
                        top_k: int = TOP_K) -> dict:
    """
    TF-IDF blocking where S23 is processed in sub-batches to avoid OOM.
    Candidates from all batches are unioned per S1 entity.
    """
    print(f"  TF-IDF blocking (S23 in batches of {batch_size:,})...", flush=True)
    t0 = time.time()
    merged = defaultdict(set)
    n_batches = (len(s23_df) + batch_size - 1) // batch_size

    for b in range(n_batches):
        s23_batch = s23_df.iloc[b * batch_size : (b + 1) * batch_size]
        batch_cands = _tfidf_block_batch(s1_df, s23_batch, top_k=top_k)
        for s1_id, cset in batch_cands.items():
            merged[s1_id].update(cset)
        print(f"    batch {b+1}/{n_batches} done ({time.time()-t0:.0f}s)", flush=True)

    avg = np.mean([len(v) for v in merged.values()]) if merged else 0
    print(f"  TF-IDF done: avg {avg:.1f} cands/S1, {time.time()-t0:.1f}s", flush=True)
    return dict(merged)


def token_block(s1_df: pd.DataFrame,
                s23_df: pd.DataFrame,
                max_block: int = MAX_TOKEN_BLOCK,
                min_tok: int   = MIN_TOK_LEN) -> dict:
    """Token-based inverted-index blocking."""
    print("  Token blocking...", flush=True)
    t0 = time.time()
    inv = defaultdict(set)
    for eid, name, country in zip(s23_df["entity_id"].values,
                                   s23_df["name_norm"].values,
                                   s23_df["country"].values):
        key_sfx = "_" + country.strip().lower()
        for tok in name.split():
            if len(tok) >= min_tok:
                inv[tok + key_sfx].add(eid)

    inv = {k: v for k, v in inv.items() if len(v) <= max_block}

    cands = {}
    for s1_id, name, country in zip(s1_df["entity_id"].values,
                                     s1_df["name_norm"].values,
                                     s1_df["country"].values):
        cset   = set()
        key_sfx = "_" + country.strip().lower()
        for tok in name.split():
            if len(tok) >= min_tok:
                k = tok + key_sfx
                if k in inv:
                    cset.update(inv[k])
        cands[s1_id] = cset

    avg = np.mean([len(v) for v in cands.values()]) if cands else 0
    print(f"  Token done: avg {avg:.1f} cands/S1, {time.time()-t0:.1f}s", flush=True)
    return cands


def score_pairs(inf_df: pd.DataFrame, model, feature_cols: list,
                batch_size: int = SCORE_BATCH) -> np.ndarray:
    """Score candidate pairs with the trained model in batches."""
    all_scores = []
    n = len(inf_df)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = inf_df.iloc[start:end]
        feats = compute_features_batch(batch)
        for col in feature_cols:
            if col not in feats.columns:
                feats[col] = 0.0
        X = np.nan_to_num(feats[feature_cols].values.astype(np.float32))
        all_scores.append(model.predict(X))
        if (start // batch_size + 1) % 5 == 0:
            print(f"    scored {end:,}/{n:,} pairs", flush=True)
    return np.concatenate(all_scores)


def conflict_resolve(inf_df: pd.DataFrame,
                     scores: np.ndarray,
                     threshold: float,
                     all_s1_ids: list,
                     existing_s23_best: dict) -> tuple:
    """
    Apply conflict resolution globally:
    each S23 ID goes to at most ONE S1 entity (the highest scorer above threshold).
    Merges with `existing_s23_best` dict from prior countries.
    Returns updated s23_best dict and {s1_id: set(matched)} for this batch.
    """
    s1_arr  = inf_df["s1_id"].values
    s23_arr = inf_df["s23_id"].values

    for idx in range(len(scores)):
        if scores[idx] < threshold:
            continue
        s1_id  = s1_arr[idx]
        s23_id = s23_arr[idx]
        if s23_id not in existing_s23_best or scores[idx] > existing_s23_best[s23_id][1]:
            existing_s23_best[s23_id] = (s1_id, scores[idx])

    return existing_s23_best


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_test_inference(base_dir: str = ".",
                       output_dir: str = "output"):
    t_start = time.time()
    os.makedirs(output_dir, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────────
    model_path = os.path.join(output_dir, "model.pkl")
    print(f"Loading model from {model_path}...")
    with open(model_path, "rb") as f:
        config = pickle.load(f)
    model        = config["model"]
    feature_cols = config["feature_cols"]
    threshold    = config["threshold"]
    print(f"  threshold={threshold:.3f}  val_F0.5={config['val_f05']:.4f}")
    print(f"  features: {feature_cols}")

    # ── Load ALL test S1 (needed to ensure every entity appears in output) ──
    print("\nLoading all test S1...")
    test_s1 = load_source_full(
        os.path.join(base_dir, "dataset/test/test_source1.tsv"))
    print(f"  Test S1: {len(test_s1):,} rows")
    test_s1 = normalize_df(test_s1, "test S1")
    all_test_s1_ids = test_s1["entity_id"].tolist()

    countries = sorted(test_s1["country"].unique())
    print(f"  Countries: {countries}")

    # ── Global accumulators ──────────────────────────────────────────────────
    # s23_best: s23_id -> (s1_id, best_score)  — enforces 1-to-1 constraint
    s23_best: dict      = {}
    # candidate sets per S1 (for candidate_pairs.tsv)
    test_candidates: dict = {s1_id: set() for s1_id in all_test_s1_ids}

    # ── Per-country processing ───────────────────────────────────────────────
    for country in countries:
        print(f"\n{'─'*60}")
        print(f"Processing country: {country}")
        t_c = time.time()

        s1_country = test_s1[test_s1["country"] == country].copy()
        s1_country = s1_country.reset_index(drop=True)
        print(f"  S1 entities: {len(s1_country):,}")

        # Load S23 for this country in chunks
        s23_parts = []
        for fname in ("test_source2.tsv", "test_source3.tsv"):
            fpath = os.path.join(base_dir, "dataset/test", fname)
            for chunk in pd.read_csv(fpath, sep="\t", dtype=str,
                                     keep_default_na=False, na_values=[],
                                     chunksize=S23_LOAD_CHUNK):
                chunk = chunk.fillna("")
                sub   = chunk[chunk["country"] == country]
                if len(sub) > 0:
                    s23_parts.append(sub)
            gc.collect()

        if not s23_parts:
            print(f"  No S23 records for {country} — all entities treated as singletons")
            continue

        s23_country = pd.concat(s23_parts, ignore_index=True)
        del s23_parts; gc.collect()
        print(f"  S23 entities: {len(s23_country):,}")
        s23_country = normalize_df(s23_country, f"{country} S23")

        # ── Blocking ────────────────────────────────────────────────────────
        cands_tfidf = tfidf_block_chunked(s1_country, s23_country,
                                          batch_size=S23_TFIDF_BATCH, top_k=TOP_K)
        cands_token = token_block(s1_country, s23_country)

        # Union
        country_cands: dict = {}
        for s1_id in s1_country["entity_id"]:
            country_cands[s1_id] = (
                cands_tfidf.get(s1_id, set()) | cands_token.get(s1_id, set())
            )
            test_candidates[s1_id] = country_cands[s1_id]

        del cands_tfidf, cands_token; gc.collect()

        avg_cands = np.mean([len(v) for v in country_cands.values()])
        total_pairs = sum(len(v) for v in country_cands.values())
        print(f"  avg {avg_cands:.1f} candidates/S1, {total_pairs:,} total pairs")

        if total_pairs == 0:
            del s23_country; gc.collect()
            continue

        # ── Build inference DataFrame ────────────────────────────────────────
        s1_lkp  = s1_country.set_index("entity_id").to_dict("index")
        s23_lkp = s23_country.set_index("entity_id").to_dict("index")

        rows = []
        for s1_id, cids in country_cands.items():
            if s1_id not in s1_lkp:
                continue
            si = s1_lkp[s1_id]
            for cid in cids:
                if cid not in s23_lkp:
                    continue
                ci = s23_lkp[cid]
                rows.append({
                    "s1_id":      s1_id,
                    "s23_id":     cid,
                    "name1_norm": si.get("name_norm", ""),
                    "name2_norm": ci.get("name_norm", ""),
                    "addr1_norm": si.get("addr_norm", ""),
                    "addr2_norm": ci.get("addr_norm", ""),
                    "country1":   si.get("country", ""),
                    "country2":   ci.get("country", ""),
                })
        del s23_country, s1_lkp, s23_lkp; gc.collect()

        inf_df = pd.DataFrame(rows)
        del rows; gc.collect()
        print(f"  Inference pairs: {len(inf_df):,}", flush=True)

        # ── Score ────────────────────────────────────────────────────────────
        print("  Scoring...", flush=True)
        scores = score_pairs(inf_df, model, feature_cols)

        # ── Conflict resolution (global) ─────────────────────────────────────
        s23_best = conflict_resolve(inf_df, scores, threshold,
                                    all_test_s1_ids, s23_best)

        del inf_df, scores; gc.collect()
        print(f"  Country done in {time.time()-t_c:.1f}s", flush=True)

    # ── Build final predictions from global s23_best ─────────────────────────
    print("\nBuilding final predictions...", flush=True)
    test_predictions: dict = {s1_id: set() for s1_id in all_test_s1_ids}
    for s23_id, (s1_id, _) in s23_best.items():
        if s1_id in test_predictions:
            test_predictions[s1_id].add(s23_id)

    n_matched   = sum(1 for v in test_predictions.values() if v)
    n_singleton = sum(1 for v in test_predictions.values() if not v)
    total_links = sum(len(v) for v in test_predictions.values())
    print(f"  {n_matched:,} entities with matches, {n_singleton:,} singletons")
    print(f"  Total match links: {total_links:,}")

    # ── Write outputs ─────────────────────────────────────────────────────────
    print("\nWriting output files...", flush=True)

    match_path = os.path.join(output_dir, "matching_results.tsv")
    with open(match_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in sorted(test_predictions.keys()):
            matched = test_predictions[s1_id]
            f.write(f"{s1_id}\t{','.join(sorted(matched)) if matched else ''}\n")
    print(f"  Wrote {len(test_predictions):,} rows → {match_path}")

    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")
    with open(cand_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in sorted(test_candidates.keys()):
            cands = test_candidates[s1_id]
            f.write(f"{s1_id}\t{','.join(sorted(cands)) if cands else ''}\n")
    print(f"  Wrote {len(test_candidates):,} rows → {cand_path}")

    # ── Validate ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("VALIDATION")
    print(f"{'='*60}")
    import subprocess
    result = subprocess.run(
        [sys.executable,
         os.path.join(base_dir, "utils/validate_submission.py"),
         "--matching",   match_path,
         "--candidate",  cand_path,
         "--test-dir",   os.path.join(base_dir, "dataset/test")],
        capture_output=True, text=True, encoding="utf-8",
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:2000])

    elapsed = time.time() - t_start
    print(f"\nTotal inference time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    return result.returncode == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Test inference (memory-safe)")
    ap.add_argument("--base-dir",    default=".")
    ap.add_argument("--output-dir",  default="output")
    args = ap.parse_args()
    ok = run_test_inference(args.base_dir, args.output_dir)
    sys.exit(0 if ok else 1)
