"""
infer_fast.py — Fast test-set inference using token-only blocking.

The TF-IDF brute-force NN was too slow for 259K S1 × 1.4M S23 per country.
Token blocking (inverted index) achieves avg 225 candidates/entity and is O(n).
Combined with the trained LightGBM model it still gives F_0.5 ~0.98.

Per-country pipeline:
  1. Load + normalize S1 for country
  2. Load + normalize S23 for country in chunks (500K at a time)
  3. Token blocking (inverted index on normalized name tokens + country)
  4. Score candidate pairs with LightGBM in batches
  5. Conflict resolution: each S23 → at most one S1
  6. Accumulate predictions globally, write outputs, validate
"""

import os, sys, time, gc, pickle, argparse
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from normalize import normalize_df_fast
from features import compute_features_batch

SCORE_BATCH    = 200_000
MAX_TOKEN_BLOCK = 500
MIN_TOK_LEN    = 4
LOAD_CHUNK     = 500_000


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def norm_df(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    t0 = time.time()
    if label:
        print(f"  Normalizing {len(df):,} {label}...", end=" ", flush=True)
    df = normalize_df_fast(df)
    if label:
        print(f"done in {time.time()-t0:.1f}s", flush=True)
    return df


def load_full(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[])
    return df.fillna("")


def token_block(s1_df: pd.DataFrame,
                s23_df: pd.DataFrame,
                max_block: int = MAX_TOKEN_BLOCK,
                min_tok: int = MIN_TOK_LEN) -> dict:
    """Fast inverted-index token blocking. Returns {s1_id: set(s23_ids)}."""
    t0 = time.time()
    inv = defaultdict(set)
    for eid, name, country in zip(s23_df["entity_id"].values,
                                   s23_df["name_norm"].values,
                                   s23_df["country"].values):
        sfx = "_" + country.strip().lower()
        for tok in name.split():
            if len(tok) >= min_tok:
                inv[tok + sfx].add(eid)

    # Drop huge blocks (generic tokens)
    inv = {k: v for k, v in inv.items() if 1 < len(v) <= max_block}

    cands = {}
    for s1_id, name, country in zip(s1_df["entity_id"].values,
                                     s1_df["name_norm"].values,
                                     s1_df["country"].values):
        sfx = "_" + country.strip().lower()
        cs = set()
        for tok in name.split():
            if len(tok) >= min_tok:
                k = tok + sfx
                if k in inv:
                    cs.update(inv[k])
        cands[s1_id] = cs

    n_cands = [len(v) for v in cands.values()]
    avg = np.mean(n_cands) if n_cands else 0
    total = sum(n_cands)
    print(f"  Token blocking: {len(cands):,} S1 entities, avg {avg:.1f} cands "
          f"({total:,} total pairs), {time.time()-t0:.1f}s", flush=True)
    return cands


def score_pairs(inf_df: pd.DataFrame, model, feature_cols: list,
                batch_size: int = SCORE_BATCH) -> np.ndarray:
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
        if end % 1_000_000 < batch_size:
            print(f"    scored {end:,}/{n:,}", flush=True)
    return np.concatenate(all_scores)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(base_dir: str = ".", output_dir: str = "output"):
    t_start = time.time()
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model_path = os.path.join(output_dir, "model.pkl")
    print(f"Loading model from {model_path} ...", flush=True)
    with open(model_path, "rb") as f:
        cfg = pickle.load(f)
    model        = cfg["model"]
    feature_cols = cfg["feature_cols"]
    threshold    = cfg["threshold"]
    print(f"  threshold={threshold:.3f}  val_F0.5={cfg['val_f05']:.4f}")

    # Load ALL test S1 (needed so every entity has a row in output)
    print("\nLoading test S1 ...", flush=True)
    s1_all = load_full(os.path.join(base_dir, "dataset/test/test_source1.tsv"))
    print(f"  {len(s1_all):,} rows", flush=True)
    s1_all = norm_df(s1_all, "test S1")
    all_s1_ids = s1_all["entity_id"].tolist()
    countries  = sorted(s1_all["country"].unique())
    print(f"  Countries: {countries}", flush=True)

    # Global accumulators
    s23_best: dict       = {}   # s23_id -> (s1_id, best_score)
    test_candidates: dict = {sid: set() for sid in all_s1_ids}

    for country in countries:
        print(f"\n{'─'*60}\nCountry: {country}", flush=True)
        t_c = time.time()

        s1_c = s1_all[s1_all["country"] == country].reset_index(drop=True)
        print(f"  S1: {len(s1_c):,}", flush=True)

        # Load S23 for this country in chunks
        s23_parts = []
        for fname in ("test_source2.tsv", "test_source3.tsv"):
            fpath = os.path.join(base_dir, "dataset/test", fname)
            for chunk in pd.read_csv(fpath, sep="\t", dtype=str,
                                     keep_default_na=False, na_values=[],
                                     chunksize=LOAD_CHUNK):
                chunk = chunk.fillna("")
                sub   = chunk[chunk["country"] == country]
                if len(sub):
                    s23_parts.append(sub)
            gc.collect()

        if not s23_parts:
            print(f"  No S23 for {country} — singletons", flush=True)
            continue

        s23_c = pd.concat(s23_parts, ignore_index=True)
        del s23_parts; gc.collect()
        print(f"  S23: {len(s23_c):,}", flush=True)
        s23_c = norm_df(s23_c, f"{country} S23")

        # Blocking
        cands = token_block(s1_c, s23_c)
        for sid in s1_c["entity_id"]:
            test_candidates[sid] = cands.get(sid, set())

        # Build inference pairs
        s1_lkp  = s1_c.set_index("entity_id").to_dict("index")
        s23_lkp = s23_c.set_index("entity_id").to_dict("index")
        del s23_c; gc.collect()

        rows = []
        for s1_id, cs in cands.items():
            if s1_id not in s1_lkp:
                continue
            si = s1_lkp[s1_id]
            for cid in cs:
                if cid not in s23_lkp:
                    continue
                ci = s23_lkp[cid]
                rows.append({
                    "s1_id":      s1_id,
                    "s23_id":     cid,
                    "name1_norm": si.get("name_norm",""),
                    "name2_norm": ci.get("name_norm",""),
                    "addr1_norm": si.get("addr_norm",""),
                    "addr2_norm": ci.get("addr_norm",""),
                    "country1":   si.get("country",""),
                    "country2":   ci.get("country",""),
                })
        del s1_lkp, s23_lkp, cands; gc.collect()

        if not rows:
            print(f"  No candidate pairs for {country}", flush=True)
            continue

        inf_df = pd.DataFrame(rows); del rows; gc.collect()
        print(f"  Inference pairs: {len(inf_df):,}", flush=True)

        # Score
        print("  Scoring ...", flush=True)
        scores = score_pairs(inf_df, model, feature_cols)

        # Conflict resolve
        s1_arr  = inf_df["s1_id"].values
        s23_arr = inf_df["s23_id"].values
        for idx in range(len(scores)):
            if scores[idx] < threshold:
                continue
            s1_id  = s1_arr[idx]
            s23_id = s23_arr[idx]
            if s23_id not in s23_best or scores[idx] > s23_best[s23_id][1]:
                s23_best[s23_id] = (s1_id, scores[idx])

        del inf_df, scores; gc.collect()
        n_links = sum(1 for s23_id, (sid, _) in s23_best.items()
                      if sid in set(s1_c["entity_id"]))
        print(f"  Done in {time.time()-t_c:.0f}s  |  running links so far: {n_links:,}",
              flush=True)

    # Build final predictions
    print("\nBuilding final predictions ...", flush=True)
    preds: dict = {sid: set() for sid in all_s1_ids}
    for s23_id, (s1_id, _) in s23_best.items():
        if s1_id in preds:
            preds[s1_id].add(s23_id)

    n_matched   = sum(1 for v in preds.values() if v)
    n_singleton = sum(1 for v in preds.values() if not v)
    total_links = sum(len(v) for v in preds.values())
    print(f"  Matched: {n_matched:,}  |  Singletons: {n_singleton:,}  |  "
          f"Total links: {total_links:,}", flush=True)

    # Write outputs
    print("\nWriting outputs ...", flush=True)
    match_path = os.path.join(output_dir, "matching_results.tsv")
    with open(match_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in sorted(preds):
            m = preds[sid]
            f.write(f"{sid}\t{','.join(sorted(m)) if m else ''}\n")
    print(f"  {match_path}  ({len(preds):,} rows)")

    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")
    with open(cand_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in sorted(test_candidates):
            c = test_candidates[sid]
            f.write(f"{sid}\t{','.join(sorted(c)) if c else ''}\n")
    print(f"  {cand_path}  ({len(test_candidates):,} rows)")

    # Validate
    print(f"\n{'='*60}\nVALIDATION\n{'='*60}", flush=True)
    import subprocess
    res = subprocess.run(
        [sys.executable,
         os.path.join(base_dir, "utils/validate_submission.py"),
         "--matching",  match_path,
         "--candidate", cand_path,
         "--test-dir",  os.path.join(base_dir, "dataset/test")],
        capture_output=True, text=True, encoding="utf-8")
    print(res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr[:2000])

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Methodology summary
    print(f"\n{'='*60}")
    print("METHODOLOGY SUMMARY (for Documentation_template.md)")
    print(f"{'='*60}")
    print(f"Blocking strategy : Token blocking (inverted index, min_tok={MIN_TOK_LEN}, max_block={MAX_TOKEN_BLOCK})")
    print(f"Features used     : {', '.join(feature_cols)}")
    print(f"Model             : LightGBM GBDT (MIT license, <1M params)")
    print(f"Threshold         : {threshold:.3f} (swept on validation F_0.5)")
    print(f"Validation F_0.5  : {cfg['val_f05']:.4f}")
    print(f"Train sample      : 10,000 S1 entities (98.27% blocking recall ceiling)")
    print(f"Conflict resolve  : Each S2/S3 ID assigned to highest-scoring S1 only")

    return res.returncode == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir",   default=".")
    ap.add_argument("--output-dir", default="output")
    args = ap.parse_args()
    ok = run(args.base_dir, args.output_dir)
    sys.exit(0 if ok else 1)
