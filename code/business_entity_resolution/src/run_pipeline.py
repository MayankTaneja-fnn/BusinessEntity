"""
run_pipeline.py — Main orchestrator for the entity resolution pipeline.

Runs all stages end-to-end:
  Stage 0: Data exploration & setup
  Stage 1: Normalization (integrated into blocking)
  Stage 2: Blocking / candidate generation
  Stage 3: Feature engineering (integrated into training)
  Stage 4: Model training
  Stage 5: Threshold optimization & conflict resolution (integrated into training/inference)
  Stage 6: Inference & output

Usage:
    python src/run_pipeline.py --base-dir <path_to_student_resource> --output-dir output
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser(description="Run full entity resolution pipeline")
    parser.add_argument("--base-dir", default=".", help="Path to student_resource/ directory")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k for TF-IDF blocking")
    parser.add_argument("--neg-ratio", type=int, default=3, help="Negative to positive pair ratio")
    parser.add_argument("--max-pairs", type=int, default=5000000, help="Max training pairs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-train", action="store_true", help="Skip training, use existing model")
    parser.add_argument("--skip-blocking", action="store_true", help="Skip blocking, use existing candidates")
    args = parser.parse_args()

    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    # =====================================================================
    # Stage 0: Exploration (print stats)
    # =====================================================================
    print(f"\n{'='*60}")
    print("STAGE 0: DATA EXPLORATION")
    print(f"{'='*60}")

    import pandas as pd
    for split, prefix in [("train", "train"), ("test", "test")]:
        d = os.path.join(args.base_dir, "dataset", split)
        for src in ["source1", "source2", "source3"]:
            fpath = os.path.join(d, f"{prefix}_{src}.tsv")
            if os.path.exists(fpath):
                # Count lines without loading entire file
                with open(fpath, encoding="utf-8") as f:
                    n_lines = sum(1 for _ in f) - 1
                print(f"  {prefix}_{src}.tsv: {n_lines:,} rows")

    gt_path = os.path.join(args.base_dir, "dataset/train/train_ground_truth.tsv")
    if os.path.exists(gt_path):
        gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
        print(f"  train_ground_truth.tsv: {len(gt):,} rows")
        gt["n_matches"] = gt["matched_entity_ids"].apply(
            lambda x: 0 if not x or x.strip() == "" else len(x.split(","))
        )
        singleton_rate = (gt["n_matches"] == 0).mean()
        print(f"  Singleton rate: {singleton_rate:.4f} ({(gt['n_matches']==0).sum():,} entities)")
        print(f"  Match count distribution:")
        print(gt["n_matches"].value_counts().sort_index().to_string())

    # =====================================================================
    # Stage 2: Blocking (includes Stage 1 normalization)
    # =====================================================================
    if not args.skip_blocking:
        print(f"\n{'='*60}")
        print("STAGE 1+2: NORMALIZATION & BLOCKING (Train)")
        print(f"{'='*60}")

        from blocking import run_blocking
        run_blocking(
            s1_path=os.path.join(args.base_dir, "dataset/train/train_source1.tsv"),
            s2_path=os.path.join(args.base_dir, "dataset/train/train_source2.tsv"),
            s3_path=os.path.join(args.base_dir, "dataset/train/train_source3.tsv"),
            gt_path=gt_path,
            output_dir=args.output_dir,
            top_k=args.top_k,
            mode="train",
        )
    else:
        print("\n  Skipping train blocking (using existing candidates)")

    # =====================================================================
    # Stage 3+4: Feature Engineering & Model Training
    # =====================================================================
    if not args.skip_train:
        print(f"\n{'='*60}")
        print("STAGE 3+4: FEATURE ENGINEERING & MODEL TRAINING")
        print(f"{'='*60}")

        from train import train_model
        config = train_model(
            base_dir=args.base_dir,
            output_dir=args.output_dir,
            neg_ratio=args.neg_ratio,
            max_pairs=args.max_pairs,
            seed=args.seed,
        )
    else:
        print("\n  Skipping training (using existing model)")

    # =====================================================================
    # Stage 5+6: Inference & Output
    # =====================================================================
    print(f"\n{'='*60}")
    print("STAGE 5+6: INFERENCE & OUTPUT")
    print(f"{'='*60}")

    from infer import run_inference
    run_inference(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        top_k=args.top_k,
    )

    # =====================================================================
    # Validation
    # =====================================================================
    print(f"\n{'='*60}")
    print("VALIDATION")
    print(f"{'='*60}")

    validate_script = os.path.join(args.base_dir, "utils/validate_submission.py")
    matching_path = os.path.join(args.output_dir, "matching_results.tsv")
    candidate_path = os.path.join(args.output_dir, "candidate_pairs.tsv")
    test_dir = os.path.join(args.base_dir, "dataset/test")

    import subprocess
    result = subprocess.run(
        [
            sys.executable, validate_script,
            "--matching", matching_path,
            "--candidate", candidate_path,
            "--test-dir", test_dir,
        ],
        capture_output=True, text=True, encoding="utf-8",
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    if result.returncode != 0:
        print("⚠️  VALIDATION FAILED — fix issues above before submitting!")
    else:
        print("✅  VALIDATION PASSED")

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE — Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
