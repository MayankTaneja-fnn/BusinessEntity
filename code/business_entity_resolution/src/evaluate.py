"""
evaluate.py — Evaluation module for entity resolution.

Computes the F_0.5 metric (precision-heavy) per entity and macro-averaged.

F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)

Singletons count:
- True singleton predicted as singleton: F_0.5 = 1.0
- True singleton predicted with any match: F_0.5 = 0.0
- True non-singleton predicted as singleton: F_0.5 = 0.0
"""

import sys
import pandas as pd
from typing import Dict, Set, Tuple


def compute_f05_per_entity(predicted_ids: set, true_ids: set) -> float:
    """
    Compute F_0.5 for a single entity.
    
    Args:
        predicted_ids: set of predicted match IDs
        true_ids: set of true match IDs
    
    Returns:
        F_0.5 score for this entity
    """
    # Both empty: correctly identified as singleton
    if not predicted_ids and not true_ids:
        return 1.0
    # Predicted empty but has true matches: missed all
    if not predicted_ids:
        return 0.0
    # Predicted non-empty but true is empty: false merge on singleton
    if not true_ids:
        return 0.0

    # Compute precision and recall
    intersection = predicted_ids & true_ids
    precision = len(intersection) / len(predicted_ids)
    recall = len(intersection) / len(true_ids)

    # F_0.5
    if precision + recall == 0:
        return 0.0
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
    return f05


def compute_macro_f05(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> Tuple[float, float, float, dict]:
    """
    Compute macro-averaged F_0.5 across all entities.
    
    Args:
        predictions: {s1_id: set(predicted_match_ids)}
        ground_truth: {s1_id: set(true_match_ids)}
    
    Returns:
        (macro_f05, avg_precision, avg_recall, per_entity_scores)
    """
    per_entity = {}
    precisions = []
    recalls = []

    all_s1_ids = set(ground_truth.keys()) | set(predictions.keys())

    for s1_id in all_s1_ids:
        pred = predictions.get(s1_id, set())
        true = ground_truth.get(s1_id, set())

        f05 = compute_f05_per_entity(pred, true)
        per_entity[s1_id] = f05

        # Also track precision/recall for reporting
        if not pred and not true:
            precisions.append(1.0)
            recalls.append(1.0)
        elif not pred:
            precisions.append(0.0)
            recalls.append(0.0)
        elif not true:
            precisions.append(0.0)
            recalls.append(0.0)
        else:
            inter = pred & true
            precisions.append(len(inter) / len(pred))
            recalls.append(len(inter) / len(true))

    n = len(per_entity)
    macro_f05 = sum(per_entity.values()) / n if n > 0 else 0.0
    avg_prec = sum(precisions) / n if n > 0 else 0.0
    avg_rec = sum(recalls) / n if n > 0 else 0.0

    return macro_f05, avg_prec, avg_rec, per_entity


def load_ground_truth(gt_path: str) -> Dict[str, Set[str]]:
    """
    Parse train_ground_truth.tsv into {s1_id: set(matched_ids)} dict.
    Empty/NaN matched_entity_ids -> empty set.
    """
    gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    truth = {}
    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        matched = row.get("matched_entity_ids", "")
        if not matched or matched.strip() == "" or matched.lower() == "nan":
            truth[s1_id] = set()
        else:
            truth[s1_id] = set(matched.strip().split(","))
    return truth


def load_predictions(pred_path: str) -> Dict[str, Set[str]]:
    """
    Parse matching_results.tsv into {s1_id: set(matched_ids)} dict.
    """
    df = pd.read_csv(pred_path, sep="\t", dtype=str, keep_default_na=False)
    preds = {}
    for _, row in df.iterrows():
        s1_id = row["source1_entity_id"]
        matched = row.get("matched_entity_ids", "")
        if not matched or matched.strip() == "" or matched.lower() == "nan":
            preds[s1_id] = set()
        else:
            preds[s1_id] = set(matched.strip().split(","))
    return preds


def evaluate_from_files(pred_path: str, gt_path: str) -> dict:
    """
    Load predictions and ground truth from files, compute macro F_0.5.
    
    Returns: dict with 'f05', 'precision', 'recall', 'n_entities'
    """
    predictions = load_predictions(pred_path)
    ground_truth = load_ground_truth(gt_path)

    f05, prec, rec, per_entity = compute_macro_f05(predictions, ground_truth)

    return {
        "f05": f05,
        "precision": prec,
        "recall": rec,
        "n_entities": len(per_entity),
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python evaluate.py <predictions.tsv> <ground_truth.tsv>")
        sys.exit(1)

    pred_path = sys.argv[1]
    gt_path = sys.argv[2]

    results = evaluate_from_files(pred_path, gt_path)
    print(f"\n=== EVALUATION RESULTS ===")
    print(f"  Entities evaluated: {results['n_entities']}")
    print(f"  Macro Precision:    {results['precision']:.4f}")
    print(f"  Macro Recall:       {results['recall']:.4f}")
    print(f"  Macro F_0.5:        {results['f05']:.4f}")
