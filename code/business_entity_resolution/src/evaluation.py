"""
evaluation.py
Comprehensive Evaluation & Validation Metrics for Business Entity Resolution.

Features:
1. Exact Macro F_0.5 Validator matching challenge specifications (including singleton credit/penalty).
2. Blocking Recall Ceiling & Reduction Ratio measurement.
3. Fine-grained Diagnostic Breakdown (Singletons vs. Non-singletons, Country-wise, Micro vs. Macro).
"""

import os
import sys
from typing import Dict, Set, List, Optional, Any, Tuple
import numpy as np


def compute_entity_f05(y_true: Set[str], y_pred: Set[str]) -> float:
    """
    Computes F_0.5 score for a single Source 1 entity.
    
    Formula:
        F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
              = (5.0 * TP) / (|y_true| + 4.0 * |y_pred|)
              
    Singleton Rules:
        - True singleton (|y_true| == 0) and predicted empty (|y_pred| == 0) -> 1.0
        - True singleton (|y_true| == 0) and predicted non-empty (|y_pred| > 0) -> 0.0
        - Non-singleton (|y_true| > 0) and predicted empty (|y_pred| == 0) -> 0.0
    """
    len_true = len(y_true)
    len_pred = len(y_pred)

    # Singleton evaluation
    if len_true == 0:
        return 1.0 if len_pred == 0 else 0.0

    if len_pred == 0:
        return 0.0

    tp = len(y_true & y_pred)
    if tp == 0:
        return 0.0

    # Correct F_0.5 formula
    return (5.0 * tp) / (float(len_true) + 4.0 * float(len_pred))


def compute_macro_f05(ground_truth: Dict[str, Set[str]], predictions: Dict[str, Set[str]]) -> float:
    """
    Calculates the Macro-averaged F_0.5 across all Source 1 entities in ground truth.
    """
    if not ground_truth:
        return 0.0

    total_score = sum(
        compute_entity_f05(y_true, predictions.get(sid, set()))
        for sid, y_true in ground_truth.items()
    )
    return total_score / float(len(ground_truth))


def evaluate_blocking_recall(
    ground_truth: Dict[str, Set[str]],
    candidate_pairs: Dict[str, Set[str]],
    total_s2_pool_size: Optional[int] = None,
    total_s3_pool_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Measures the Blocking Recall Ceiling and candidate reduction efficiency.
    
    Returns:
        - recall_ceiling: Fraction of all true matches present in candidate sets.
        - entity_coverage_all: Fraction of non-singleton entities where ALL true matches were found.
        - entity_coverage_any: Fraction of non-singleton entities where AT LEAST ONE true match was found.
        - avg_candidates_per_entity: Mean candidate set size.
        - reduction_ratio: Percentage reduction over Cartesian product (if pool sizes provided).
    """
    total_true_matches = 0
    retained_matches = 0
    non_singleton_entities = 0
    full_coverage_entities = 0
    partial_coverage_entities = 0
    candidate_counts = []

    for sid, y_true in ground_truth.items():
        candidates = candidate_pairs.get(sid, set())
        candidate_counts.append(len(candidates))

        len_true = len(y_true)
        if len_true > 0:
            non_singleton_entities += 1
            total_true_matches += len_true
            captured = len(y_true & candidates)
            retained_matches += captured

            if captured == len_true:
                full_coverage_entities += 1
            if captured > 0:
                partial_coverage_entities += 1

    recall_ceiling = (retained_matches / total_true_matches) if total_true_matches > 0 else 1.0
    entity_full_cov = (full_coverage_entities / non_singleton_entities) if non_singleton_entities > 0 else 1.0
    entity_any_cov = (partial_coverage_entities / non_singleton_entities) if non_singleton_entities > 0 else 1.0
    avg_cands = float(np.mean(candidate_counts)) if candidate_counts else 0.0

    report = {
        "blocking_recall_ceiling": recall_ceiling,
        "total_true_matches": total_true_matches,
        "retained_in_candidates": retained_matches,
        "missed_in_blocking": total_true_matches - retained_matches,
        "non_singleton_entities": non_singleton_entities,
        "entity_full_coverage_rate": entity_full_cov,
        "entity_partial_coverage_rate": entity_any_cov,
        "avg_candidates_per_s1": avg_cands,
        "max_candidates_per_s1": max(candidate_counts) if candidate_counts else 0,
    }

    if total_s2_pool_size is not None and total_s3_pool_size is not None:
        total_possible_pairs = len(ground_truth) * (total_s2_pool_size + total_s3_pool_size)
        total_actual_pairs = sum(candidate_counts)
        reduction_ratio = 1.0 - (total_actual_pairs / total_possible_pairs) if total_possible_pairs > 0 else 1.0
        report["reduction_ratio"] = reduction_ratio

    return report


def detailed_evaluation_report(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    candidate_pairs: Optional[Dict[str, Set[str]]] = None,
    countries: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Produces a full diagnostic report across all entity segments.
    """
    singleton_scores = []
    non_singleton_scores = []
    country_scores: Dict[str, List[float]] = {}
    
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for sid, y_true in ground_truth.items():
        y_pred = predictions.get(sid, set())
        score = compute_entity_f05(y_true, y_pred)

        if len(y_true) == 0:
            singleton_scores.append(score)
        else:
            non_singleton_scores.append(score)

        if countries and sid in countries:
            c = countries[sid]
            country_scores.setdefault(c, []).append(score)

        # Micro-level counters
        tp = len(y_true & y_pred)
        fp = len(y_pred - y_true)
        fn = len(y_true - y_pred)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    macro_f05 = compute_macro_f05(ground_truth, predictions)
    singleton_acc = float(np.mean(singleton_scores)) if singleton_scores else 1.0
    non_singleton_f05 = float(np.mean(non_singleton_scores)) if non_singleton_scores else 0.0

    # Global Micro metrics
    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f05 = (1.25 * micro_prec * micro_rec) / (0.25 * micro_prec + micro_rec) if (0.25 * micro_prec + micro_rec) > 0 else 0.0

    report = {
        "macro_f05": macro_f05,
        "singleton_count": len(singleton_scores),
        "singleton_accuracy": singleton_acc,
        "non_singleton_count": len(non_singleton_scores),
        "non_singleton_macro_f05": non_singleton_f05,
        "global_micro_precision": micro_prec,
        "global_micro_recall": micro_rec,
        "global_micro_f05": micro_f05,
        "by_country": {c: float(np.mean(scores)) for c, scores in country_scores.items()} if country_scores else {},
    }

    if candidate_pairs:
        report["blocking_diagnostics"] = evaluate_blocking_recall(ground_truth, candidate_pairs)

    return report


if __name__ == "__main__":
    # Self-test using the example from Page 6 of the challenge document
    print("Running evaluation metric unit tests...")
    
    # Test 1: Example from problem statement
    # GT: S1-00001 -> [S2-00047, S3-00812]
    # Pred: S1-00001 -> [S2-00047, S2-00193, S3-00812]
    # Precision = 2/3, Recall = 2/2 = 1.0 -> F_0.5 = 0.7142857...
    score_ex = compute_entity_f05({'S2-00047', 'S3-00812'}, {'S2-00047', 'S2-00193', 'S3-00812'})
    assert abs(score_ex - (5.0 * 2) / (2 + 4 * 3)) < 1e-6
    assert round(score_ex, 3) == 0.714
    print(f"✓ Test 1 (Problem statement example): score = {score_ex:.4f} (expected 0.714)")

    # Test 2: Singleton correct empty prediction -> 1.0
    score_sing_correct = compute_entity_f05(set(), set())
    assert score_sing_correct == 1.0
    print("✓ Test 2 (Singleton correct): score = 1.0")

    # Test 3: Singleton false positive -> 0.0
    score_sing_fp = compute_entity_f05(set(), {'S2-00001'})
    assert score_sing_fp == 0.0
    print("✓ Test 3 (Singleton false positive): score = 0.0")

    # Test 4: Non-singleton missed completely -> 0.0
    score_miss = compute_entity_f05({'S2-00001'}, set())
    assert score_miss == 0.0
    print("✓ Test 4 (Non-singleton missed): score = 0.0")

    # Test 5: Macro averaging
    gt = {
        'S1-1': {'S2-00047', 'S3-00812'},
        'S1-2': set(),
        'S1-3': set(),
        'S1-4': {'S2-10000'}
    }
    pred = {
        'S1-1': {'S2-00047', 'S2-00193', 'S3-00812'}, # 0.7142857
        'S1-2': set(),                                  # 1.0
        'S1-3': {'S3-99999'},                          # 0.0 (false merge on singleton)
        'S1-4': {'S2-10000'}                           # 1.0
    }
    macro_score = compute_macro_f05(gt, pred)
    expected_macro = (score_ex + 1.0 + 0.0 + 1.0) / 4.0
    assert abs(macro_score - expected_macro) < 1e-6
    print(f"✓ Test 5 (Macro F0.5): score = {macro_score:.4f}")

    print("\nAll evaluation metric unit tests passed successfully!")
