"""
train.py
Adaptive Business Entity Resolution Model Training, Validation & Optimization Pipeline.

Key Highlights:
- Task 1: Macro F_0.5 validation exactly matching the challenge specification (including singleton handling).
- Task 2: Explicit measurement of the Blocking Recall Ceiling & candidate reduction ratio.
- Task 3: Hard-negative mining from blocking near-misses and class-imbalance-aware LightGBM training.
- 4 Parallel Views & Dynamic Corpus IDF: Diacritic-stripping NFKD, delimited tokens, compact signatures, and numeric profiles.
- Monotonic constraints preserving physical similarity properties.
- High-throughput inference (>1,000,000 predictions/sec on CPU).
"""

import os
import sys
import time
import json
import math
import pickle
import argparse
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import lightgbm as lgb

# Ensure src directory is in sys.path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from features import (
    FEATURE_NAMES,
    ParsedRecord,
    compute_corpus_idf,
    compute_parsed_features,
    parse_record,
)
from blocking import AdaptiveInvertedIndex
from evaluation import (
    compute_entity_f05,
    compute_macro_f05,
    evaluate_blocking_recall,
    detailed_evaluation_report,
)

# Monotonic constraints: +1 for positive correlation with match probability,
# -1 for conflict, 0 for neutral indicators
MONOTONE_CONSTRAINTS = [
    1,   # name_jw
    1,   # name_jw_compact
    1,   # name_token_sort
    1,   # name_token_set
    1,   # name_char_3gram_jaccard
    1,   # name_char_4gram_jaccard
    1,   # name_idf_weighted_jaccard
    1,   # name_exact_compact
    1,   # addr_token_sort
    1,   # addr_token_set
    1,   # addr_char_3gram_jaccard
    1,   # addr_number_jaccard
    -1,  # addr_number_conflict
    1,   # addr_number_match_count
    0,   # addr_is_null
    0,   # addr_both_null
    0    # is_source2
]


def resolve_data_paths(data_dir: Optional[str] = None) -> str:
    """Auto-detect data directory supporting local project paths and Kaggle paths."""
    candidates = []
    if data_dir:
        candidates.append(data_dir)

    base_dir = os.path.abspath(os.path.join(SRC_DIR, "..", "..", ".."))
    candidates.extend([
        os.path.join(base_dir, "student_resource", "dataset", "train"),
        os.path.join(base_dir, "dataset", "train"),
        os.path.join(base_dir, "train"),
        r"C:\Users\anmol\OneDrive\Desktop\Work\ML_challenge\student_resource\dataset\train",
        "/kaggle/input/amazon-ml-challenge-2026/train",
        "/kaggle/input/dataset/train",
    ])

    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "train_ground_truth.tsv" in files:
                candidates.insert(0, root)
                break

    for p in candidates:
        if p and os.path.exists(os.path.join(p, "train_ground_truth.tsv")):
            print(f"[Data Detector] Found valid training data directory: {p}")
            return p

    raise FileNotFoundError(
        f"Could not find training data. Looked in: {candidates}"
    )


def load_dataset(data_dir: str, sample_size: int = 50000, random_state: int = 42):
    """Load S1, GT, and matching S2/S3 target pools with Polars."""
    print("=" * 70)
    print("STEP 1: LOADING DATASET & GROUND TRUTH")
    print("=" * 70)
    t0 = time.time()

    s1_path = os.path.join(data_dir, "train_source1.tsv")
    gt_path = os.path.join(data_dir, "train_ground_truth.tsv")
    s2_path = os.path.join(data_dir, "train_source2.tsv")
    s3_path = os.path.join(data_dir, "train_source3.tsv")

    print(f"Reading S1 records from: {s1_path}")
    if sample_size > 0:
        s1 = pl.read_csv(s1_path, separator="\t", n_rows=sample_size)
    else:
        s1 = pl.read_csv(s1_path, separator="\t")
    print(f"Loaded {len(s1):,} S1 entities across countries: {s1['country'].unique().to_list()}")

    print(f"Reading Ground Truth from: {gt_path}")
    gt = pl.read_csv(gt_path, separator="\t")
    s1_set = set(s1["entity_id"].to_list())

    # Build Ground Truth Map: S1 -> Set of matched target IDs
    gt_map: Dict[str, Set[str]] = {sid: set() for sid in s1_set}
    all_matched_targets: Set[str] = set()

    for row in gt.iter_rows(named=True):
        sid = row["source1_entity_id"]
        if sid in gt_map:
            raw_matches = row.get("matched_entity_ids")
            if raw_matches is not None and str(raw_matches) != "None" and str(raw_matches) != "nan" and str(raw_matches).strip():
                matches = set(str(raw_matches).strip().split(","))
                gt_map[sid] = matches
                all_matched_targets.update(matches)

    total_singletons = sum(1 for sid, m in gt_map.items() if len(m) == 0)
    total_non_singletons = len(gt_map) - total_singletons
    total_true_matches = sum(len(m) for m in gt_map.values())

    print(f"Ground Truth Analysis for sampled {len(gt_map):,} S1 entities:")
    print(f"  Singletons (0 matches)    : {total_singletons:,} ({total_singletons/len(gt_map)*100:.1f}%)")
    print(f"  Non-Singletons (>=1 match): {total_non_singletons:,} ({total_non_singletons/len(gt_map)*100:.1f}%)")
    print(f"  Total True Match Links    : {total_true_matches:,}")

    # Read Target Pool (S2 and S3)
    print(f"\nReading S2 records from: {s2_path}")
    s2 = pl.read_csv(s2_path, separator="\t")
    print(f"Loaded {len(s2):,} S2 entities")

    print(f"Reading S3 records from: {s3_path}")
    s3 = pl.read_csv(s3_path, separator="\t")
    print(f"Loaded {len(s3):,} S3 entities")

    print(f"Dataset loaded in {time.time() - t0:.2f}s")
    return s1, gt_map, s2, s3


def parse_all_records(s1_df: pl.DataFrame, s2_df: pl.DataFrame, s3_df: pl.DataFrame):
    """Pre-parse text fields across all sources into structured ParsedRecord tuples."""
    print("\n" + "=" * 70)
    print("STEP 2: PRE-PARSING MULTI-VIEW TEXT REPRESENTATIONS")
    print("=" * 70)
    t0 = time.time()

    def _parse_df(df: pl.DataFrame) -> Tuple[List[str], List[ParsedRecord], List[str]]:
        eids = df["entity_id"].to_list()
        names = df["business_name"].to_list()
        addrs = df["business_address"].to_list()
        countries = df["country"].to_list() if "country" in df.columns else [""] * len(eids)

        parsed = [
            parse_record(n, a)
            for n, a in zip(names, addrs)
        ]
        return eids, parsed, countries

    print(f"Parsing {len(s1_df):,} S1 records...")
    s1_ids, s1_parsed, s1_countries = _parse_df(s1_df)

    print(f"Parsing {len(s2_df):,} S2 records...")
    s2_ids, s2_parsed, _ = _parse_df(s2_df)

    print(f"Parsing {len(s3_df):,} S3 records...")
    s3_ids, s3_parsed, _ = _parse_df(s3_df)

    print(f"All records parsed into 4 parallel representations in {time.time() - t0:.2f}s")
    return (s1_ids, s1_parsed, s1_countries), (s2_ids, s2_parsed), (s3_ids, s3_parsed)


def prepare_training_validation_data(
    s1_data: Tuple[List[str], List[ParsedRecord], List[str]],
    s2_data: Tuple[List[str], List[ParsedRecord]],
    s3_data: Tuple[List[str], List[ParsedRecord]],
    gt_map: Dict[str, Set[str]],
    test_size: float = 0.20,
    random_state: int = 42
):
    """Build blocking index, compute dynamic IDF, measure blocking recall ceiling,
    and generate hard-negative training pairs and validation candidate pool.
    """
    s1_ids, s1_parsed, s1_countries = s1_data
    s2_ids, s2_parsed = s2_data
    s3_ids, s3_parsed = s3_data

    # Dynamic Corpus IDF on target pool
    print("\nComputing dynamic corpus IDF on target pool (S2 + S3)...")
    target_parsed = s2_parsed + s3_parsed
    corpus = compute_corpus_idf(target_parsed)
    print(f"Corpus Vocabulary: {len(corpus.df):,} unique tokens")
    print(f"Corpus Stopwords : {len(corpus.stopwords):,} tokens (top frequent domain words)")

    # Build Adaptive Inverted Index
    print("Building adaptive inverted index...")
    t_idx = time.time()
    index = AdaptiveInvertedIndex(max_posting=30, stopwords=corpus.stopwords, df_dict=corpus.df)
    index.add_records(s2_parsed, is_s2_flag=1)
    index.add_records(s3_parsed, is_s2_flag=0)
    print(f"Inverted index built in {time.time() - t_idx:.2f}s with {len(index.index):,} keys")

    # Fast ID lookups
    s2_map = {eid: p for eid, p in zip(s2_ids, s2_parsed)}
    s3_map = {eid: p for eid, p in zip(s3_ids, s3_parsed)}

    # Stratified Train/Val split at S1 entity level (Zero Entity Leakage)
    print("\n" + "=" * 70)
    print("STEP 3: STRATIFIED SPLIT & BLOCKING RECALL CEILING MEASUREMENT")
    print("=" * 70)
    s1_indices = np.arange(len(s1_ids))
    is_singleton = np.array([1 if len(gt_map[sid]) == 0 else 0 for sid in s1_ids])

    train_idx, val_idx = train_test_split(
        s1_indices,
        test_size=test_size,
        random_state=random_state,
        stratify=is_singleton
    )

    train_s1_ids = set(s1_ids[i] for i in train_idx)
    val_s1_ids = set(s1_ids[i] for i in val_idx)
    print(f"Split S1 entities: {len(train_s1_ids):,} Train | {len(val_s1_ids):,} Validation")

    # Measure Validation Blocking Candidates & Recall Ceiling explicitly (TASK 2)
    print("\nQuerying validation blocking index and measuring recall ceiling...")
    val_candidates_dict: Dict[str, Set[str]] = {}
    val_cand_pairs: List[Tuple[str, str, List[float]]] = []
    val_gt_dict = {sid: gt_map[sid] for sid in val_s1_ids}

    for i in val_idx:
        sid = s1_ids[i]
        p1 = s1_parsed[i]
        cands = index.query(p1)
        cand_ids = set()

        for is_s2, row_idx in cands:
            tid = s2_ids[row_idx] if is_s2 == 1 else s3_ids[row_idx]
            cand_ids.add(tid)
            p2 = s2_parsed[row_idx] if is_s2 == 1 else s3_parsed[row_idx]
            feats = compute_parsed_features(p1, p2, is_s2 == 1, corpus.idf, corpus.default_idf)
            val_cand_pairs.append((sid, tid, feats))

        val_candidates_dict[sid] = cand_ids

    # Run blocking evaluation metric
    blocking_metrics = evaluate_blocking_recall(
        val_gt_dict,
        val_candidates_dict,
        total_s2_pool_size=len(s2_ids),
        total_s3_pool_size=len(s3_ids)
    )

    print("-" * 60)
    print(f"BLOCKING RECALL CEILING : {blocking_metrics['blocking_recall_ceiling']*100:.2f}%")
    print(f"  Retained True Matches : {blocking_metrics['retained_in_candidates']:,} / {blocking_metrics['total_true_matches']:,}")
    print(f"  Missed in Blocking    : {blocking_metrics['missed_in_blocking']:,}")
    print(f"  Full Entity Coverage  : {blocking_metrics['entity_full_coverage_rate']*100:.2f}% of non-singletons")
    print(f"  Avg Candidates per S1 : {blocking_metrics['avg_candidates_per_s1']:.2f}")
    if "reduction_ratio" in blocking_metrics:
        print(f"  Reduction Ratio       : {blocking_metrics['reduction_ratio']*100:.5f}%")
    print("-" * 60)

    # Extract Train Pairs: Positives + Hard Negatives from blocking (TASK 3)
    print("\nGenerating training pairs with Hard-Negative Mining...")
    t_pairs = time.time()
    X_train: List[List[float]] = []
    y_train: List[int] = []
    pos_train = 0
    neg_train = 0

    for i in train_idx:
        sid = s1_ids[i]
        p1 = s1_parsed[i]
        true_targets = gt_map.get(sid, set())

        # 1. True Positives
        for tid in true_targets:
            if tid in s2_map:
                feats = compute_parsed_features(p1, s2_map[tid], True, corpus.idf, corpus.default_idf)
                X_train.append(feats)
                y_train.append(1)
                pos_train += 1
            elif tid in s3_map:
                feats = compute_parsed_features(p1, s3_map[tid], False, corpus.idf, corpus.default_idf)
                X_train.append(feats)
                y_train.append(1)
                pos_train += 1

        # 2. Hard Negatives mined from blocking candidates
        cands = index.query(p1)
        for is_s2, row_idx in cands:
            tid = s2_ids[row_idx] if is_s2 == 1 else s3_ids[row_idx]
            if tid not in true_targets:
                p2 = s2_parsed[row_idx] if is_s2 == 1 else s3_parsed[row_idx]
                feats = compute_parsed_features(p1, p2, is_s2 == 1, corpus.idf, corpus.default_idf)
                X_train.append(feats)
                y_train.append(0)
                neg_train += 1

    print(f"Training pairs collected in {time.time() - t_pairs:.2f}s:")
    print(f"  Positive pairs (Matches)      : {pos_train:,}")
    print(f"  Hard-Negative pairs (Non-match): {neg_train:,} (Ratio ~ {neg_train/max(1, pos_train):.1f}:1)")
    print(f"  Total training instances      : {len(X_train):,}")

    # Build validation pairwise evaluation matrix
    X_val: List[List[float]] = []
    y_val: List[int] = []
    for sid, tid, feats in val_cand_pairs:
        label = 1 if tid in val_gt_dict[sid] else 0
        X_val.append(feats)
        y_val.append(label)

    X_train_arr = np.array(X_train, dtype=np.float32)
    y_train_arr = np.array(y_train, dtype=np.int32)
    X_val_arr = np.array(X_val, dtype=np.float32)
    y_val_arr = np.array(y_val, dtype=np.int32)

    return (
        X_train_arr, y_train_arr,
        X_val_arr, y_val_arr,
        val_cand_pairs, val_s1_ids, val_gt_dict,
        val_candidates_dict, blocking_metrics
    )


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = 42
) -> Tuple[lgb.LGBMClassifier, float]:
    """Train LightGBM Classifier with monotonic constraints and binary logloss objective."""
    print("\n" + "=" * 70)
    print("STEP 4: TRAINING LIGHTGBM CLASSIFIER WITH MONOTONIC CONSTRAINTS")
    print("=" * 70)

    model = lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        n_estimators=300,
        learning_rate=0.08,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        monotone_constraints=MONOTONE_CONSTRAINTS,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1
    )

    t0 = time.time()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )
    t_train = time.time() - t0
    print(f"Model trained successfully in {t_train:.2f}s (Best Iteration: {model.best_iteration_})")

    val_probs = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs)
    print(f"Validation Pairwise ROC-AUC: {val_auc:.5f}")

    print("\n--- FEATURE IMPORTANCES ---")
    importances = model.feature_importances_
    sorted_features = sorted(zip(FEATURE_NAMES, importances, MONOTONE_CONSTRAINTS), key=lambda x: x[1], reverse=True)
    print(f"{'Feature Name':28s} | {'Gain/Split':10s} | {'Monotonicity':12s}")
    print("-" * 56)
    for fname, imp, mc in sorted_features:
        mc_str = "+1 (Positive)" if mc == 1 else ("-1 (Negative)" if mc == -1 else "None (0)")
        print(f"{fname:28s} | {imp:10d} | {mc_str:12s}")

    return model, val_auc


def optimize_threshold_macro_f05(
    model: lgb.LGBMClassifier,
    val_cand_pairs: List[Tuple[str, str, List[float]]],
    val_s1_ids: Set[str],
    val_gt_dict: Dict[str, Set[str]],
    val_candidates_dict: Optional[Dict[str, Set[str]]] = None
) -> Tuple[float, float, Dict[str, Any]]:
    """Perform fine-grained grid search for optimal decision threshold theta
    specifically optimizing the competition Macro F0.5 metric with competitive assignment.
    """
    print("\n" + "=" * 70)
    print("STEP 5: MACRO F0.5 THRESHOLD OPTIMIZATION (GRID SEARCH)")
    print("=" * 70)

    t0 = time.time()
    val_X_pairs = np.array([p[2] for p in val_cand_pairs], dtype=np.float32)
    booster = model.booster_
    pair_probs = booster.predict(val_X_pairs)
    print(f"Scored {len(val_cand_pairs):,} validation candidate pairs in {time.time() - t0:.2f}s")

    thresholds = np.linspace(0.40, 0.95, 29)
    best_theta = 0.80
    best_f05 = -1.0
    best_report = {}

    print(f"{'Threshold (theta)':18s} | {'Macro F0.5':12s} | {'Singleton Acc':14s} | {'Non-Sing F0.5':14s}")
    print("-" * 65)

    for theta in thresholds:
        best_assignment: Dict[str, Tuple[float, str]] = {}
        for (sid, cid, _), prob in zip(val_cand_pairs, pair_probs):
            if prob >= theta:
                if cid not in best_assignment or prob > best_assignment[cid][0]:
                    best_assignment[cid] = (prob, sid)

        s1_preds: Dict[str, Set[str]] = {sid: set() for sid in val_s1_ids}
        for cid, (prob, sid) in best_assignment.items():
            s1_preds[sid].add(cid)

        # Detailed evaluation report using the standardized evaluation module
        report = detailed_evaluation_report(
            val_gt_dict,
            s1_preds,
            candidate_pairs=val_candidates_dict
        )
        f05 = report["macro_f05"]
        s_acc = report["singleton_accuracy"]
        ns_f05 = report["non_singleton_macro_f05"]

        print(f"theta = {theta:.2f}            | {f05:12.4f} | {s_acc:14.4f} | {ns_f05:14.4f}")

        if f05 > best_f05:
            best_f05 = f05
            best_theta = float(theta)
            best_report = report

    print("-" * 65)
    print(f"[OPTIMIZATION RESULT] Optimal Threshold: theta* = {best_theta:.2f}")
    print(f"  Overall Macro F0.5 : {best_f05:.4f}")
    print(f"  Singleton Accuracy : {best_report.get('singleton_accuracy', 0.0)*100:.2f}%")
    print(f"  Non-Singleton F0.5 : {best_report.get('non_singleton_macro_f05', 0.0):.4f}")
    print(f"  Global Micro Prec  : {best_report.get('global_micro_precision', 0.0):.4f}")
    print(f"  Global Micro Rec   : {best_report.get('global_micro_recall', 0.0):.4f}")

    return best_theta, best_f05, best_report


def benchmark_cpu_prediction_speed(model: lgb.LGBMClassifier, n_samples: int = 200000) -> float:
    """Verify that model evaluates at > 1,000,000 predictions/sec on CPU."""
    print("\n" + "=" * 70)
    print("STEP 6: BENCHMARKING CPU PREDICTION THROUGHPUT")
    print("=" * 70)

    rng = np.random.RandomState(42)
    X_bench = rng.uniform(0.0, 1.0, size=(n_samples, len(FEATURE_NAMES))).astype(np.float32)

    booster = model.booster_
    _ = booster.predict(X_bench[:1000])

    t0 = time.time()
    _ = booster.predict(X_bench)
    t_elapsed = time.time() - t0

    preds_per_sec = n_samples / t_elapsed
    print(f"Evaluated {n_samples:,} feature rows on CPU in {t_elapsed:.4f}s")
    print(f"Prediction Speed: {preds_per_sec:,.0f} predictions/sec")

    if preds_per_sec >= 1000000:
        print("[SUCCESS] CPU Throughput Benchmark PASSED (> 1,000,000 predictions/sec)!")
    return preds_per_sec


def save_trained_artifacts(
    model: lgb.LGBMClassifier,
    metadata: dict,
    output_dir: str
):
    """Save trained model pickle and metadata JSON."""
    print("\n" + "=" * 70)
    print("STEP 7: SAVING TRAINED ARTIFACTS")
    print("=" * 70)
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "adaptive_matcher.pkl")
    meta_path = os.path.join(output_dir, "adaptive_matcher_meta.json")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved trained model to: {model_path}")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata & threshold to: {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Adaptive Business Entity Resolution Training")
    parser.add_argument("--data-dir", default=None, help="Directory containing train TSV files")
    parser.add_argument("--output-dir", default=os.path.join(SRC_DIR, "models"), help="Directory to save model")
    parser.add_argument("--sample-size", type=int, default=50000, help="Number of S1 entities to train on (0 for all)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 70)
    print("ADAPTIVE BUSINESS ENTITY RESOLUTION — TRAINING & VALIDATION PIPELINE")
    print("=" * 70)

    # 1. Resolve data paths
    data_dir = resolve_data_paths(args.data_dir)

    # 2. Load dataset
    s1_df, gt_map, s2_df, s3_df = load_dataset(data_dir, sample_size=args.sample_size, random_state=args.random_state)

    # 3. Parse text representations
    s1_data, s2_data, s3_data = parse_all_records(s1_df, s2_df, s3_df)

    # 4. Prepare training & validation data
    (
        X_train, y_train,
        X_val, y_val,
        val_cand_pairs, val_s1_ids, val_gt_dict,
        val_candidates_dict, blocking_metrics
    ) = prepare_training_validation_data(
        s1_data, s2_data, s3_data, gt_map,
        test_size=0.20, random_state=args.random_state
    )

    # 5. Train LightGBM model
    model, val_auc = train_classifier(X_train, y_train, X_val, y_val, random_state=args.random_state)

    # 6. Optimize threshold for Macro F0.5
    best_theta, best_f05, best_report = optimize_threshold_macro_f05(
        model, val_cand_pairs, val_s1_ids, val_gt_dict, val_candidates_dict
    )

    # 7. CPU throughput benchmark
    preds_per_sec = benchmark_cpu_prediction_speed(model)

    # 8. Save artifacts
    metadata = {
        "model_name": "LightGBM Adaptive Business Matcher",
        "features": FEATURE_NAMES,
        "optimal_threshold": float(best_theta),
        "validation_macro_f05": float(best_f05),
        "validation_pairwise_auc": float(val_auc),
        "blocking_recall_ceiling": float(blocking_metrics["blocking_recall_ceiling"]),
        "avg_candidates_per_s1": float(blocking_metrics["avg_candidates_per_s1"]),
        "cpu_throughput_preds_per_sec": float(preds_per_sec),
        "sample_size": args.sample_size,
        "random_state": args.random_state
    }
    save_trained_artifacts(model, metadata, args.output_dir)

    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {time.time() - t_start:.2f}s")
    print(f"Final Model Validation Macro F0.5: {best_f05:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
