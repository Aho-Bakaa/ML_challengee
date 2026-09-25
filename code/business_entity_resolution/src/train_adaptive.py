"""
train_adaptive.py
Adaptive Business Entity Resolution Model Training & Validation Pipeline.

Key Features:
- Universal relative similarity features in [0, 1] (via features_adaptive.py).
- Monotonic constraints preserving physical similarity properties.
- Dynamic country-level corpus IDF to eliminate geographic/domain stopwords.
- High-efficiency inverted index blocking with bounded candidates.
- Entity-level stratified train/val split (zero entity leakage).
- Exact competition Macro F0.5 threshold optimization with competitive assignment.
- High-throughput inference (>1,000,000 predictions/sec on CPU).
"""

import os
import sys
import time
import json
import math
import pickle
import argparse
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import lightgbm as lgb

# Ensure src directory is in sys.path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from features_adaptive import (
        FEATURE_NAMES,
        ParsedRecord,
        compute_corpus_idf,
        compute_parsed_features,
        parse_record,
    )
    from blocking_adaptive import AdaptiveInvertedIndex
except ImportError:
    from .features_adaptive import (
        FEATURE_NAMES,
        ParsedRecord,
        compute_corpus_idf,
        compute_parsed_features,
        parse_record,
    )
    from .blocking_adaptive import AdaptiveInvertedIndex

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

    # Standard project paths
    base_dir = os.path.abspath(os.path.join(SRC_DIR, "..", "..", ".."))
    candidates.extend([
        os.path.join(base_dir, "student_resource", "dataset", "train"),
        os.path.join(base_dir, "dataset", "train"),
        r"C:\Users\anmol\OneDrive\Desktop\Work\ML_challenge\student_resource\dataset\train",
        "/kaggle/input/amazon-ml-challenge-2026/train",
        "/kaggle/input/dataset/train",
    ])

    # Check for kaggle input recursively
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


def compute_macro_f05(ground_truth: Dict[str, Set[str]], predictions: Dict[str, Set[str]]) -> float:
    """Exact competition Macro F_0.5 score:
    - Singletons (true empty): 1.0 if empty pred, 0.0 if any false positive.
    - Non-singletons: (5 * TP) / (len_pred + 4 * len_true).
    """
    total_score = 0.0
    n = len(ground_truth)
    if n == 0:
        return 0.0

    for sid, y_true in ground_truth.items():
        y_pred = predictions.get(sid, set())
        len_true = len(y_true)
        len_pred = len(y_pred)

        if len_true == 0:
            if len_pred == 0:
                total_score += 1.0
            continue

        if len_pred == 0:
            continue

        tp = len(y_true & y_pred)
        if tp == 0:
            continue

        score = (5.0 * tp) / (float(len_pred) + 4.0 * float(len_true))
        total_score += score

    return total_score / float(n)


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

    s1_set = set(s1["entity_id"])
    print(f"Reading Ground Truth from: {gt_path}...")
    gt = pl.read_csv(gt_path, separator="\t")
    gt_sample = gt.filter(pl.col("source1_entity_id").is_in(s1_set))

    gt_map: Dict[str, Set[str]] = {}
    needed_s2: Set[str] = set()
    needed_s3: Set[str] = set()
    singletons = 0
    total_pos_links = 0

    for r in gt_sample.iter_rows(named=True):
        sid = r["source1_entity_id"]
        m = r["matched_entity_ids"]
        if m and str(m).strip() != "" and str(m) != "None" and str(m) != "nan":
            targets = set(x.strip() for x in str(m).split(",") if x.strip())
            gt_map[sid] = targets
            total_pos_links += len(targets)
            for tid in targets:
                if tid.startswith("S2-"):
                    needed_s2.add(tid)
                elif tid.startswith("S3-"):
                    needed_s3.add(tid)
        else:
            gt_map[sid] = set()
            singletons += 1

    # Ensure all S1 entities have an entry in gt_map
    for sid in s1["entity_id"]:
        if sid not in gt_map:
            gt_map[sid] = set()
            singletons += 1

    print(f"Ground Truth Statistics:")
    print(f"  Total S1 entities   : {len(s1):,}")
    print(f"  Matched S1 entities : {len(s1) - singletons:,}")
    print(f"  Singletons (no match): {singletons:,} ({singletons / len(s1) * 100:.2f}%)")
    print(f"  Total Positive Links: {total_pos_links:,}")
    print(f"  Unique S2 needed    : {len(needed_s2):,}")
    print(f"  Unique S3 needed    : {len(needed_s3):,}")

    # Scan and collect needed S2 and S3 records
    print(f"Scanning target pools for {len(needed_s2):,} S2 and {len(needed_s3):,} S3 records...")
    t_targets = time.time()
    s2_df = pl.scan_csv(s2_path, separator="\t").filter(pl.col("entity_id").is_in(needed_s2)).collect()
    s3_df = pl.scan_csv(s3_path, separator="\t").filter(pl.col("entity_id").is_in(needed_s3)).collect()
    print(f"Retrieved targets in {time.time() - t_targets:.2f}s (S2={len(s2_df):,}, S3={len(s3_df):,})")
    print(f"Data loading completed in {time.time() - t0:.2f}s")

    return s1, gt_map, s2_df, s3_df


def prepare_features_and_splits(
    s1: pl.DataFrame,
    gt_map: Dict[str, Set[str]],
    s2_df: pl.DataFrame,
    s3_df: pl.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
):
    """Pre-parse records, compute country-level dynamic IDF, build inverted index,
    and generate entity-stratified train/val feature datasets with hard negatives.
    """
    print("\n" + "=" * 70)
    print("STEP 2: PRE-PARSING RECORDS & BUILDING BLOCKING INDEX")
    print("=" * 70)
    t0 = time.time()

    # Pre-parse S1
    print(f"Pre-parsing {len(s1):,} S1 records...")
    s1_names = s1["business_name"].to_list()
    s1_addrs = s1["business_address"].to_list()
    s1_ids = s1["entity_id"].to_list()
    s1_parsed = [parse_record(n, a) for n, a in zip(s1_names, s1_addrs)]

    # Pre-parse S2 and S3
    print(f"Pre-parsing {len(s2_df):,} S2 and {len(s3_df):,} S3 records...")
    s2_names = s2_df["business_name"].to_list()
    s2_addrs = s2_df["business_address"].to_list()
    s2_ids = s2_df["entity_id"].to_list()
    s2_parsed = [parse_record(n, a) for n, a in zip(s2_names, s2_addrs)]

    s3_names = s3_df["business_name"].to_list()
    s3_addrs = s3_df["business_address"].to_list()
    s3_ids = s3_df["entity_id"].to_list()
    s3_parsed = [parse_record(n, a) for n, a in zip(s3_names, s3_addrs)]

    print(f"All records pre-parsed in {time.time() - t0:.2f}s")

    # Dynamic Corpus IDF from target pool
    print("\nComputing dynamic corpus IDF on target pool (S2 + S3)...")
    target_parsed = s2_parsed + s3_parsed
    corpus = compute_corpus_idf(target_parsed)
    print(f"Corpus Vocabulary: {len(corpus.df):,} unique tokens")
    print(f"Corpus Stopwords : {len(corpus.stopwords):,} tokens (top frequent/domain words)")

    # Build Adaptive Inverted Index
    print("Building adaptive inverted index...")
    t_idx = time.time()
    index = AdaptiveInvertedIndex(max_posting=30, stopwords=corpus.stopwords, df_dict=corpus.df)
    index.add_records(s2_parsed, is_s2_flag=1)
    index.add_records(s3_parsed, is_s2_flag=0)
    print(f"Inverted index built in {time.time() - t_idx:.2f}s with {len(index.index):,} keys")

    # Build lookup maps
    s2_map = {eid: p for eid, p in zip(s2_ids, s2_parsed)}
    s3_map = {eid: p for eid, p in zip(s3_ids, s3_parsed)}

    # Stratified Train/Val split at the S1 entity level (Zero Entity Leakage)
    print("\n" + "=" * 70)
    print("STEP 3: STRATIFIED TRAIN/VAL SPLIT & HARD NEGATIVE PAIR EXTRACTION")
    print("=" * 70)
    s1_indices = np.arange(len(s1_ids))
    # Stratify by whether the S1 entity is a singleton or matched entity
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

    # Extract Train Pairs: Positives + Hard Negatives from blocking
    print("Generating balanced training pairs...")
    t_pairs = time.time()
    X_train: List[List[float]] = []
    y_train: List[int] = []

    pos_train = 0
    neg_train = 0

    for i in train_idx:
        sid = s1_ids[i]
        p1 = s1_parsed[i]
        true_targets = gt_map.get(sid, set())

        # 1. True Positive Pairs
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

        # 2. Hard Negatives from Inverted Index Blocking
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
    print(f"  Positive pairs : {pos_train:,}")
    print(f"  Negative pairs : {neg_train:,}")
    print(f"  Total train pairs: {len(X_train):,}")

    # Extract Validation Pairs & Candidate Pool for End-to-End Metric Evaluation
    print("\nPreparing validation pairs and candidate pool...")
    X_val: List[List[float]] = []
    y_val: List[int] = []
    val_cand_pairs: List[Tuple[str, str, List[float]]] = []  # (s1_id, cand_id, features)

    pos_val = 0
    neg_val = 0

    for i in val_idx:
        sid = s1_ids[i]
        p1 = s1_parsed[i]
        true_targets = gt_map.get(sid, set())

        # True positives for ROC-AUC & pair metrics
        for tid in true_targets:
            if tid in s2_map:
                feats = compute_parsed_features(p1, s2_map[tid], True, corpus.idf, corpus.default_idf)
                X_val.append(feats)
                y_val.append(1)
                pos_val += 1
            elif tid in s3_map:
                feats = compute_parsed_features(p1, s3_map[tid], False, corpus.idf, corpus.default_idf)
                X_val.append(feats)
                y_val.append(1)
                pos_val += 1

        # Blocking candidates for end-to-end Macro F0.5 evaluation
        cands = index.query(p1)
        for is_s2, row_idx in cands:
            tid = s2_ids[row_idx] if is_s2 == 1 else s3_ids[row_idx]
            p2 = s2_parsed[row_idx] if is_s2 == 1 else s3_parsed[row_idx]
            feats = compute_parsed_features(p1, p2, is_s2 == 1, corpus.idf, corpus.default_idf)
            val_cand_pairs.append((sid, tid, feats))

            if tid not in true_targets:
                X_val.append(feats)
                y_val.append(0)
                neg_val += 1

    print(f"Validation pairs collected:")
    print(f"  Positive pairs : {pos_val:,}")
    print(f"  Negative pairs : {neg_val:,}")
    print(f"  Total val pairs: {len(X_val):,}")
    print(f"  Val candidates to score: {len(val_cand_pairs):,}")

    X_train_arr = np.array(X_train, dtype=np.float32)
    y_train_arr = np.array(y_train, dtype=np.int32)
    X_val_arr = np.array(X_val, dtype=np.float32)
    y_val_arr = np.array(y_val, dtype=np.int32)

    val_gt_dict = {sid: gt_map[sid] for sid in val_s1_ids}

    return (
        X_train_arr, y_train_arr,
        X_val_arr, y_val_arr,
        val_cand_pairs, val_s1_ids, val_gt_dict
    )


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = 42
) -> lgb.LGBMClassifier:
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

    # Validation Pairwise ROC-AUC
    val_probs = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs)
    print(f"Validation Pairwise ROC-AUC: {val_auc:.5f}")

    # Feature Importance Table
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
    val_gt_dict: Dict[str, Set[str]]
) -> Tuple[float, float, float, float]:
    """Perform fine-grained grid search for optimal decision threshold theta
    specifically optimizing the competition Macro F0.5 metric with competitive assignment.
    """
    print("\n" + "=" * 70)
    print("STEP 5: MACRO F0.5 THRESHOLD OPTIMIZATION (GRID SEARCH)")
    print("=" * 70)

    # Batch score all validation candidate pairs
    t0 = time.time()
    val_X_pairs = np.array([p[2] for p in val_cand_pairs], dtype=np.float32)
    booster = model.booster_
    pair_probs = booster.predict(val_X_pairs)
    print(f"Scored {len(val_cand_pairs):,} validation candidate pairs in {time.time() - t0:.2f}s")

    # Grid of candidate thresholds
    thresholds = np.linspace(0.40, 0.95, 29)
    best_theta = 0.80
    best_f05 = -1.0
    best_prec = 0.0
    best_rec = 0.0

    print(f"{'Threshold (theta)':18s} | {'Macro F0.5':12s} | {'Precision':12s} | {'Recall':12s}")
    print("-" * 60)

    for theta in thresholds:
        # Competitive 1-to-1 assignment: each candidate target matches the S1 with highest score >= theta
        best_assignment: Dict[str, Tuple[float, str]] = {}
        for (sid, cid, _), prob in zip(val_cand_pairs, pair_probs):
            if prob >= theta:
                if cid not in best_assignment or prob > best_assignment[cid][0]:
                    best_assignment[cid] = (prob, sid)

        # Collect S1 -> matched candidate set
        s1_preds: Dict[str, Set[str]] = {sid: set() for sid in val_s1_ids}
        for cid, (prob, sid) in best_assignment.items():
            s1_preds[sid].add(cid)

        # Macro F0.5 calculation
        f05 = compute_macro_f05(val_gt_dict, s1_preds)

        # Pair-level Precision and Recall across all validation entities
        total_tp = 0
        total_pred = 0
        total_true = 0
        for sid, y_true in val_gt_dict.items():
            y_pred = s1_preds[sid]
            total_tp += len(y_true & y_pred)
            total_pred += len(y_pred)
            total_true += len(y_true)

        prec = total_tp / total_pred if total_pred > 0 else 1.0
        rec = total_tp / total_true if total_true > 0 else 0.0

        print(f"theta = {theta:.2f}            | {f05:12.4f} | {prec:12.4f} | {rec:12.4f}")

        if f05 > best_f05:
            best_f05 = f05
            best_theta = float(theta)
            best_prec = prec
            best_rec = rec

    print("-" * 60)
    print(f"[OPTIMIZATION RESULT] Optimal Threshold: theta* = {best_theta:.2f}")
    print(f"  Macro F0.5 Score : {best_f05:.4f}")
    print(f"  Precision        : {best_prec:.4f}")
    print(f"  Recall           : {best_rec:.4f}")

    return best_theta, best_f05, best_prec, best_rec


def benchmark_cpu_prediction_speed(model: lgb.LGBMClassifier, n_samples: int = 200000) -> float:
    """Verify that model evaluates at > 1,000,000 predictions/sec on CPU."""
    print("\n" + "=" * 70)
    print("STEP 6: BENCHMARKING CPU PREDICTION THROUGHPUT")
    print("=" * 70)

    # Generate realistic feature matrix in [0, 1]
    rng = np.random.RandomState(42)
    X_bench = rng.uniform(0.0, 1.0, size=(n_samples, len(FEATURE_NAMES))).astype(np.float32)

    # Warmup
    booster = model.booster_
    _ = booster.predict(X_bench[:1000])

    # Timed prediction run
    t0 = time.time()
    _ = booster.predict(X_bench)
    t_elapsed = time.time() - t0

    preds_per_sec = n_samples / t_elapsed
    print(f"Evaluated {n_samples:,} feature rows on CPU in {t_elapsed:.4f}s")
    print(f"Prediction Speed: {preds_per_sec:,.0f} predictions/sec")

    if preds_per_sec >= 1000000:
        print("[SUCCESS] CPU Throughput Benchmark PASSED (> 1,000,000 predictions/sec)!")
    else:
        print(f"[NOTICE] CPU Throughput: {preds_per_sec:,.0f} predictions/sec")

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

    # Also save metadata directly to root/models if running locally
    alt_meta = os.path.join(SRC_DIR, "models", "adaptive_matcher_meta.json")
    if os.path.abspath(alt_meta) != os.path.abspath(meta_path):
        os.makedirs(os.path.dirname(alt_meta), exist_ok=True)
        with open(alt_meta, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)


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
    print(f"Sample Size : {args.sample_size if args.sample_size > 0 else 'Full'}")
    print(f"Output Dir  : {args.output_dir}")
    print(f"Random State: {args.random_state}")
    print("=" * 70)

    # 1. Resolve data paths
    data_dir = resolve_data_paths(args.data_dir)

    # 2. Load dataset and ground truth
    s1, gt_map, s2_df, s3_df = load_dataset(data_dir, sample_size=args.sample_size, random_state=args.random_state)

    # 3. Pre-parse, build index, generate pairs with entity-stratified split
    (
        X_train, y_train,
        X_val, y_val,
        val_cand_pairs, val_s1_ids, val_gt_dict
    ) = prepare_features_and_splits(
        s1=s1,
        gt_map=gt_map,
        s2_df=s2_df,
        s3_df=s3_df,
        test_size=0.2,
        random_state=args.random_state
    )

    # 4. Train LightGBM classifier with monotonic constraints
    model, val_auc = train_classifier(X_train, y_train, X_val, y_val, random_state=args.random_state)

    # 5. Optimize Macro F0.5 decision threshold
    best_theta, best_f05, best_prec, best_rec = optimize_threshold_macro_f05(
        model, val_cand_pairs, val_s1_ids, val_gt_dict
    )

    # 6. Benchmark prediction speed on CPU
    speed_preds_per_sec = benchmark_cpu_prediction_speed(model, n_samples=200000)

    # 7. Package and save artifacts
    metadata = {
        "model_type": "LightGBM Classifier",
        "objective": "binary_logloss",
        "optimal_threshold": round(best_theta, 4),
        "validation_macro_f05": round(best_f05, 4),
        "validation_roc_auc": round(val_auc, 5),
        "validation_precision": round(best_prec, 4),
        "validation_recall": round(best_rec, 4),
        "features": FEATURE_NAMES,
        "monotone_constraints": MONOTONE_CONSTRAINTS,
        "sample_size": args.sample_size,
        "total_train_pairs": len(X_train),
        "total_val_pairs": len(X_val),
        "prediction_speed_cpu_rows_per_sec": int(speed_preds_per_sec),
        "trained_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    save_trained_artifacts(model, metadata, args.output_dir)

    print("\n" + "=" * 70)
    print("TRAINING & VALIDATION COMPLETED SUCCESSFULLY!")
    print(f"Total Pipeline Runtime: {time.time() - t_start:.2f}s")
    print(f"Optimal Threshold     : {best_theta:.2f}")
    print(f"Macro F0.5 at Optimum : {best_f05:.4f}")
    print(f"Validation ROC-AUC    : {val_auc:.5f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
