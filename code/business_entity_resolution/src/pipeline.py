"""
pipeline_adaptive.py
Amazon ML Challenge 2026 — Adaptive Country-Agnostic Business Entity Resolution Pipeline
Zero hardcoded rules: universal NFKD text normalization, multi-representation signatures,
dynamic country-level corpus IDF, bounded candidate blocking, and competitive 1-to-1 assignment.
"""

import os
import sys
import argparse
import time
import pickle
import gc
import json
import polars as pl
import numpy as np
from typing import Optional, List, Dict, Set, Tuple

# Add local directory to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from features import (
        parse_record,
        compute_corpus_idf,
        compute_parsed_features,
        FEATURE_NAMES
    )
except ImportError:
    from features_adaptive import (
        parse_record,
        compute_corpus_idf,
        compute_parsed_features,
        FEATURE_NAMES
    )

try:
    from blocking import AdaptiveInvertedIndex
except ImportError:
    from blocking_adaptive import AdaptiveInvertedIndex

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "adaptive_matcher.pkl")

def run_adaptive_pipeline(
    test_dir: str,
    output_dir: str,
    model_path: str = DEFAULT_MODEL_PATH,
    threshold: Optional[float] = None,
    batch_size: int = 10000
):
    t_start = time.time()
    os.makedirs(output_dir, exist_ok=True)
    
    matching_file = os.path.join(output_dir, "matching_results.tsv")
    candidate_file = os.path.join(output_dir, "candidate_pairs.tsv")
    
    # Auto-resolve threshold from metadata if not explicitly given
    if threshold is None:
        meta_path = os.path.splitext(model_path)[0] + "_meta.json"
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    threshold = float(meta.get("optimal_threshold", 0.40))
            except Exception:
                threshold = 0.40
        else:
            threshold = 0.40
            
    print("=" * 70, flush=True)
    print("AMAZON ML CHALLENGE 2026 — ADAPTIVE ENTITY RESOLUTION PIPELINE", flush=True)
    print("=" * 70, flush=True)
    print(f"Test Directory     : {test_dir}", flush=True)
    print(f"Output Directory   : {output_dir}", flush=True)
    print(f"Model Path         : {model_path}", flush=True)
    print(f"Decision Threshold : {threshold:.2f}", flush=True)
    print("=" * 70, flush=True)
    
    # 1. Load trained model
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("Loaded trained model successfully.", flush=True)
    
    # 2. Inspect Test S1 entities and countries
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")
    
    print(f"Reading test reference entities from {s1_path}...", flush=True)
    s1_full = pl.read_csv(s1_path, separator="\t")
    total_s1 = len(s1_full)
    countries = s1_full["country"].unique().to_list()
    print(f"Total S1 entities: {total_s1} across countries: {countries}", flush=True)
    
    # 3. Open output files and write standard headers
    f_match = open(matching_file, "w", encoding="utf-8", newline="")
    f_cand = open(candidate_file, "w", encoding="utf-8", newline="")
    
    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
    
    total_written = 0
    total_matches_found = 0
    total_singletons = 0
    
    # 4. Stream country by country
    for country in countries:
        t_c = time.time()
        print(f"\n>>> Processing Country: {country} <<<", flush=True)
        
        # S1 for country
        c_s1 = s1_full.filter(pl.col("country") == country)
        n_s1 = len(c_s1)
        s1_ids = c_s1["entity_id"].to_list()
        s1_names = c_s1["business_name"].to_list()
        s1_addrs = c_s1["business_address"].to_list()
        print(f"  Source 1 records: {n_s1}", flush=True)
        
        t_p_s1 = time.time()
        s1_parsed = [parse_record(n, a) for n, a in zip(s1_names, s1_addrs)]
        print(f"  Pre-parsed S1 in {time.time()-t_p_s1:.2f}s", flush=True)
        
        # Lazy scan and filter S2 and S3 for this country
        print(f"  Loading S2 and S3 for {country}...", flush=True)
        s2_df = pl.scan_csv(s2_path, separator="\t").filter(pl.col("country") == country).collect()
        s3_df = pl.scan_csv(s3_path, separator="\t").filter(pl.col("country") == country).collect()
        
        s2_ids = s2_df["entity_id"].to_list()
        s2_names = s2_df["business_name"].to_list()
        s2_addrs = s2_df["business_address"].to_list()
        
        s3_ids = s3_df["entity_id"].to_list()
        s3_names = s3_df["business_name"].to_list()
        s3_addrs = s3_df["business_address"].to_list()
        print(f"  Target pools: S2={len(s2_ids)}, S3={len(s3_ids)}", flush=True)
        
        t_p_targets = time.time()
        s2_parsed = [parse_record(n, a) for n, a in zip(s2_names, s2_addrs)]
        s3_parsed = [parse_record(n, a) for n, a in zip(s3_names, s3_addrs)]
        print(f"  Pre-parsed S2 & S3 ({len(s2_parsed)+len(s3_parsed)} records) in {time.time()-t_p_targets:.2f}s", flush=True)
        
        # Dynamic Country-Level Corpus IDF
        t_idf = time.time()
        target_pool = s2_parsed + s3_parsed
        corpus_idf = compute_corpus_idf(target_pool)
        print(f"  Dynamic Corpus IDF built in {time.time()-t_idf:.2f}s: {len(corpus_idf.df)} unique tokens, {len(corpus_idf.stopwords)} stopwords identified", flush=True)
        
        # Build bounded inverted index
        t_idx = time.time()
        index = AdaptiveInvertedIndex(
            max_posting=30,
            max_candidates=15,
            stopwords=corpus_idf.stopwords,
            df_dict=corpus_idf.df
        )
        index.add_records(s2_parsed, 1)
        index.add_records(s3_parsed, 0)
        print(f"  Inverted index built in {time.time()-t_idx:.2f}s with {len(index.index)} keys", flush=True)
        
        # Batch candidate generation and feature extraction
        t_eval = time.time()
        cand_dict = {}
        pair_scores = [] # (s1_id, cand_id, prob)
        
        for b_start in range(0, n_s1, batch_size):
            b_end = min(b_start + batch_size, n_s1)
            batch_X = []
            batch_meta = []
            
            for i in range(b_start, b_end):
                sid = s1_ids[i]
                p1 = s1_parsed[i]
                
                cands_ranked = index.query(p1)
                
                resolved_ids = []
                for is_s2, row_idx in cands_ranked:
                    if is_s2 == 1:
                        cid = s2_ids[row_idx]
                        p2 = s2_parsed[row_idx]
                    else:
                        cid = s3_ids[row_idx]
                        p2 = s3_parsed[row_idx]
                        
                    resolved_ids.append(cid)
                    feats = compute_parsed_features(p1, p2, is_s2 == 1, corpus_idf=corpus_idf)
                    batch_X.append(feats)
                    batch_meta.append((sid, cid))
                    
                cand_dict[sid] = resolved_ids
                
            if batch_X:
                X_arr = np.array(batch_X, dtype=np.float32)
                probs = model.predict_proba(X_arr)[:, 1]
                for (sid, cid), prob in zip(batch_meta, probs):
                    if prob >= threshold:
                        pair_scores.append((sid, cid, float(prob)))
                        
            if (b_end % 50000 == 0) or b_end == n_s1:
                elapsed = time.time() - t_eval
                rate = b_end / max(0.1, elapsed)
                print(f"    Scored {b_end}/{n_s1} entities in {elapsed:.1f}s ({rate:,.0f} entities/sec)...", flush=True)
                
        # 5. Global Competitive Assignment for this country
        print("  Applying competitive 1-to-1 target assignment...", flush=True)
        best_assignment = {}
        for sid, cid, prob in pair_scores:
            if cid not in best_assignment or prob > best_assignment[cid][0]:
                best_assignment[cid] = (prob, sid)
                
        # Map S1 -> matched candidates
        s1_to_matches = {sid: [] for sid in s1_ids}
        for cid, (prob, sid) in best_assignment.items():
            if sid in s1_to_matches:
                s1_to_matches[sid].append(cid)
                
        # Write results for this country to disk
        print("  Writing results for this country...", flush=True)
        for sid in s1_ids:
            matches = s1_to_matches.get(sid, [])
            candidates = cand_dict.get(sid, [])
            
            # Ensure final matches are strictly a subset of candidates
            cand_set = set(candidates)
            valid_matches = [m for m in matches if m in cand_set]
            
            match_str = ",".join(valid_matches)
            cand_str = ",".join(candidates)
            
            f_match.write(f"{sid}\t{match_str}\n")
            f_cand.write(f"{sid}\t{cand_str}\n")
            
            total_written += 1
            if valid_matches:
                total_matches_found += len(valid_matches)
            else:
                total_singletons += 1
                
        f_match.flush()
        f_cand.flush()
        print(f"  Country {country} complete in {time.time()-t_c:.2f}s.", flush=True)
        
        # Free memory per country
        del s2_df, s3_df, s2_ids, s3_ids, s1_parsed, s2_parsed, s3_parsed, target_pool, corpus_idf, index, cand_dict, pair_scores, best_assignment, s1_to_matches
        gc.collect()
        
    f_match.close()
    f_cand.close()
    
    print("\n" + "=" * 70, flush=True)
    print("PIPELINE COMPLETED SUCCESSFULLY!", flush=True)
    print("=" * 70, flush=True)
    print(f"Total S1 entities written   : {total_written} (expected {total_s1})", flush=True)
    print(f"Total matched links found   : {total_matches_found}", flush=True)
    print(f"Total singletons (no match) : {total_singletons} ({total_singletons/total_written*100:.2f}%)", flush=True)
    print(f"Total Elapsed Time          : {time.time()-t_start:.2f}s", flush=True)
    print("=" * 70, flush=True)

def main():
    parser = argparse.ArgumentParser(description="Run Adaptive Entity Resolution Pipeline")
    parser.add_argument("--test-dir", default="student_resource/dataset/test", help="Path to dataset/test directory")
    parser.add_argument("--output-dir", default="output", help="Path to output directory")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to trained model")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold for matching (default: optimal from model metadata)")
    parser.add_argument("--batch-size", type=int, default=10000, help="Batch size for query & feature evaluation")
    args = parser.parse_args()
    
    run_adaptive_pipeline(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        threshold=args.threshold,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
