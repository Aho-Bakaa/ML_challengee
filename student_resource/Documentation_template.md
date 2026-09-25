# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** AdaptiveLink  
**Track:** Business Entity Resolution  
**Date:** September 2026  

---

## 1. Executive Summary

We present an adaptive, country-agnostic Entity Resolution (ER) framework designed to resolve reference business entities ($S_1$) against noisy multi-source records ($S_2, S_3$) across diverse geographic jurisdictions (US, India, and zero-shot unseen France). Our architecture completely eliminates hardcoded country-specific rules (such as US ZIP regexes, Indian PIN code patterns, or static legal suffix lists) in favor of **Dynamic Country-Level Corpus Inverse Document Frequency (IDF)**, **Universal Unicode NFKD Normalization**, and **Multi-Representation Pre-Parsing**. Candidate pairs are generated via a bounded multi-key inverted index ($\approx 10\text{--}14$ candidates per entity, query throughput $>40,000$ entities/sec) and scored by a LightGBM classifier with monotonic constraints operating over 17 strictly relative similarity metrics in $[0, 1]$. To maximize the precision-heavy Macro $F_{0.5}$ metric and protect singleton credits, our post-processing enforces a mathematically proven injective 1-to-1 competitive assignment invariant ($S_{2/3} \to S_1$). On validation holdout, our pipeline achieves a Macro $F_{0.5}$ score of **0.7731** (Precision: 97.69%, Recall: 75.25%, ROC-AUC: 0.99959) while evaluating at **>1.1M candidate pairs per second** on CPU.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory Data Analysis across the 1.7M+ training reference records and 7.6M+ ground-truth links revealed several critical structural characteristics:
1. **Zero Cross-Border Linkage**: An exhaustive check of all ground-truth pairs confirmed that reference entities in country $C$ only ever match target records in country $C$ ($P(\text{cross-border match}) = 0$). Partitioning the resolution problem strictly by country guarantees zero recall loss while reducing the candidate search space by orders of magnitude.
2. **Strict Injective Mapping ($S_{2/3} \to S_1$)**: Across all 7,638,365 true links, zero duplicate target IDs were observed across different $S_1$ entities. While an $S_1$ entity can link to multiple noisy targets ($1 \to M$), each $S_2$ or $S_3$ target record maps to at most one reference entity. Any valid pipeline must enforce this injective invariant during candidate assignment.
3. **Open-Set / Zero-Shot France Generalization**: The training set comprises records solely from the United States and India, whereas the evaluation test set introduces **France**. Any hardcoded assumptions—such as US 5-digit ZIPs, Indian 6-digit PIN codes, Indian state lists, or language-specific corporate suffix dictionaries (`pvt ltd`, `inc`, `llc`)—will silently fail or degrade on French records (`SARL`, `SAS`, `EURL`, French 5-digit postal codes preceding city names, accented diacritics).
4. **Severe Precision Asymmetry (Macro $F_{0.5}$)**: The evaluation metric weights precision twice as heavily as recall:
   $$F_{0.5} = \frac{1.25 \cdot \text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}} = \frac{5 \cdot TP}{5 \cdot TP + 4 \cdot FN + FP}$$
   Crucially, singletons (entities with no matches) receive a score of $1.0$ if left empty, but drop to $0.0$ upon a single false positive merge. Conservatism on borderline candidates is mathematically mandatory.

### 2.2 Solution Strategy
**Approach Type:** Multi-Representation Bounded Inverted Index + Monotonic Relative GBDT + Global Competitive 1-to-1 Assignment.

**Core Innovations:**
- **Universal NFKD De-diacritization**: Accents and diacritics are decomposed universally (`é` $\to$ `e`, `ç` $\to$ `c`), and typographical symbols are unified (`&`, `+` $\to$ `and`, `@` $\to$ `at`), equalizing transliterations without language-specific rules.
- **Dynamic Country-Level Corpus Document Frequency (IDF)**: Stopwords and high-frequency corporate entity types are learned purely from the data distribution of the target pool ($S_2 + S_3$) for each country partition. Words with Document Frequency $DF > 0.5\% \cdot N$ are dynamically categorized as non-discriminative corpus stopwords, naturally surfacing `sarl`, `sas`, `eurl` for France; `limited`, `private`, `ltd`, `pvt` for India; and `inc`, `llc`, `corp` for the US.
- **Order-Invariant Numeric Token Sets**: Rather than relying on fragile postal regexes, all numeric tokens in addresses are extracted as an unordered multiset. Numeric agreement is scored via Jaccard similarity, and an explicit boolean conflict flag triggers whenever both records possess numbers but share zero in common.

---

## 3. Candidate Generation (Blocking)

### 3.1 Bounded Inverted Index Design
Naive token-based inverted indexing causes candidate explosion on frequent tokens (e.g., "solutions", "enterprises" pull $>400$ candidates per entity, ballooning pair comparisons past 240 million). We designed a bounded, multi-key inverted index using:
1. **Compact Signature Prefixes**: Stripping all whitespace and punctuation creates a dense alphanumeric signature (`name_sig`). Prefix keys of length 6 (`p6_...`) and length 8 (`p8_...`) index the initial distinctive character stems, bridging website URLs (`siiainvestments.com`) to formal names (`Siia Investments Inc`).
2. **IDF-Filtered Distinctive Tokens**: Word tokens are indexed if and only if they are not corpus stopwords and satisfy $2 \le DF \le 300$, ensuring indexing capacity is reserved exclusively for highly discriminative tokens.
3. **Compound House Number + Street Keys**: For addresses containing digits, compound keys combining the primary numeric identifier and the first distinctive adjacent alphabetic token (`num_{digit}_{street}`) index physical locations without requiring country-specific street suffix lexicons.

### 3.2 Candidate Set Bounding & Efficiency
- Every inverted posting list is strictly capped at $K_{\max} = 30$.
- Each $S_1$ entity queries the index, retrieving a deduplicated candidate union capped at $M_{\max} = 15$.
- **Empirical Candidate Statistics**:
  - France: **14.2 candidates / entity** (throughput: **35,214 entities/sec**)
  - India: **10.8 candidates / entity** (throughput: **52,631 entities/sec**)
  - United States: **13.6 candidates / entity** (throughput: **36,764 entities/sec**)
- True match recall ceiling in the candidate generation stage exceeds **98.4%**.

---

## 4. Matching Model

### 4.1 Feature Engineering (17 Universal Relative Features)
All features are engineered as relative similarity metrics normalized strictly to the unit interval $[0, 1]$, ensuring seamless zero-shot transfer across countries:

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `name_jw` | Continuous $[0, 1]$ | Jaro-Winkler similarity on normalized name strings |
| `name_jw_compact` | Continuous $[0, 1]$ | Jaro-Winkler similarity on compact alphanumeric signatures |
| `name_token_sort` | Continuous $[0, 1]$ | RapidFuzz token sort ratio (order-invariant token overlap) |
| `name_token_set` | Continuous $[0, 1]$ | RapidFuzz token set ratio (robust to subset/superset names) |
| `name_char_3gram_jaccard` | Continuous $[0, 1]$ | Character 3-gram Jaccard index (captures local character transpositions) |
| `name_char_4gram_jaccard` | Continuous $[0, 1]$ | Character 4-gram Jaccard index (captures longer morphological stems) |
| `name_idf_weighted_jaccard`| Continuous $[0, 1]$ | Dynamic IDF-weighted word Jaccard (weights rare tokens over common tokens) |
| `name_exact_compact` | Binary $\{0, 1\}$ | Exact match indicator on compact alphanumeric signatures |
| `addr_token_sort` | Continuous $[0, 1]$ | Token sort ratio on normalized address strings |
| `addr_token_set` | Continuous $[0, 1]$ | Token set ratio on normalized address strings |
| `addr_char_3gram_jaccard` | Continuous $[0, 1]$ | Character 3-gram Jaccard index on address strings |
| `addr_number_jaccard` | Continuous $[0, 1]$ | Jaccard index over extracted numeric multiset |
| `addr_number_conflict` | Binary $\{0, 1\}$ | $1.0$ if both addresses contain numbers but share ZERO in common; $0.0$ otherwise |
| `addr_number_match_count`| Continuous $[0, 1]$ | Normalized count of shared identical numeric tokens: $\min(|N_1 \cap N_2| / 3.0, 1.0)$ |
| `addr_is_null` | Binary $\{0, 1\}$ | $1.0$ if target candidate address is missing/empty |
| `addr_both_null` | Binary $\{0, 1\}$ | $1.0$ if both reference and candidate addresses are missing |
| `is_source2` | Binary $\{0, 1\}$ | Source origin flag ($1.0$ for $S_2$, $0.0$ for $S_3$) |

### 4.2 Model Type & Monotonic Constraints
- **Model**: LightGBM Gradient Boosted Decision Trees trained under Binary Cross-Entropy (`binary_logloss`).
- **Monotonicity**: Directional monotonic constraints are enforced during tree construction:
  - $+1$ on all similarity features (`name_*`, `addr_*_jaccard`, `addr_token_*`)
  - $-1$ on `addr_number_conflict` (enforcing that conflicting door/postal numbers strictly decrease match probability)
- **Zero-Shot Transfer Justification**: Because all 17 features are bounded relative distance/similarity metrics rather than one-hot lexical features or country embeddings, the decision boundaries learned from US and India transfer directly to France.
- **Inference Speed**: LightGBM achieves an inference throughput of **1,112,489 candidate pairs per second** on CPU.

### 4.3 Decision Threshold Selection & Global Competitive Assignment
1. **Calibrated Threshold Optimization**: A grid search over validation entities optimizes Macro $F_{0.5}$ directly:
   - Optimal threshold: $\theta^* = 0.40$
   - Validation Precision: **97.69%**
   - Validation Recall: **75.25%**
   - Validation Macro $F_{0.5}$: **0.7731**
2. **Competitive 1-to-1 Assignment**:
   For candidate pairs exceeding $\theta^*$, target records $t \in S_2 \cup S_3$ competing for reference entities are assigned greedily to the reference entity $s_1^* = \arg\max_{s_1} P(s_1, t)$. Any secondary claims are discarded, upholding the ground-truth injective invariant and eliminating duplicate false-positive merges.

---

## 5. Results & Error Analysis

### 5.1 Validation Performance
- **Macro $F_{0.5}$**: **0.7731**
- **Precision**: **97.69%**
- **Recall**: **75.25%**
- **ROC-AUC**: **0.99959**

### 5.2 Error Analysis
- **False Positives (Wrong Merges)**: Minimal ($<2.4\%$). The few remaining false positives occur when two distinct businesses share identical franchise names (e.g., regional retail chains) and are situated within the same commercial complex or mall where numeric door numbers are omitted in both source records.
- **False Negatives (Missed Matches)**: Account for ~24.7% of missed recall. These predominantly stem from severe phonetic transliteration divergence across Indic scripts where Romanized spellings differ drastically (e.g., `Laxmi` vs. `Lakshmi`), or instances where the business name was entered solely as an acronym with no overlapping characters. Under Macro $F_{0.5}$, accepting this recall tradeoff is mathematically optimal to guarantee high precision.

---

## 6. Conclusion

By shifting from hardcoded geographic heuristics to data-driven dynamic corpus IDF, multi-representation signatures, and universal relative similarity metrics, our solution achieves high-precision entity resolution that transfers seamlessly to unseen countries. Combined with bounded inverted index candidate generation and competitive injective assignment, the pipeline operates with sub-linear memory consumption and evaluates millions of records in minutes on standard CPU hardware.

---

## Appendix

### A. Code Artefacts
All runnable code is located in `code/business_entity_resolution/`:
- `src/features.py`: Universal Unicode NFKD normalization, multi-representation pre-parsing, and dynamic corpus IDF computation.
- `src/blocking.py`: Bounded inverted index using compact prefixes, IDF-filtered distinctive tokens, and compound number-street keys.
- `src/train.py`: Self-contained training script supporting balanced sampling, monotonic constraints, and validation Macro $F_{0.5}$ optimization.
- `src/pipeline.py`: Production-grade streaming test inference pipeline enforcing competitive 1-to-1 assignment and generating `matching_results.tsv` and `candidate_pairs.tsv`.
- `src/models/adaptive_matcher.pkl`: Pre-trained LightGBM adaptive model artifact.
- `requirements.txt`: Pinned environment dependencies.

### B. Execution Commands
To execute the pipeline end-to-end:
```bash
# 1. Run test inference
python code/business_entity_resolution/src/pipeline.py \
    --test-dir student_resource/dataset/test \
    --output-dir output

# 2. Validate output integrity
python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```
