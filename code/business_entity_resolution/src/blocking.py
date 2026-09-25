"""
blocking_adaptive.py
Adaptive inverted index for entity resolution blocking with bounded candidate sizes.
Produces an average of 5-15 deduplicated candidates per S1 entity with sub-millisecond query latency.
"""

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union
try:
    from .features import CorpusIDF, ParsedRecord, compute_corpus_idf
except ImportError:
    try:
        from features import CorpusIDF, ParsedRecord, compute_corpus_idf
    except ImportError:
        from features_adaptive import CorpusIDF, ParsedRecord, compute_corpus_idf


class AdaptiveInvertedIndex:
    """Inverted index with bounded candidate size (5-15 candidates per S1 entity):
    - Compact prefix keys: sig[:6] and sig[:8] (if length >= 5)
    - IDF-filtered distinctive name tokens (only tokens with 2 <= DF <= 300, ignoring high-frequency stopwords)
    - Compound house number + first street token: f'num_{digit}_{first_street_word}'
    - Cap each posting list to max_posting=30
    - Deduplicate candidates per S1 entity
    """

    def __init__(
        self,
        max_posting: int = 30,
        max_candidates: int = 15,
        stopwords: Optional[Set[str]] = None,
        df_dict: Optional[Dict[str, int]] = None
    ):
        self.index = defaultdict(list)
        self.max_posting = max_posting
        self.max_candidates = max_candidates
        self.stopwords = stopwords if stopwords is not None else set()
        self.df = df_dict if df_dict is not None else {}
        self.idf = {}

    def fit_corpus(self, parsed_records: List[ParsedRecord]) -> None:
        """Fit country-level corpus document frequency and stopwords from the target pool."""
        corpus = compute_corpus_idf(parsed_records)
        self.idf = corpus.idf
        self.stopwords = corpus.stopwords
        self.df = corpus.df

    def add_records(
        self,
        parsed_records: List[ParsedRecord],
        is_s2_flag: Union[int, bool],
        stopwords: Optional[Set[str]] = None,
        df_dict: Optional[Dict[str, int]] = None
    ) -> None:
        """Add parsed target records (S2 or S3) to the inverted index."""
        sw = stopwords if stopwords is not None else self.stopwords
        df = df_dict if df_dict is not None else self.df
        flag = 1 if bool(is_s2_flag) else 0

        for i, r in enumerate(parsed_records):
            # 1. Compact prefix keys: sig[:6] and sig[:8] (if length >= 5)
            sig = r.name_sig
            if len(sig) >= 5:
                k6 = f"p6_{sig[:6]}"
                lst6 = self.index[k6]
                if len(lst6) < self.max_posting:
                    lst6.append((flag, i))

                if len(sig) >= 8:
                    k8 = f"p8_{sig[:8]}"
                    lst8 = self.index[k8]
                    if len(lst8) < self.max_posting:
                        lst8.append((flag, i))

            # 2. IDF-filtered distinctive name tokens (only tokens with 2 <= DF <= 300)
            for w in r.name_words_set:
                if len(w) >= 3 and (w not in sw):
                    token_df = df.get(w, 0)
                    if 2 <= token_df <= 300:
                        k_tok = f"t_{w}"
                        lst_tok = self.index[k_tok]
                        if len(lst_tok) < self.max_posting:
                            lst_tok.append((flag, i))

            # 3. Compound house number + first street token: f'num_{digit}_{first_street_word}'
            if (r.is_null < 0.5) and r.addr_nums and r.first_street_word:
                digit = sorted(list(r.addr_nums))[0]
                k_addr = f"num_{digit}_{r.first_street_word}"
                lst_addr = self.index[k_addr]
                if len(lst_addr) < self.max_posting:
                    lst_addr.append((flag, i))

    def query(
        self,
        parsed_s1: ParsedRecord,
        stopwords: Optional[Set[str]] = None,
        df_dict: Optional[Dict[str, int]] = None
    ) -> List[Tuple[int, int]]:
        """Query inverted index for an S1 entity and return bounded deduplicated candidates."""
        sw = stopwords if stopwords is not None else self.stopwords
        df = df_dict if df_dict is not None else self.df
        cand_scores = Counter()

        # 1. Compact prefix lookup
        sig = parsed_s1.name_sig
        if len(sig) >= 8 and f"p8_{sig[:8]}" in self.index:
            for item in self.index[f"p8_{sig[:8]}"]:
                cand_scores[item] += 3
        elif len(sig) >= 5 and f"p6_{sig[:6]}" in self.index:
            for item in self.index[f"p6_{sig[:6]}"]:
                cand_scores[item] += 2

        # 2. Distinctive name tokens sorted by DF (most distinctive first)
        tokens = [
            w for w in parsed_s1.name_words_set
            if len(w) >= 3 and (w not in sw) and (2 <= df.get(w, 0) <= 300)
        ]
        if tokens:
            tokens.sort(key=lambda w: df.get(w, 0))
            for w in tokens[:3]:  # Top 3 most distinctive tokens
                for item in self.index.get(f"t_{w}", []):
                    cand_scores[item] += 1

        # 3. Compound house number + street token
        if (parsed_s1.is_null < 0.5) and parsed_s1.addr_nums and parsed_s1.first_street_word:
            digit = sorted(list(parsed_s1.addr_nums))[0]
            k_addr = f"num_{digit}_{parsed_s1.first_street_word}"
            for item in self.index.get(k_addr, []):
                cand_scores[item] += 2

        if not cand_scores:
            return []

        # Return bounded candidates: if candidates <= max_candidates return all, else top ranked
        if len(cand_scores) <= self.max_candidates:
            return list(cand_scores.keys())

        return [cand for cand, _ in cand_scores.most_common(self.max_candidates)]


# Alias for compatibility with previous codebase
CountryInvertedIndex = AdaptiveInvertedIndex
