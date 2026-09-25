"""
features_adaptive.py
Universal feature extraction and dynamic country-level corpus IDF for adaptive entity resolution.
All similarity features are normalized to the unit interval [0, 1].
"""

import math
import re
import unicodedata
from collections import Counter, namedtuple
from typing import Dict, List, Optional, Set, Tuple, Union
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

# Precompiled regexes for maximum performance
RE_NON_ALPHANUM = re.compile(r'[^a-z0-9\s]+')
RE_DIGITS = re.compile(r'\b\d+\b')

# Feature names in exact order
FEATURE_NAMES = [
    'name_jw',
    'name_jw_compact',
    'name_token_sort',
    'name_token_set',
    'name_char_3gram_jaccard',
    'name_char_4gram_jaccard',
    'name_idf_weighted_jaccard',
    'name_exact_compact',
    'addr_token_sort',
    'addr_token_set',
    'addr_char_3gram_jaccard',
    'addr_number_jaccard',
    'addr_number_conflict',
    'addr_number_match_count',
    'addr_is_null',
    'addr_both_null',
    'is_source2'
]

# ParsedRecord structure with both named attributes and positional indexing
ParsedRecord = namedtuple(
    'ParsedRecord',
    [
        'name_norm',
        'name_sig',
        'name_words',
        'name_words_set',
        'name_3g',
        'name_4g',
        'addr_norm',
        'addr_words',
        'addr_nums',
        'addr_3g',
        'is_null',
        'first_street_word'
    ]
)


def normalize_text(text: Optional[str]) -> str:
    """Universal NFKD Unicode normalization:
    - Diacritics and accents stripped
    - '&' and '+' converted to 'and'
    - '@' converted to 'at'
    - Lowercased and stripped of non-alphanumeric punctuation
    """
    if text is None:
        return ''
    s = str(text)
    if not s or s == 'None' or s == 'nan':
        return ''
    s = s.replace('&', ' and ').replace('+', ' and ').replace('@', ' at ')
    s = ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c)).lower()
    return ' '.join(RE_NON_ALPHANUM.sub(' ', s).split())


def _extract_char_ngrams(s: str, n: int) -> Set[str]:
    """Extract character n-grams from string."""
    if not s:
        return set()
    if len(s) < n:
        return {s}
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def _extract_first_street_word(addr_words: List[str]) -> str:
    """Extract the first distinctive non-numeric street token adjacent to the first house number."""
    digits = [w for w in addr_words if w.isdigit()]
    if not digits:
        return ''
    first_digit = digits[0]
    try:
        idx = addr_words.index(first_digit)
        # Search after the house number for a word of length >= 3
        after = [w for w in addr_words[idx + 1:] if not w.isdigit() and len(w) >= 3]
        if after:
            return after[0]
        # Otherwise search before the house number
        before = [w for w in addr_words[:idx] if not w.isdigit() and len(w) >= 3]
        if before:
            return before[-1]
    except ValueError:
        pass
    return ''


def parse_record(name: Optional[str], addr: Optional[str]) -> ParsedRecord:
    """High-speed pre-parsing function for business records.
    Produces multi-representations:
    - tokenized words list
    - compact alphanumeric signature (all punctuation and spaces removed)
    - set of numeric tokens (digits)
    - character n-grams for fast similarity computation
    """
    # 1. Name parsing
    name_norm = normalize_text(name)
    if name_norm:
        name_words = name_norm.split()
        name_words_set = set(name_words)
        name_sig = ''.join(name_words)
        name_3g = _extract_char_ngrams(name_norm, 3)
        name_4g = _extract_char_ngrams(name_norm, 4)
    else:
        name_words = []
        name_words_set = set()
        name_sig = ''
        name_3g = set()
        name_4g = set()

    # 2. Address parsing
    addr_norm = normalize_text(addr)
    if addr_norm:
        addr_words = addr_norm.split()
        addr_nums = {w for w in addr_words if w.isdigit()}
        addr_3g = _extract_char_ngrams(addr_norm, 3)
        is_null = 0.0
        street = _extract_first_street_word(addr_words)
    else:
        addr_words = []
        addr_nums = set()
        addr_3g = set()
        is_null = 1.0
        street = ''

    return ParsedRecord(
        name_norm=name_norm,
        name_sig=name_sig,
        name_words=name_words,
        name_words_set=name_words_set,
        name_3g=name_3g,
        name_4g=name_4g,
        addr_norm=addr_norm,
        addr_words=addr_words,
        addr_nums=addr_nums,
        addr_3g=addr_3g,
        is_null=is_null,
        first_street_word=street
    )


class CorpusIDF:
    """Dynamic Country-Level Corpus IDF container."""

    def __init__(self, idf: Dict[str, float], stopwords: Set[str], df: Dict[str, int], default_idf: float, n_records: int):
        self.idf = idf
        self.stopwords = stopwords
        self.df = df
        self.default_idf = default_idf
        self.n_records = n_records

    def get_idf(self, token: str) -> float:
        return self.idf.get(token, self.default_idf)

    def __iter__(self):
        return iter((self.idf, self.stopwords, self.df))

    def __getitem__(self, item):
        return (self.idf, self.stopwords, self.df)[item]


def compute_corpus_idf(records_parsed: List[ParsedRecord]) -> CorpusIDF:
    """Computes document frequency DF for all word tokens across the target pool (S2+S3) of that country.
    - Tokens with DF > 0.005 * N (or top 100) are marked as corpus stopwords.
    - IDF computed as ln(1 + N / (1 + DF)).
    """
    n_records = len(records_parsed)
    if n_records == 0:
        return CorpusIDF({}, set(), {}, 0.0, 0)

    df_counter = Counter()
    for r in records_parsed:
        df_counter.update(r.name_words_set)

    # Threshold for corpus stopwords: DF > 0.005 * N
    df_threshold = 0.005 * n_records
    stopwords = {w for w, count in df_counter.items() if count > df_threshold}
    # Union with top 100 most frequent tokens (with count >= 2)
    top_100 = {w for w, count in df_counter.most_common(100) if count >= 2}
    stopwords.update(top_100)

    default_idf = math.log(1.0 + float(n_records))
    idf_dict = {
        w: math.log(1.0 + float(n_records) / (1.0 + float(c)))
        for w, c in df_counter.items()
    }

    return CorpusIDF(
        idf=idf_dict,
        stopwords=stopwords,
        df=dict(df_counter),
        default_idf=default_idf,
        n_records=n_records
    )


def compute_parsed_features(
    p1: ParsedRecord,
    p2: ParsedRecord,
    is_s2: Union[bool, int, float],
    idf_dict: Optional[Dict[str, float]] = None,
    default_idf: Optional[float] = None
) -> List[float]:
    """Universal relative similarity features (all in [0, 1]):
    Name:
      - name_jw
      - name_jw_compact
      - name_token_sort
      - name_token_set
      - name_char_3gram_jaccard
      - name_char_4gram_jaccard
      - name_idf_weighted_jaccard
      - name_exact_compact
    Address:
      - addr_token_sort
      - addr_token_set
      - addr_char_3gram_jaccard
      - addr_number_jaccard
      - addr_number_conflict (1.0 if both have numbers but 0 in common, else 0.0)
      - addr_number_match_count
      - addr_is_null (1.0 if target address is None)
      - addr_both_null
    Source indicator:
      - is_source2
    """
    # 1. Name features
    # Jaro-Winkler
    f_name_jw = JaroWinkler.similarity(p1.name_norm, p2.name_norm)
    f_name_jw_compact = JaroWinkler.similarity(p1.name_sig, p2.name_sig)

    # Token sort & set
    f_name_token_sort = fuzz.token_sort_ratio(p1.name_norm, p2.name_norm) / 100.0
    f_name_token_set = fuzz.token_set_ratio(p1.name_norm, p2.name_norm) / 100.0

    # Char 3-gram & 4-gram Jaccard
    u3 = len(p1.name_3g | p2.name_3g)
    f_name_char_3gram = len(p1.name_3g & p2.name_3g) / u3 if u3 > 0 else 0.0

    u4 = len(p1.name_4g | p2.name_4g)
    f_name_char_4gram = len(p1.name_4g & p2.name_4g) / u4 if u4 > 0 else 0.0

    # Name IDF-weighted Jaccard
    union_words = p1.name_words_set | p2.name_words_set
    if not union_words:
        f_name_idf_jaccard = 0.0
    else:
        inter_words = p1.name_words_set & p2.name_words_set
        if not inter_words:
            f_name_idf_jaccard = 0.0
        elif inter_words == union_words:
            f_name_idf_jaccard = 1.0
        else:
            def_idf = default_idf if default_idf is not None else 1.0
            if idf_dict:
                sum_inter = sum(idf_dict.get(w, def_idf) for w in inter_words)
                sum_union = sum(idf_dict.get(w, def_idf) for w in union_words)
            else:
                sum_inter = float(len(inter_words))
                sum_union = float(len(union_words))
            f_name_idf_jaccard = min(max(sum_inter / sum_union, 0.0), 1.0) if sum_union > 0 else 0.0

    # Exact compact match
    f_name_exact_compact = 1.0 if (p1.name_sig and p1.name_sig == p2.name_sig) else 0.0

    # 2. Address features
    null1 = p1.is_null >= 0.5
    null2 = p2.is_null >= 0.5
    f_addr_is_null = 1.0 if null2 else 0.0
    f_addr_both_null = 1.0 if (null1 and null2) else 0.0

    if null1 or null2:
        f_addr_token_sort = 0.0
        f_addr_token_set = 0.0
        f_addr_char_3gram = 0.0
        f_addr_number_jaccard = 0.0
        f_addr_number_conflict = 0.0
        f_addr_number_match_count = 0.0
    else:
        f_addr_token_sort = fuzz.token_sort_ratio(p1.addr_norm, p2.addr_norm) / 100.0
        f_addr_token_set = fuzz.token_set_ratio(p1.addr_norm, p2.addr_norm) / 100.0

        u_addr3 = len(p1.addr_3g | p2.addr_3g)
        f_addr_char_3gram = len(p1.addr_3g & p2.addr_3g) / u_addr3 if u_addr3 > 0 else 0.0

        num_inter = len(p1.addr_nums & p2.addr_nums)
        num_union = len(p1.addr_nums | p2.addr_nums)

        f_addr_number_jaccard = num_inter / num_union if num_union > 0 else 0.0
        f_addr_number_conflict = 1.0 if (len(p1.addr_nums) > 0 and len(p2.addr_nums) > 0 and num_inter == 0) else 0.0
        f_addr_number_match_count = min(num_inter / 3.0, 1.0)

    # 3. Source indicator
    f_is_source2 = 1.0 if bool(is_s2) else 0.0

    return [
        f_name_jw,
        f_name_jw_compact,
        f_name_token_sort,
        f_name_token_set,
        f_name_char_3gram,
        f_name_char_4gram,
        f_name_idf_jaccard,
        f_name_exact_compact,
        f_addr_token_sort,
        f_addr_token_set,
        f_addr_char_3gram,
        f_addr_number_jaccard,
        f_addr_number_conflict,
        f_addr_number_match_count,
        f_addr_is_null,
        f_addr_both_null,
        f_is_source2
    ]


# Alias for compatibility
compute_features = compute_parsed_features
