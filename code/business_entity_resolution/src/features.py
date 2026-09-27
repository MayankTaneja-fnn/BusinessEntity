"""
features.py — Feature engineering for entity resolution pair scoring.

Computes string similarity features for (S1, S2/S3) candidate pairs:
- Name similarities (Levenshtein, Jaro-Winkler, token-based, n-gram Jaccard)
- Address similarities (same suite)
- Country match flag

All string distance functions are implemented from scratch (no external deps
beyond numpy/pandas).
"""

import re
import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from typing import Set


# ---------------------------------------------------------------------------
# String distance functions (implemented from scratch)
# ---------------------------------------------------------------------------

def _levenshtein(s1: str, s2: str) -> int:
    """
    Levenshtein edit distance using optimized single-row DP.
    For strings > 500 chars, return max length for efficiency.
    """
    if len(s1) > 500 or len(s2) > 500:
        return max(len(s1), len(s2))
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            ins = prev_row[j + 1] + 1
            dele = curr_row[j] + 1
            sub = prev_row[j] + (c1 != c2)
            curr_row.append(min(ins, dele, sub))
        prev_row = curr_row
    return prev_row[-1]


def norm_levenshtein(s1: str, s2: str) -> float:
    """Normalized Levenshtein similarity: 1 - dist/max_len."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    dist = _levenshtein(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1.0 - (dist / max_len) if max_len > 0 else 1.0


def jaro_winkler(s1: str, s2: str, p: float = 0.1, max_l: int = 4) -> float:
    """Jaro-Winkler similarity, implemented from scratch."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    len1, len2 = len(s1), len(s2)
    match_dist = max(len1, len2) // 2 - 1
    if match_dist < 0:
        match_dist = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0

    for i in range(len1):
        lo = max(0, i - match_dist)
        hi = min(i + match_dist + 1, len2)
        for j in range(lo, hi):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    # Count transpositions
    t = 0
    k = 0
    for i in range(len1):
        if s1_matches[i]:
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                t += 1
            k += 1
    t /= 2.0

    jaro = (matches / len1 + matches / len2 + (matches - t) / matches) / 3.0

    # Winkler prefix bonus
    prefix = 0
    for i in range(min(len1, len2, max_l)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return jaro + prefix * p * (1 - jaro)


def get_ngrams(s: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from a string."""
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    """Jaccard similarity of character n-grams."""
    set1 = get_ngrams(s1, n)
    set2 = get_ngrams(s2, n)
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def token_sort_ratio(s1: str, s2: str) -> float:
    """Sort tokens alphabetically, then SequenceMatcher ratio."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    t1 = " ".join(sorted(s1.split()))
    t2 = " ".join(sorted(s2.split()))
    return SequenceMatcher(None, t1, t2).ratio()


def token_set_ratio(s1: str, s2: str) -> float:
    """
    Ratio based on token set intersection/union — inspired by fuzzywuzzy.
    Compares: intersection alone, intersection+remainder1, intersection+remainder2.
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    set1 = set(s1.split())
    set2 = set(s2.split())
    if not set1 or not set2:
        return 0.0

    inter = set1 & set2
    diff1 = set1 - set2
    diff2 = set2 - set1

    s_inter = " ".join(sorted(inter))
    s_d1 = " ".join(sorted(diff1))
    s_d2 = " ".join(sorted(diff2))

    combined1 = (s_inter + " " + s_d1).strip()
    combined2 = (s_inter + " " + s_d2).strip()

    ratios = [
        SequenceMatcher(None, s_inter, combined1).ratio() if s_inter else 0.0,
        SequenceMatcher(None, s_inter, combined2).ratio() if s_inter else 0.0,
        SequenceMatcher(None, combined1, combined2).ratio(),
    ]
    return max(ratios)


def common_tokens_fraction(s1: str, s2: str) -> float:
    """Jaccard similarity on whitespace-tokenized sets."""
    if not s1 and not s2:
        return 1.0
    set1 = set(s1.split())
    set2 = set(s2.split())
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def numeric_overlap(s1: str, s2: str) -> float:
    """Jaccard similarity on numeric tokens (street numbers, PINs)."""
    nums1 = set(re.findall(r"\d+", s1))
    nums2 = set(re.findall(r"\d+", s2))
    if not nums1 and not nums2:
        return 1.0
    if not nums1 or not nums2:
        return 0.0
    return len(nums1 & nums2) / len(nums1 | nums2)


def len_ratio(s1: str, s2: str) -> float:
    """min(len)/max(len) ratio."""
    l1, l2 = len(s1), len(s2)
    if l1 == 0 and l2 == 0:
        return 1.0
    if l1 == 0 or l2 == 0:
        return 0.0
    return min(l1, l2) / max(l1, l2)


# ---------------------------------------------------------------------------
# Main feature computation
# ---------------------------------------------------------------------------

def compute_features_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise features for a DataFrame of candidate pairs.
    
    Expected columns: name1_norm, name2_norm, addr1_norm, addr2_norm, country1, country2
    
    Returns: DataFrame with one feature column per computed metric.
    """
    features = pd.DataFrame(index=df.index)

    # Extract columns, fill NaN with empty strings
    n1 = df.get("name1_norm", pd.Series("", index=df.index)).fillna("").astype(str).values
    n2 = df.get("name2_norm", pd.Series("", index=df.index)).fillna("").astype(str).values
    a1 = df.get("addr1_norm", pd.Series("", index=df.index)).fillna("").astype(str).values
    a2 = df.get("addr2_norm", pd.Series("", index=df.index)).fillna("").astype(str).values
    c1 = df.get("country1", pd.Series("", index=df.index)).fillna("").astype(str).values
    c2 = df.get("country2", pd.Series("", index=df.index)).fillna("").astype(str).values

    # --- Name features ---
    features["name_levenshtein"] = [norm_levenshtein(x, y) for x, y in zip(n1, n2)]
    features["name_jaro_winkler"] = [jaro_winkler(x, y) for x, y in zip(n1, n2)]
    features["name_token_sort_ratio"] = [token_sort_ratio(x, y) for x, y in zip(n1, n2)]
    features["name_token_set_ratio"] = [token_set_ratio(x, y) for x, y in zip(n1, n2)]
    features["name_ngram_jaccard"] = [ngram_jaccard(x, y) for x, y in zip(n1, n2)]
    features["name_len_ratio"] = [len_ratio(x, y) for x, y in zip(n1, n2)]
    features["name_exact_match"] = np.array([1 if x == y and x != "" else 0 for x, y in zip(n1, n2)], dtype=np.int8)
    features["name_len_diff"] = np.array([abs(len(x) - len(y)) for x, y in zip(n1, n2)], dtype=np.int32)
    features["name_common_tokens"] = [common_tokens_fraction(x, y) for x, y in zip(n1, n2)]

    # --- Address features ---
    features["addr_levenshtein"] = [norm_levenshtein(x, y) for x, y in zip(a1, a2)]
    features["addr_token_overlap"] = [common_tokens_fraction(x, y) for x, y in zip(a1, a2)]
    features["addr_numeric_overlap"] = [numeric_overlap(x, y) for x, y in zip(a1, a2)]
    features["addr_ngram_jaccard"] = [ngram_jaccard(x, y) for x, y in zip(a1, a2)]
    features["addr_len_ratio"] = [len_ratio(x, y) for x, y in zip(a1, a2)]
    features["addr_exact_match"] = np.array([1 if x == y and x != "" else 0 for x, y in zip(a1, a2)], dtype=np.int8)

    # --- Country feature ---
    features["same_country"] = np.array(
        [1 if x.strip().lower() == y.strip().lower() and x.strip() != "" else 0
         for x, y in zip(c1, c2)],
        dtype=np.int8,
    )

    # Placeholder for TF-IDF cosine similarity (filled externally if needed)
    features["combined_tfidf_sim"] = 0.0

    return features
