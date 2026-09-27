"""Pairwise feature engineering for (Source-1, candidate) pairs.

Runs only over the candidate set produced by blocking — never the full cross
product — so it stays cheap even though the corpus itself is tens of millions
of rows. Features combine cheap set-based similarity (token/char-ngram
Jaccard, computed in Python) with rapidfuzz's C-implemented edit-distance
metrics, plus the raw blocking `score` (how many blocking channels agreed).
"""
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz

from normalize import core_name_tokens, squeeze, char_ngrams, address_tokens_for_blocking, digit_tokens
from progress import log, auto_n_jobs, scale_to_ram

# Reference chunk size for an m6i.2xlarge at the default 70% RAM budget;
# scaled to the actual box via scale_to_ram() so larger/smaller instances
# split work into proportionally larger/smaller chunks.
REFERENCE_CHUNK_ROWS = 200_000
# Each worker holds one chunk of Python row-tuples + rapidfuzz scratch space
# in memory at a time; ~0.5 GB/worker is a conservative estimate used to cap
# pool width so wide pools can't collectively bust the RAM budget.
MEM_PER_WORKER_GB = 0.5

FEATURE_COLUMNS = [
    "blocking_score",
    "country_match",
    "name_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "name_squeeze_ratio",
    "name_token_jaccard",
    "name_char_ngram_jaccard",
    "name_len_ratio",
    "name_first_token_match",
    "addr_both_missing",
    "addr_one_missing",
    "addr_ratio",
    "addr_token_sort_ratio",
    "addr_token_jaccard",
    "digit_jaccard",
    "digit_overlap_count",
]


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    u = len(a | b)
    if u == 0:
        return 0.0
    return len(a & b) / u


def _row_features(name1, name2, addr1, addr2, country1, country2, cname1, cname2, caddr1, caddr2, score):
    name1 = name1 or ""
    name2 = name2 or ""
    cname1 = cname1 or ""
    cname2 = cname2 or ""
    caddr1 = caddr1 or ""
    caddr2 = caddr2 or ""

    sq1, sq2 = squeeze(cname1), squeeze(cname2)
    tok1, tok2 = set(core_name_tokens(cname1)), set(core_name_tokens(cname2))
    ng1, ng2 = char_ngrams(cname1, 3), char_ngrams(cname2, 3)

    addr1_missing = 1.0 if not caddr1.strip() else 0.0
    addr2_missing = 1.0 if not caddr2.strip() else 0.0
    both_missing = 1.0 if (addr1_missing and addr2_missing) else 0.0
    one_missing = 1.0 if (addr1_missing != addr2_missing) else 0.0

    atok1, atok2 = set(address_tokens_for_blocking(caddr1)), set(address_tokens_for_blocking(caddr2))
    dtok1, dtok2 = set(digit_tokens(caddr1)), set(digit_tokens(caddr2))

    len1, len2 = len(sq1), len(sq2)
    len_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0.0
    first_tok_match = 1.0 if (tok1 and tok2 and next(iter(sorted(tok1))) == next(iter(sorted(tok2)))) else 0.0

    return (
        float(score),
        1.0 if country1 and country2 and country1 == country2 else 0.0,
        fuzz.ratio(cname1, cname2) / 100.0,
        fuzz.token_sort_ratio(cname1, cname2) / 100.0,
        fuzz.token_set_ratio(cname1, cname2) / 100.0,
        fuzz.partial_ratio(sq1, sq2) / 100.0,
        fuzz.ratio(sq1, sq2) / 100.0,
        _jaccard(tok1, tok2),
        _jaccard(ng1, ng2),
        len_ratio,
        first_tok_match,
        both_missing,
        one_missing,
        fuzz.ratio(caddr1, caddr2) / 100.0 if not both_missing else 0.0,
        fuzz.token_sort_ratio(caddr1, caddr2) / 100.0 if not both_missing else 0.0,
        _jaccard(atok1, atok2),
        _jaccard(dtok1, dtok2),
        float(len(dtok1 & dtok2)),
    )


def compute_features(pairs: pl.DataFrame) -> np.ndarray:
    """``pairs`` must have columns: business_name/cleaned_name/business_address/
    cleaned_addr/country for both sides, prefixed ``s1_`` and ``o_``, plus a
    ``score`` column from blocking. Returns an (n, len(FEATURE_COLUMNS)) float32
    array, row order matching ``pairs``."""
    cols = pairs.select(
        "s1_business_name", "o_business_name",
        "s1_business_address", "o_business_address",
        "s1_country", "o_country",
        "s1_cleaned_name", "o_cleaned_name",
        "s1_cleaned_addr", "o_cleaned_addr",
        "score",
    ).rows()

    feats = [_row_features(*row) for row in cols]
    return np.asarray(feats, dtype=np.float32)


def _compute_chunk(rows):
    return [_row_features(*row) for row in rows]


def compute_features_parallel(pairs: pl.DataFrame, n_jobs: int = None, chunk_rows: int = None) -> np.ndarray:
    """Same as ``compute_features`` but splits ``pairs`` into chunks processed
    by a process pool — the per-pair work is pure-Python + rapidfuzz calls, so
    multiprocessing gives a near-linear speedup on the multi-core instance
    this is meant to run on at full scale.

    ``n_jobs`` and ``chunk_rows`` default to RAM-aware values (capped at 70%
    of system RAM, see ``progress.py``) so the pool width scales with the
    instance instead of a value hand-picked for one specific machine."""
    from multiprocessing import Pool

    n_jobs = n_jobs or auto_n_jobs(MEM_PER_WORKER_GB)
    chunk_rows = chunk_rows or scale_to_ram(REFERENCE_CHUNK_ROWS)
    n = pairs.height
    if n == 0:
        return np.empty((0, len(FEATURE_COLUMNS)), dtype=np.float32)
    if n_jobs <= 1 or n <= chunk_rows:
        return compute_features(pairs)

    cols = pairs.select(
        "s1_business_name", "o_business_name",
        "s1_business_address", "o_business_address",
        "s1_country", "o_country",
        "s1_cleaned_name", "o_cleaned_name",
        "s1_cleaned_addr", "o_cleaned_addr",
        "score",
    ).rows()

    chunks = [cols[i:i + chunk_rows] for i in range(0, len(cols), chunk_rows)]
    n_chunks = len(chunks)
    log(f"  feature engineering: {n} pairs -> {n_chunks} chunks x {n_jobs} workers")

    results = [None] * n_chunks
    t0 = time.time()
    report_every = max(1, n_chunks // 10)
    with Pool(n_jobs) as pool:
        for i, res in enumerate(pool.imap(_compute_chunk, chunks)):
            results[i] = res
            done = i + 1
            if done == n_chunks or done % report_every == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (n_chunks - done) / rate if rate > 0 else 0.0
                log(f"    chunks {done}/{n_chunks}  elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    feats = [row for chunk in results for row in chunk]
    return np.asarray(feats, dtype=np.float32)
