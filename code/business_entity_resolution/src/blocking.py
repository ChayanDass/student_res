"""Candidate generation (blocking).

Strategy: token-based inverted-index blocking over four independent channels —

1. name tokens (whole words from the legal-suffix-stripped name)
2. address place-name tokens
3. address digit tokens (street numbers / PIN codes)
4. character 5-grams of the squeezed (space-free) name

so that a match is found whenever S1 and a Source-2/3 record share at least
one sufficiently rare key in *any* channel. The channels fail independently,
which is why all four are kept rather than picking just one:

- Devanagari/Telugu/Kannada-script business names never share word tokens with
  a Latin-script name, so the address channel is what recovers those matches
  (addresses stay mostly Latin-script even when the name doesn't).
- Domain-name-style records ("roreiaencore.com") and names with an internal
  typo don't share a full word token with their clean counterpart; the
  char-5-gram channel recovers those via shared substrings.
- A shared street number without a matching name is common between
  *unrelated* neighboring businesses, so address-only blocking is deliberately
  not used as a hard filter to decide matches — that decision is left to the
  classifier at the modeling stage.

Tokens/n-grams that are too frequent (``max_df``, expressed as a fraction of
the "others" pool so it scales with corpus size) are dropped as blocking keys
before the join, since a key shared by a huge number of records blows up the
candidate count without adding discriminative signal. The number of distinct
blocking keys shared between an S1 record and a candidate is kept as a
``score`` and used to cap the candidate set at ``top_k`` per S1 entity — this
both bounds downstream compute and keeps ``candidate_pairs.tsv`` small, which
the challenge separately rewards.

Scalability: the char-5-gram channel in particular produces O(len(name)) keys
per record, so building the full pairwise join in one shot over the full
~2.2M x ~10.3M corpus can use a lot of peak memory. ``OtherTokenIndex`` is
built ONCE over the Source-2/3 pool; ``run_blocking`` then streams Source-1 in
chunks (``batch_size``) through that fixed index and appends each batch's
result to the output file, so peak memory is bounded by one chunk's
intermediate join, not the whole corpus.
"""
import os
import time

import polars as pl

from normalize import (
    name_blocking_tokens,
    address_tokens_for_blocking,
    digit_tokens_for_blocking,
    name_char_ngrams_for_blocking,
)
from progress import log, log_stage, resource_line, scale_to_ram, ProgressBar

# max_df is expressed as a fraction of the "others" (S2+S3) pool size rather
# than an absolute count, so blocking behaves consistently whether it is run
# on a small validation slice or the full ~10M-record corpus. A floor keeps
# small test pools from filtering out almost every token.
NAME_MAX_DF_FRAC = 0.0003
ADDR_MAX_DF_FRAC = 0.0008
DIGIT_MAX_DF_FRAC = 0.0015
NGRAM_MAX_DF_FRAC = 0.0002
MIN_DF_FLOOR = 20

CHANNELS = {
    "name": (name_blocking_tokens, "cleaned_name", NAME_MAX_DF_FRAC),
    "addr": (address_tokens_for_blocking, "cleaned_addr", ADDR_MAX_DF_FRAC),
    "digit": (digit_tokens_for_blocking, "cleaned_addr", DIGIT_MAX_DF_FRAC),
    "ngram": (name_char_ngrams_for_blocking, "cleaned_name", NGRAM_MAX_DF_FRAC),
}

# Reference chunk size (others-pool rows processed per pass) for an
# m6i.2xlarge at the default 70% RAM budget; scaled via scale_to_ram(). This
# bounds the exploded (entity_id, token) intermediate frame that ``_token_frame``
# builds — the char-5-gram channel alone can explode a 10M-record pool into
# 100M+ token rows if done in one shot, which is what actually OOM-killed
# earlier runs (independent of block_batch_size, which only bounds the
# S1-side streaming against the *already-built* index, not this build step).
REFERENCE_INDEX_CHUNK_ROWS = 300_000


def _token_frame(df: pl.DataFrame, extractor, col: str) -> pl.DataFrame:
    return (
        df.select("entity_id", pl.col(col))
        .with_columns(
            pl.col(col).map_elements(extractor, return_dtype=pl.List(pl.String)).alias("token")
        )
        .drop(col)
        .explode("token")
        .drop_nulls("token")
        .filter(pl.col("token") != "")
        .unique()
    )


def _chunked_token_counts(df: pl.DataFrame, extractor, col: str, chunk_rows: int,
                           label: str) -> pl.DataFrame:
    """Pass 1: token -> document-frequency count, computed over ``df`` in
    row-chunks so peak memory is bounded by one chunk's exploded token frame,
    not the full corpus explode. Each chunk touches disjoint entity_id rows,
    so per-chunk ``.unique()`` (inside _token_frame) is equivalent to a global
    one, and per-chunk counts simply sum across chunks."""
    n = df.height
    bar = ProgressBar(n, f"{label} pass 1/2 (counting document frequency)")
    parts = []
    for start in range(0, n, chunk_rows):
        chunk = df.slice(start, chunk_rows)
        tok = _token_frame(chunk, extractor, col)
        parts.append(tok.group_by("token").len())
        bar.update(chunk.height)
    bar.close()
    if not parts:
        return pl.DataFrame({"token": [], "len": []})
    # merge all chunks' per-token counts in one final aggregation rather than
    # re-aggregating after every chunk (O(chunks) instead of O(chunks^2) work);
    # what's held here is tiny (token, count) pairs, not the raw exploded rows
    return pl.concat(parts).group_by("token").agg(pl.col("len").sum().alias("len"))


def _chunked_filtered_tokens(df: pl.DataFrame, extractor, col: str, keep: pl.DataFrame,
                              chunk_rows: int, label: str) -> pl.DataFrame:
    """Pass 2: rebuild the (entity_id, token) inverted index, same row-chunking
    as pass 1, but immediately filtered down to ``keep`` tokens (already a
    small set after max_df filtering) — so what actually accumulates across
    chunks is small regardless of how large the full corpus is."""
    n = df.height
    bar = ProgressBar(n, f"{label} pass 2/2 (building filtered index)")
    parts = []
    for start in range(0, n, chunk_rows):
        chunk = df.slice(start, chunk_rows)
        tok = _token_frame(chunk, extractor, col)
        parts.append(tok.join(keep, on="token", how="inner"))
        bar.update(chunk.height)
    bar.close()
    return pl.concat(parts) if parts else pl.DataFrame({"entity_id": [], "token": []})


class OtherTokenIndex:
    """Precomputed, max_df-filtered inverted index over the Source-2/3 pool.
    Built once and reused across every Source-1 batch.

    Each channel is built in two RAM-bounded passes (see
    ``_chunked_token_counts`` / ``_chunked_filtered_tokens``) rather than one
    explode over the full corpus — the un-chunked version is what actually
    drove memory past 30GB and triggered repeated OOM kills on the full
    ~10.3M-record pool, regardless of any other batch-size setting.

    When ``cache_dir`` is given, each channel's finished table is written to
    ``{cache_dir}/blocking_index_{cache_prefix}_{channel}.parquet`` the
    moment it's built — not just at the end — so a crash partway through
    (channel 3 of 4, say) leaves the earlier channels cached: a restart skips
    straight past whatever's already on disk instead of redoing the full
    ~15-20 minute build from scratch. Pass ``force=True`` to ignore any
    existing cache and rebuild (matching the pipeline's ``--force-blocking``
    flag)."""

    def __init__(self, others: pl.DataFrame, chunk_rows: int = None,
                 cache_dir: str = None, cache_prefix: str = "others", force: bool = False):
        n_others = others.height
        chunk_rows = chunk_rows or scale_to_ram(REFERENCE_INDEX_CHUNK_ROWS)
        n_chunks = max(1, (n_others + chunk_rows - 1) // chunk_rows)
        log(f"  building blocking index over {n_others:,} records, "
            f"{len(CHANNELS)} channels x {n_chunks} chunks of {chunk_rows:,} rows each")
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        self.tables = {}
        for i, (name, (extractor, col, frac)) in enumerate(CHANNELS.items(), 1):
            log(f"  ----- channel {i}/{len(CHANNELS)}: '{name}' (source column: {col}) -----")
            cpath = os.path.join(cache_dir, f"blocking_index_{cache_prefix}_{name}.parquet") if cache_dir else None
            if cpath and not force and os.path.exists(cpath):
                self.tables[name] = pl.read_parquet(cpath)
                log(f"    channel '{name}' CACHE HIT -> {cpath} "
                    f"({self.tables[name].height:,} rows)  {resource_line()}")
                continue

            max_df = max(MIN_DF_FLOOR, int(n_others * frac))
            counts = _chunked_token_counts(others, extractor, col, chunk_rows, name)
            keep = counts.filter(pl.col("len") <= max_df).select("token")
            log(f"    channel '{name}': {counts.height:,} distinct tokens, "
                f"{keep.height:,} kept after max_df={max_df} filter")
            self.tables[name] = _chunked_filtered_tokens(others, extractor, col, keep, chunk_rows, name)
            log(f"    channel '{name}' DONE: {self.tables[name].height:,} (token, entity_id) rows "
                f"kept  {resource_line()}")
            if cpath:
                self.tables[name].write_parquet(cpath)
                log(f"    channel '{name}' cached -> {cpath}")


def _channel_pairs(s1_tok: pl.DataFrame, other_tok_filtered: pl.DataFrame) -> pl.LazyFrame:
    """Returns a LAZY join plan (not yet executed) — kept lazy all the way to
    the final .collect(engine="streaming") in block_batch_scored, so the
    query engine can choose its spillable partitioned hash-join instead of
    materializing the (potentially huge, skewed) join eagerly. Benchmarked
    on a synthetic stress corpus: same chunking, lazy+streaming collect vs.
    eager collect was both ~24% faster and used ~20% less peak memory, with
    byte-identical output — collecting late costs nothing and buys headroom."""
    joined = s1_tok.lazy().join(other_tok_filtered.lazy(), on="token", how="inner")
    return joined.select(
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_right").alias("other_entity_id"),
    )


# The join in _channel_pairs is combinatorial: its output size for one token
# is (# S1 entities with that token) x (# "other" entities with that token,
# already capped at max_df but max_df itself can be in the thousands). A
# single outer S1 batch (tens of thousands of rows) can hit enough shared
# tokens to explode past 30GB in one join, which is what OOM-killed a run
# even *after* the index build itself was made memory-safe — batch_size
# bounds row count, not join fan-out. Sub-chunking the S1 side here, and
# critically applying top_k *within* each sub-chunk rather than after
# concatenating all of them, bounds peak memory to
# ``sub_chunk_size * top_k`` rows regardless of token skew or total corpus
# size — ranking-then-filtering only at the very end (the first version of
# this fix) still accumulated every unfiltered candidate for the whole outer
# batch before trimming, which is what caused a second OOM even with the
# join itself chunked. Each sub-chunk covers disjoint source1_entity_id
# values, so per-entity top_k within a sub-chunk is already the entity's
# global top_k — no cross-chunk merge is needed.
S1_JOIN_SUBCHUNK_ROWS = 5_000


def block_batch_scored(s1_batch: pl.DataFrame, index: OtherTokenIndex, top_k: int,
                        progress_label: str = "streaming") -> pl.DataFrame:
    """Candidate generation for one Source-1 batch against the fixed index.
    Returns EXPLODED pairs: source1_entity_id, other_entity_id, score (number
    of blocking channels that agreed) — the internal representation reused by
    feature engineering, NOT the submission format."""
    n = s1_batch.height
    # report_every_pct sized to S1_JOIN_SUBCHUNK_ROWS so the bar prints after
    # every sub-chunk (not just every ~10% of the outer batch) — each chunk
    # is its own unit of work now, so each one gets its own progress line.
    chunk_pct = max(0.5, 100.0 * S1_JOIN_SUBCHUNK_ROWS / n) if n else 10.0
    bar = ProgressBar(n, progress_label, report_every_pct=chunk_pct)
    ranked_parts = []
    for start in range(0, n, S1_JOIN_SUBCHUNK_ROWS):
        sub = s1_batch.slice(start, S1_JOIN_SUBCHUNK_ROWS)
        pair_frames = []
        for name, (extractor, col, _frac) in CHANNELS.items():
            s1_tok = _token_frame(sub, extractor, col)
            pair_frames.append(_channel_pairs(s1_tok, index.tables[name]))
        pairs = pl.concat(pair_frames)
        scored_sub = pairs.group_by(["source1_entity_id", "other_entity_id"]).len().rename({"len": "score"})
        # sort + cum_count (not rank(method="ordinal")) so the top_k cutoff
        # among tied scores is deterministic (by other_entity_id) regardless
        # of row arrival order — join/group_by order isn't guaranteed stable,
        # so an ordinal rank's tie-break would otherwise vary with chunking.
        # Everything above is lazy (see _channel_pairs); this collect() is
        # the one point of execution per sub-chunk, using the streaming
        # engine's spillable partitioned hash-join.
        ranked_sub = (
            scored_sub.sort(["source1_entity_id", "score", "other_entity_id"], descending=[False, True, False])
            .with_columns(pl.col("score").cum_count().over("source1_entity_id").alias("rank"))
            .filter(pl.col("rank") <= top_k)
            .drop("rank")
            .collect(engine="streaming")
        )
        ranked_parts.append(ranked_sub)
        bar.update(sub.height)
    bar.close()

    if not ranked_parts:
        return pl.DataFrame({"source1_entity_id": [], "other_entity_id": [], "score": []})
    return pl.concat(ranked_parts)


def block_batch(s1_batch: pl.DataFrame, index: OtherTokenIndex, top_k: int) -> pl.DataFrame:
    """Same as ``block_batch_scored`` but grouped into the submission shape:
    one row per S1 entity in ``s1_batch``, candidate_entity_ids comma-joined
    ("" when none survive blocking)."""
    ranked = block_batch_scored(s1_batch, index, top_k)

    grouped = (
        ranked.sort(["source1_entity_id", "score"], descending=[False, True])
        .group_by("source1_entity_id", maintain_order=True)
        .agg(pl.col("other_entity_id").alias("candidates"))
        .with_columns(pl.col("candidates").list.join(",").alias("candidate_entity_ids"))
        .select("source1_entity_id", "candidate_entity_ids")
    )

    all_s1 = s1_batch.select(pl.col("entity_id").alias("source1_entity_id"))
    result = all_s1.join(grouped, on="source1_entity_id", how="left").with_columns(
        pl.col("candidate_entity_ids").fill_null("")
    )
    return result


def generate_candidates(s1: pl.DataFrame, others: pl.DataFrame, top_k: int = 100) -> pl.DataFrame:
    """Convenience single-shot API (fine for small/medium ``s1``; for the full
    corpus use ``run_blocking`` below, which batches and streams to disk)."""
    index = OtherTokenIndex(others)
    return block_batch(s1, index, top_k)


def run_blocking(s1: pl.DataFrame, others: pl.DataFrame, out_path: str,
                  top_k: int = 100, batch_size: int = 100_000, force_index: bool = False) -> None:
    """Stream candidate generation for the full Source-1 table, writing the
    EXPLODED, scored representation (source1_entity_id, other_entity_id,
    score) to ``out_path`` (tab-separated) in append batches so peak memory is
    bounded by one batch, not the full ~2.2M x ~10.3M join. This file is the
    reusable internal artifact for feature engineering / training / inference;
    ``to_submission_tsv`` derives the official candidate_pairs.tsv from it.

    The blocking index itself is checkpointed per-channel next to
    ``out_path`` (see ``OtherTokenIndex``), keyed by ``out_path``'s basename
    so a train run and a test run — different "others" pools — never share
    an index cache. ``force_index`` rebuilds it from scratch (wired to the
    pipeline's ``--force-blocking`` flag)."""
    n_others = others.height
    log_stage(f"BLOCKING START ({n_others:,} others)")
    log("  sub-stage 1/2: building the blocking token index (one-time, over the full others pool)")
    t_index0 = time.time()
    index_cache_dir = os.path.dirname(out_path) or "."
    index_cache_prefix = os.path.splitext(os.path.basename(out_path))[0]
    index = OtherTokenIndex(others, cache_dir=index_cache_dir, cache_prefix=index_cache_prefix,
                             force=force_index)
    log(f"  built token index over {n_others:,} records in {time.time() - t_index0:.1f}s")

    n = s1.height
    log(f"  sub-stage 2/2: streaming {n:,} Source-1 entities against the index "
        f"(batch_size={batch_size:,})")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tother_entity_id\tscore\n")
    t0 = time.time()
    n_batches = max(1, (n + batch_size - 1) // batch_size)
    for batch_num, start in enumerate(range(0, n, batch_size), 1):
        batch = s1.slice(start, batch_size)
        label = f"streaming batch {batch_num}/{n_batches}"
        result = block_batch_scored(batch, index, top_k, progress_label=label)
        with open(out_path, "a", encoding="utf-8") as f:
            for row in result.iter_rows():
                f.write(f"{row[0]}\t{row[1]}\t{row[2]}\n")
        done = min(start + batch_size, n)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (n - done) / rate if rate > 0 else 0.0
        log(f"  blocked {done}/{n} S1 entities  ({rate:.0f}/s, eta {eta:.0f}s)  {resource_line()}")
    log_stage("BLOCKING DONE")


def to_submission_tsv(scored_path: str, s1_ids: pl.Series, out_path: str,
                       id_column: str = "candidate_entity_ids") -> None:
    """Group the exploded scored-pairs file into the required
    ``source1_entity_id\\t<id_column>`` format, one row per S1 entity (empty
    string for entities with no surviving candidates). ``id_column`` must be
    ``"candidate_entity_ids"`` for candidate_pairs.tsv and
    ``"matched_entity_ids"`` for matching_results.tsv — the challenge
    validator rejects a file whose header doesn't match exactly."""
    scored = pl.scan_csv(scored_path, separator="\t")
    grouped = (
        scored.sort(["source1_entity_id", "score"], descending=[False, True])
        .group_by("source1_entity_id", maintain_order=True)
        .agg(pl.col("other_entity_id").alias("candidates"))
        .with_columns(pl.col("candidates").list.join(",").alias(id_column))
        .select("source1_entity_id", id_column)
        .collect()
    )
    all_s1 = pl.DataFrame({"source1_entity_id": s1_ids})
    result = all_s1.join(grouped, on="source1_entity_id", how="left").with_columns(
        pl.col(id_column).fill_null("")
    )
    # quote_style="never": polars' default quoting wraps an empty string in
    # "" to distinguish it from a null (both would otherwise render as a
    # blank field) — but the challenge format requires a genuinely empty
    # field for singletons, and a literal "" would be parsed as one bogus
    # candidate ID by the validator/scorer. No field here ever contains a
    # tab, newline, or quote (IDs and commas only), so unquoted is safe.
    result.write_csv(out_path, separator="\t", quote_style="never")
