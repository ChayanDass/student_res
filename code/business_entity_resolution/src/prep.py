"""Load raw source TSVs, apply normalization, and cache as Parquet.

Parquet caching matters at this scale: the raw TSVs total ~2GB of text across
~26M rows, and every downstream stage (blocking, feature engineering, training)
re-reads the same columns. Caching the cleaned strings once avoids re-parsing
TSV and re-running Python-level string cleaning on every run.
"""
import os
import time

import polars as pl

from normalize import clean_name, clean_address
from progress import log

CACHE_DIR = os.environ.get("BER_CACHE_DIR", "cache")


def _cache_path(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    return os.path.join(CACHE_DIR, base + ".parquet")


def load_source(path: str, force: bool = False) -> pl.DataFrame:
    """Read one source TSV, clean name/address, return a cached Polars DataFrame
    with columns: entity_id, business_name, business_address, country,
    cleaned_name, cleaned_addr.
    """
    cpath = _cache_path(path)
    if not force and os.path.exists(cpath):
        t0 = time.time()
        df = pl.read_parquet(cpath)
        log(f"  [load_source] CACHE HIT  {path} -> {cpath}  "
            f"({df.height:,} rows, {time.time() - t0:.1f}s)")
        return df

    log(f"  [load_source] CACHE MISS {path} -> building {cpath} "
        f"(reading TSV + normalizing name/address)...")
    t0 = time.time()
    df = pl.read_csv(
        path,
        separator="\t",
        infer_schema_length=0,
        null_values=[""],
    ).with_columns(
        pl.col("business_name").fill_null(""),
        pl.col("business_address").fill_null(""),
        pl.col("country").fill_null(""),
    )
    log(f"  [load_source] read {df.height:,} raw rows in {time.time() - t0:.1f}s, normalizing...")

    t1 = time.time()
    df = df.with_columns(
        pl.col("business_name").map_elements(clean_name, return_dtype=pl.String).alias("cleaned_name"),
        pl.col("business_address").map_elements(clean_address, return_dtype=pl.String).alias("cleaned_addr"),
    )

    os.makedirs(CACHE_DIR, exist_ok=True)
    df.write_parquet(cpath)
    log(f"  [load_source] normalized + cached {df.height:,} rows -> {cpath} "
        f"(normalize {time.time() - t1:.1f}s, total {time.time() - t0:.1f}s)")
    return df
