"""Build the Parquet cache for every raw source file, in parallel (one process
per file — cheap parallelism since each file is independent and the cleaning
step is Python-callback-bound / holds the GIL within a single process)."""
import sys
import time
from multiprocessing import Pool

from prep import load_source

FILES = [
    "dataset/train/train_source1.tsv",
    "dataset/train/train_source2.tsv",
    "dataset/train/train_source3.tsv",
    "dataset/test/test_source1.tsv",
    "dataset/test/test_source2.tsv",
    "dataset/test/test_source3.tsv",
]


def _build(path):
    t0 = time.time()
    df = load_source(path, force=False)
    return path, df.height, time.time() - t0


if __name__ == "__main__":
    with Pool(6) as pool:
        for path, n, dt in pool.imap_unordered(_build, FILES):
            print(f"{path}: {n} rows in {dt:.1f}s", flush=True)
