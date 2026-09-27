"""Train the matching-model stage end-to-end:

  train_source1/2/3.tsv --[blocking]--> candidate pairs
                          --[features]--> labeled feature matrix
                          --[LightGBM]--> classifier + tuned threshold

Splits Source-1 entities (not S2/S3 records) into a train/validation set so
validation F_0.5 is a fair estimate of held-out performance. Saves the model,
feature list, and chosen threshold to ``--model-dir`` (default ``models/``).

Usage (from the ``student_resource/`` directory, with the venv from
``requirements.txt`` active)::

    python code/business_entity_resolution/src/train.py

Key flags — see ``--help``. ``--sample-frac`` runs a quick end-to-end check on
a random subset of Source-1 entities (useful to validate the pipeline before
committing to a full run); omit it for the real, full-scale training run.
"""
import argparse
import json
import os
import random
import sys
import time
import traceback

import lightgbm as lgb
import numpy as np
import polars as pl

from blocking import run_blocking
from features import compute_features_parallel, FEATURE_COLUMNS
from pairs import build_pair_frame
from prep import load_source
from evaluate import f_beta_macro
from progress import (
    log, log_stage, resource_line, system_info_banner, Steps,
    auto_n_jobs, scale_to_ram,
)


def parse_ground_truth(path: str) -> dict:
    gt = pl.read_csv(path, separator="\t", infer_schema_length=0, null_values=[""])
    truth = {}
    for s1, ids in gt.iter_rows():
        truth[s1] = set(ids.split(",")) if ids else set()
    return truth


def gt_pairs_frame(truth_map: dict) -> pl.DataFrame:
    s1s, others = [], []
    for s1, ids in truth_map.items():
        for oid in ids:
            s1s.append(s1)
            others.append(oid)
    return pl.DataFrame({"source1_entity_id": s1s, "other_entity_id": others, "label": [1] * len(s1s)})


def split_ids(ids, val_frac: float, seed: int):
    ids = list(ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = int(len(ids) * val_frac)
    return set(ids[n_val:]), set(ids[:n_val])


# Reference count of distinct S1 entities processed per cap_negatives chunk
# (scaled to RAM like every other chunk size in this pipeline, see
# progress.scale_to_ram). sort()+cum_count().over() ranks per
# source1_entity_id, so chunking by disjoint entity-id batches and ranking
# each chunk independently gives byte-identical results to ranking the whole
# set at once (same reasoning as blocking.py's S1_JOIN_SUBCHUNK_ROWS) — but
# bounds each eager sort to one chunk's rows instead of the full ~181M-row
# uncapped negative set. A single sort over that full set is what OOM-killed
# a run even after the read of scored_path itself was made lazy/streaming:
# the two-pass join beforehand preserves row count, but a global eager sort
# needs the whole (sortable) dataset resident at once, with no chunk
# boundary to bound it.
CAP_NEG_ID_CHUNK = 50_000


def cap_negatives(labeled: pl.DataFrame, neg_cap: int, s1: pl.DataFrame, others: pl.DataFrame) -> pl.DataFrame:
    """Keep the ``neg_cap`` hardest negatives per S1 entity: primarily the
    blocking score (how many channels agreed — coarse, so it ties constantly),
    tie-broken by name-length closeness as a free, fully-vectorized proxy for
    name similarity. Hard-negative mining (training on near-miss non-matches
    rather than random ones) is a well-established entity-matching technique
    for sharpening the decision boundary; a per-row text-similarity score
    would be a stronger hardness signal but requires the full feature pass,
    which is exactly what capping *before* feature engineering exists to
    avoid at this scale, so the tie-break has to stay a native polars op
    (no map_elements) to keep this step cheap over hundreds of millions of
    candidate pairs."""
    pos = labeled.filter(pl.col("label") == 1)
    neg = labeled.filter(pl.col("label") == 0)

    s1_len = s1.select(
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("cleaned_name").str.len_chars().alias("s1_len"),
    )
    o_len = others.select(
        pl.col("entity_id").alias("other_entity_id"),
        pl.col("cleaned_name").str.len_chars().alias("o_len"),
    )

    chunk_ids = neg["source1_entity_id"].unique().to_list()
    chunk_size = scale_to_ram(CAP_NEG_ID_CHUNK)
    n_chunks = max(1, (len(chunk_ids) + chunk_size - 1) // chunk_size)
    capped_parts = []
    for i in range(0, len(chunk_ids), chunk_size):
        id_chunk = chunk_ids[i:i + chunk_size]
        sub = (
            neg.filter(pl.col("source1_entity_id").is_in(id_chunk))
               .join(s1_len, on="source1_entity_id", how="left")
               .join(o_len, on="other_entity_id", how="left")
               .with_columns((pl.col("s1_len") - pl.col("o_len")).abs().fill_null(999).alias("len_diff"))
               .sort(["source1_entity_id", "score", "len_diff"], descending=[False, True, False])
               .with_columns(pl.col("score").cum_count().over("source1_entity_id").alias("rk"))
               .filter(pl.col("rk") <= neg_cap)
               .drop("rk", "s1_len", "o_len", "len_diff")
        )
        capped_parts.append(sub)
        done = i // chunk_size + 1
        if done == n_chunks or done % max(1, n_chunks // 10) == 0:
            log(f"    cap_negatives: chunk {done}/{n_chunks}  {resource_line()}")

    neg_capped = pl.concat(capped_parts) if capped_parts else neg.clear()
    return pl.concat([pos, neg_capped])


def build_xy(labeled: pl.DataFrame, s1: pl.DataFrame, others: pl.DataFrame, n_jobs: int):
    pf = build_pair_frame(labeled, s1, others)
    X = compute_features_parallel(pf, n_jobs=n_jobs)
    y = pf["label"].to_numpy()
    s1_ids = pf["source1_entity_id"].to_list()
    other_ids = pf["other_entity_id"].to_list()
    return X, y, s1_ids, other_ids


def tune_threshold(probs: np.ndarray, s1_ids, other_ids, truth_map: dict, val_ids: set) -> tuple:
    by_s1 = {}
    for p, s1, oid in zip(probs, s1_ids, other_ids):
        by_s1.setdefault(s1, []).append((p, oid))

    best_t, best_f = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        pred_map = {
            s1: {oid for p, oid in items if p >= t}
            for s1, items in by_s1.items()
        }
        truth_val = {s1: truth_map[s1] for s1 in val_ids}
        score = f_beta_macro(pred_map, truth_val, beta=0.5)
        if score > best_f:
            best_f, best_t = score, float(t)
    return best_t, best_f


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="dataset/train")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--top-k", type=int, default=100, help="max blocking candidates per S1 entity")
    ap.add_argument("--block-batch-size", type=int, default=None,
                     help="S1 rows per blocking batch; default auto-scales with system RAM (70%% budget)")
    ap.add_argument("--neg-cap", type=int, default=15, help="max negative candidates kept per S1 entity for TRAINING only (validation always uses the full blocked candidate set)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-frac", type=float, default=None,
                     help="optional: subsample S1 entities to this fraction for a quick end-to-end "
                          "validation run before committing to the full-scale run. Omit for the real run.")
    ap.add_argument("--others-sample-n", type=int, default=None,
                     help="optional (used together with --sample-frac): subsample the S2+S3 pool to this "
                          "many records (plus every ground-truth match for the sampled S1 entities, so "
                          "recall stays measurable). Omit for the real run.")
    ap.add_argument("--n-jobs", type=int, default=None,
                     help="worker processes for feature engineering; default auto-scales with CPU cores "
                          "and the RAM budget (see progress.auto_n_jobs)")
    ap.add_argument("--mem-fraction", type=float, default=None,
                     help="override the fraction of system RAM the pipeline plans around (default 0.7 / "
                          "70%%, or $BER_RAM_FRACTION)")
    ap.add_argument("--force-blocking", action="store_true")
    return ap.parse_args()


def run(args) -> None:
    os.environ["BER_CACHE_DIR"] = args.cache_dir
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    if args.mem_fraction:
        from progress import set_ram_fraction
        set_ram_fraction(args.mem_fraction)

    n_jobs = args.n_jobs or auto_n_jobs()
    block_batch_size = args.block_batch_size or scale_to_ram(100_000)

    log_stage("TRAIN START")
    system_info_banner({
        "n_jobs": n_jobs,
        "block_batch_size": f"{block_batch_size:,}",
        "top_k": args.top_k,
    })
    log(f"args: {vars(args)}")
    steps = Steps(6)

    steps.start("Load & normalize source data")
    t0 = time.time()
    s1 = load_source(os.path.join(args.train_dir, "train_source1.tsv"))
    s2 = load_source(os.path.join(args.train_dir, "train_source2.tsv"))
    s3 = load_source(os.path.join(args.train_dir, "train_source3.tsv"))
    truth_map = parse_ground_truth(os.path.join(args.train_dir, "train_ground_truth.tsv"))
    log(f"loaded sources in {time.time()-t0:.1f}s: s1={s1.height} s2={s2.height} s3={s3.height}  {resource_line()}")

    if args.sample_frac:
        s1 = s1.sample(fraction=args.sample_frac, seed=args.seed)
        truth_map = {k: v for k, v in truth_map.items() if k in set(s1["entity_id"].to_list())}

    others = pl.concat([s2, s3])
    if args.others_sample_n:
        true_ids = set()
        for ids in truth_map.values():
            true_ids |= ids
        keep = others.filter(pl.col("entity_id").is_in(true_ids))
        rest = others.filter(~pl.col("entity_id").is_in(true_ids)).sample(
            n=min(args.others_sample_n, others.height), seed=args.seed)
        others = pl.concat([keep, rest]).unique(subset=["entity_id"])
        log(f"subsampled others pool to {others.height} records (--others-sample-n)")

    steps.start("Blocking (candidate generation)")
    scored_path = os.path.join(args.cache_dir, "train_candidates_scored.tsv")
    if args.force_blocking or not os.path.exists(scored_path):
        t0 = time.time()
        run_blocking(s1, others, scored_path, top_k=args.top_k, batch_size=block_batch_size,
                     force_index=args.force_blocking)
        log(f"blocking done in {time.time()-t0:.1f}s")
    else:
        log(f"reusing cached blocking output: {scored_path}")

    steps.start("Feature engineering")
    gt_pairs = gt_pairs_frame(truth_map)
    # scored_path is the full blocking output — 220M+ rows / several GB at
    # full scale. scan_csv keeps it as a lazy query plan (no eager read), and
    # each split below is collected with the streaming engine one at a time,
    # so peak memory holds one split, not an eager "labeled" copy of the
    # whole file plus a train copy plus a val copy simultaneously — that
    # triple-materialization is what actually drove RSS past 30GB and
    # triggered the OOM-kill on entry to this step (same lazy+streaming
    # pattern already applied in blocking.py's index build/join).
    scored_lazy = pl.scan_csv(scored_path, separator="\t").join(
        gt_pairs.lazy().select("source1_entity_id", "other_entity_id", "label"),
        on=["source1_entity_id", "other_entity_id"], how="left",
    ).with_columns(pl.col("label").fill_null(0))

    train_ids, val_ids = split_ids(s1["entity_id"].to_list(), args.val_frac, args.seed)

    log(f"  materializing TRAIN split... {resource_line()}")
    train_labeled = scored_lazy.filter(
        pl.col("source1_entity_id").is_in(list(train_ids))
    ).collect(engine="streaming")

    n_pos = train_labeled.filter(pl.col("label") == 1).height
    n_neg = train_labeled.height - n_pos
    log(f"train candidate pairs: {train_labeled.height} ({n_pos} positive, {n_neg} negative)  {resource_line()}")
    train_labeled = cap_negatives(train_labeled, args.neg_cap, s1, others)
    log(f"after negative capping (neg_cap={args.neg_cap}): {train_labeled.height} rows  {resource_line()}")

    t0 = time.time()
    log(f"building TRAIN feature matrix... (n_jobs={n_jobs}, parallel)")
    X_train, y_train, _, _ = build_xy(train_labeled, s1, others, n_jobs)
    log(f"  train features: {X_train.shape}  {resource_line()}")
    del train_labeled  # free the (pre-cap, uncapped-scale) train split before materializing val

    log(f"  materializing VAL split... {resource_line()}")
    val_labeled = scored_lazy.filter(
        pl.col("source1_entity_id").is_in(list(val_ids))
    ).collect(engine="streaming")
    log(f"building VAL feature matrix... (n_jobs={n_jobs}, parallel)")
    X_val, y_val, val_s1_ids, val_other_ids = build_xy(val_labeled, s1, others, n_jobs)
    log(f"  val features: {X_val.shape}  {resource_line()}")
    del val_labeled
    log(f"feature engineering done in {time.time()-t0:.1f}s")

    steps.start("Model training (LightGBM)")
    # carve a small slice out of TRAIN (not VAL) purely for early stopping, so
    # the validation set stays untouched for threshold selection / reporting.
    n = X_train.shape[0]
    idx = np.random.RandomState(args.seed).permutation(n)
    # clamp so a small train set (e.g. a --sample-frac validation run) always
    # leaves at least one row for the actual fit set, not just the ES slice
    n_es = min(max(1000, int(0.1 * n)), max(1, n - 1))
    es_idx, fit_idx = idx[:n_es], idx[n_es:]

    train_set = lgb.Dataset(X_train[fit_idx], label=y_train[fit_idx], feature_name=FEATURE_COLUMNS)
    es_set = lgb.Dataset(X_train[es_idx], label=y_train[es_idx], feature_name=FEATURE_COLUMNS, reference=train_set)

    params = dict(
        objective="binary",
        metric="binary_logloss",
        num_leaves=63,
        learning_rate=0.05,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_data_in_leaf=50,
        verbose=-1,
    )
    t0 = time.time()
    booster = lgb.train(
        params, train_set, num_boost_round=1000,
        valid_sets=[es_set], valid_names=["early_stop"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
    )
    log(f"LightGBM training done in {time.time()-t0:.1f}s  best_iteration={booster.best_iteration}  {resource_line()}")

    steps.start("Threshold tuning & save")
    val_probs = booster.predict(X_val, num_iteration=booster.best_iteration)
    threshold, val_f05 = tune_threshold(val_probs, val_s1_ids, val_other_ids, truth_map, val_ids)
    log(f"chosen threshold={threshold:.2f}  validation F_0.5 (macro)={val_f05:.4f}")

    model_path = os.path.join(args.model_dir, "lgbm_model.txt")
    booster.save_model(model_path)
    config_path = os.path.join(args.model_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "threshold": threshold,
            "validation_f05": val_f05,
            "feature_columns": FEATURE_COLUMNS,
            "top_k": args.top_k,
        }, f, indent=2)
    log(f"saved model -> {model_path}")
    log(f"saved config -> {config_path}")
    log_stage("TRAIN DONE")


def main():
    args = parse_args()
    try:
        run(args)
    except Exception:
        log("FATAL: training failed with an exception:")
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()
