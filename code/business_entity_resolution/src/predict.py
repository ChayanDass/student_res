"""Run the trained pipeline over the test set, producing the two required
submission files in ``--output-dir``:

  * ``candidate_pairs.tsv``  — every blocking candidate (last stage before the
    model scores them), derived directly from the blocking output.
  * ``matching_results.tsv`` — candidates the trained model scored at or above
    the tuned threshold.

Candidate pairs are scored in fixed-size chunks (``--feature-batch-size``) via
a streaming CSV reader so peak memory is bounded regardless of how many total
candidate pairs the full ~1.7M-entity test Source-1 table produces.

Usage (from ``student_resource/``, after ``train.py`` has produced a model)::

    python code/business_entity_resolution/src/predict.py
"""
import argparse
import json
import os
import sys
import time
import traceback

import lightgbm as lgb
import polars as pl

from blocking import run_blocking, to_submission_tsv
from features import compute_features_parallel
from pairs import build_pair_frame
from prep import load_source
from progress import log, log_stage, resource_line, system_info_banner, Steps, auto_n_jobs, scale_to_ram


def score_candidates_to_predictions(scored_path: str, s1: pl.DataFrame, others: pl.DataFrame,
                                     booster: lgb.Booster, threshold: float, out_path: str,
                                     n_jobs: int, feature_batch_size: int) -> None:
    """Stream through ``scored_path`` (source1_entity_id, other_entity_id,
    score), score every pair with the model, and append pairs with
    prob >= threshold to ``out_path`` as (source1_entity_id, other_entity_id,
    score=prob) — the same shape ``to_submission_tsv`` expects."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tother_entity_id\tscore\n")

    reader = pl.read_csv_batched(scored_path, separator="\t", batch_size=feature_batch_size)
    n_scored = 0
    n_kept = 0
    t0 = time.time()
    batches = reader.next_batches(1)
    while batches:
        chunk = batches[0]
        n_scored += chunk.height
        pf = build_pair_frame(chunk, s1, others)
        if pf.height > 0:
            X = compute_features_parallel(pf, n_jobs=n_jobs)
            probs = booster.predict(X)
            keep = probs >= threshold
            if keep.any():
                out = pl.DataFrame({
                    "source1_entity_id": pf["source1_entity_id"].to_numpy()[keep],
                    "other_entity_id": pf["other_entity_id"].to_numpy()[keep],
                    "score": probs[keep],
                })
                with open(out_path, "a", encoding="utf-8") as f:
                    for row in out.iter_rows():
                        f.write(f"{row[0]}\t{row[1]}\t{row[2]:.6f}\n")
                n_kept += int(keep.sum())
        elapsed = time.time() - t0
        rate = n_scored / elapsed if elapsed > 0 else 0.0
        log(f"  scored {n_scored} candidate pairs, kept {n_kept} matches  ({rate:.0f}/s)  {resource_line()}")
        batches = reader.next_batches(1)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="dataset/test")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--top-k", type=int, default=None,
                     help="override blocking top_k; default reuses the value used at training time")
    ap.add_argument("--block-batch-size", type=int, default=None,
                     help="S1 rows per blocking batch; default auto-scales with system RAM (70%% budget)")
    ap.add_argument("--feature-batch-size", type=int, default=None,
                     help="candidate pairs scored per chunk; default auto-scales with system RAM "
                          "(70%% budget) — lower this manually if you still hit memory limits")
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
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    if args.mem_fraction:
        from progress import set_ram_fraction
        set_ram_fraction(args.mem_fraction)

    n_jobs = args.n_jobs or auto_n_jobs()
    block_batch_size = args.block_batch_size or scale_to_ram(100_000)
    feature_batch_size = args.feature_batch_size or scale_to_ram(300_000)

    log_stage("PREDICT START")
    system_info_banner({
        "n_jobs": n_jobs,
        "block_batch_size": f"{block_batch_size:,}",
        "feature_batch_size": f"{feature_batch_size:,}",
    })
    log(f"args: {vars(args)}")
    steps = Steps(4)

    steps.start("Load model & test data")
    model_path = os.path.join(args.model_dir, "lgbm_model.txt")
    config_path = os.path.join(args.model_dir, "model_config.json")
    if not os.path.exists(model_path) or not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Model not found ({model_path} / {config_path}). Run train.py first."
        )

    with open(config_path) as f:
        config = json.load(f)
    top_k = args.top_k or config["top_k"]
    threshold = config["threshold"]
    log(f"loaded model config: top_k={top_k} threshold={threshold:.3f} "
        f"validation_f05={config.get('validation_f05', float('nan')):.4f}")

    booster = lgb.Booster(model_file=model_path)

    t0 = time.time()
    s1 = load_source(os.path.join(args.test_dir, "test_source1.tsv"))
    s2 = load_source(os.path.join(args.test_dir, "test_source2.tsv"))
    s3 = load_source(os.path.join(args.test_dir, "test_source3.tsv"))
    others = pl.concat([s2, s3])
    log(f"loaded test sources in {time.time()-t0:.1f}s: s1={s1.height} others={others.height}  {resource_line()}")

    steps.start("Blocking (candidate generation)")
    scored_path = os.path.join(args.cache_dir, "test_candidates_scored.tsv")
    if args.force_blocking or not os.path.exists(scored_path):
        t0 = time.time()
        run_blocking(s1, others, scored_path, top_k=top_k, batch_size=block_batch_size,
                     force_index=args.force_blocking)
        log(f"blocking done in {time.time()-t0:.1f}s")
    else:
        log(f"reusing cached blocking output: {scored_path}")

    steps.start("Write candidate_pairs.tsv")
    candidate_out = os.path.join(args.output_dir, "candidate_pairs.tsv")
    to_submission_tsv(scored_path, s1["entity_id"], candidate_out)
    log(f"wrote {candidate_out}  ({os.path.getsize(candidate_out)} bytes)")

    steps.start("Score candidates & write matching_results.tsv")
    predicted_path = os.path.join(args.cache_dir, "test_predicted_pairs.tsv")
    t0 = time.time()
    score_candidates_to_predictions(
        scored_path, s1, others, booster, threshold, predicted_path,
        n_jobs=n_jobs, feature_batch_size=feature_batch_size,
    )
    log(f"scoring done in {time.time()-t0:.1f}s")

    matching_out = os.path.join(args.output_dir, "matching_results.tsv")
    to_submission_tsv(predicted_path, s1["entity_id"], matching_out, id_column="matched_entity_ids")
    log(f"wrote {matching_out}  ({os.path.getsize(matching_out)} bytes)")
    log_stage("PREDICT DONE")


def main():
    args = parse_args()
    try:
        run(args)
    except Exception:
        log("FATAL: inference failed with an exception:")
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()
