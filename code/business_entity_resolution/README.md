# Business Entity Resolution Pipeline

End-to-end ML solution for matching business entities across 3 independent data sources using token-based blocking and LightGBM classification.

## Architecture

**Blocking (candidate generation):** Multi-channel token-based inverted-index blocking across:
- Business name tokens (legal suffixes removed)
- Address place-name tokens
- Address digit tokens (street numbers / PIN codes)  
- Character 5-grams of business names (survives typos)

Recall ceiling: ~97% on validation data with top_k=100.

**Features (18 total):**
- String similarity: Levenshtein, token-sort, partial ratios via RapidFuzz
- Set similarity: Jaccard over tokens and char-ngrams
- Address features (with nulls handled separately)
- Digit/street-number matching
- Country match, name length ratio, etc.

**Model:** LightGBM binary classifier with early stopping on validation logloss, threshold tuned on F_0.5 (precision-heavy metric from the challenge).

## Running on AWS

### 1. Launch EC2 Instance

**Recommended:** `m6i.2xlarge` (8 vCPU, 32GB RAM, ~$24/day on-demand, stays within $100 budget)

```bash
# Via AWS Console or CLI:
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type m6i.2xlarge \
  --key-name YOUR_KEY_PAIR \
  --security-group-ids sg-YOUR_SECURITY_GROUP \
  --root-volume-size 100 \
  --region us-east-1
```

**Or use the AWS Console:**
- EC2 Dashboard → Launch Instances
- AMI: Ubuntu 24.04 LTS (ami-0c55b159cbfafe1f0)
- Instance Type: m6i.2xlarge
- Storage: 100 GB (gp3, default)
- Security Group: Allow SSH (port 22)

### 2. Connect & Set Up

```bash
ssh -i YOUR_KEY.pem ubuntu@<instance-ip>

# Clone/download your code (adjust path as needed)
git clone <your-repo> ber_project
cd ber_project/code/business_entity_resolution

# Create venv and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Download dataset (if not already in the repo)
# Adjust paths/download method as needed
```

### 3. Run Full Pipeline

```bash
cd /path/to/student_resource
source code/business_entity_resolution/venv/bin/activate

# Training (full dataset, ~2.5-3 hours on m6i.2xlarge)
python code/business_entity_resolution/src/train.py

# Inference on test set (~1-1.5 hours)
python code/business_entity_resolution/src/predict.py

# Validate output
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test

echo "✓ Check output/matching_results.tsv and output/candidate_pairs.tsv"
```

### 4. (Optional) Smoke Test

Quick sanity check on tiny subset (5-10 min instead of 3+ hours):

```bash
python code/business_entity_resolution/src/train.py \
  --sample-frac 0.004 \
  --others-sample-n 200000 \
  --model-dir models_smoke \
  --cache-dir cache_smoke
```

## File Structure

```
src/
  normalize.py       — Text cleaning, tokenization, leet-speak undo
  prep.py            — Load TSVs, cache as Parquet
  blocking.py        — Token-based candidate generation
  features.py        — Pairwise similarity features (18 features)
  pairs.py           — Join pairs with entity attributes
  train.py           — Train LightGBM + tune threshold (F_0.5)
  predict.py         — Score test candidates, produce submission TSVs
  evaluate.py        — F_0.5 scoring (challenge metric)
  build_cache.py     — Precompute normalized data (optional, for production)
```

## Key Hyperparameters

Edit `train.py` to tune (or pass as `--arg`):

- `--top-k`: Max blocking candidates per Source-1 entity (default 100; tradeoff recall vs candidate set size)
- `--neg-cap`: Max negative training examples per S1 entity (default 15; imbalance control)
- `--val-frac`: Hold-out fraction for threshold selection (default 0.15)

LightGBM params are in `train.py` `params` dict:
- `num_leaves=63`: model capacity
- `learning_rate=0.05`: convergence speed
- `min_data_in_leaf=50`: regularization

## Expected Performance

- **Blocking recall:** ~97% (97 of 100 true matches captured before model sees them)
- **Validation F_0.5:** ~0.94–0.96 on held-out training data
- **Training time:** ~2–3 hours (full ~2.2M × 10.3M corpus, m6i.2xlarge)
- **Inference time:** ~1–1.5 hours (full ~1.7M test entities)

## Cost Estimate ($100 budget, 3 days)

| Instance | vCPU | RAM | Price/hr | 3 days | Headroom |
|----------|------|-----|----------|--------|----------|
| m6i.2xlarge | 8 | 32 GB | $0.34 | ~$72 | ~$28 (storage, data xfer) |
| c6i.4xlarge | 16 | 32 GB | $0.68 | ~$144 | ❌ over budget |
| r6i.2xlarge | 8 | 64 GB | $0.50 | ~$108 | ❌ tight |

**m6i.2xlarge is the sweet spot.** Stop the instance when done to avoid overage.

## Troubleshooting

**Memory error during blocking/features:** Reduce `--feature-batch-size` (default 300k) in `predict.py` or `--block-batch-size` in `train.py`.

**Slow feature computation:** Increase `--n-jobs` (default auto-detects CPU count) to use more cores.

**Validation score seems low:** Check that ground truth and blocking are working — run smoke test first.

## References

- **Challenge:** Unstop ML Challenge 2026, Business Entity Resolution
- **Metric:** F_0.5 macro (precision-heavy) over Source-1 entities
- **Data:** ~2.2M deduplicated S1, ~5M noisy S2, ~5M noisy S3 (train); ~1.7M S1, ~4.9M S2, ~5.1M S3 (test, includes unseen country "France")
