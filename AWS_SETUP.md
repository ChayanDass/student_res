# AWS Setup for Business Entity Resolution Training

## Quick Start (3 steps, 5 minutes)

### Step 1: Launch EC2 Instance

Go to **AWS Console** → **EC2** → **Launch Instances**

**Configuration:**
- **Name:** `ber-training` (or any name)
- **AMI:** Ubuntu 24.04 LTS (search for `ubuntu/images/hvm-ssd/ubuntu-noble-24.04-amd64-server`)
- **Instance Type:** `m6i.2xlarge` ← **KEY CHOICE** (8 vCPU, 32 GB RAM, ~$24/day)
- **Key Pair:** Create or select your SSH key
- **Security Group:** 
  - Inbound SSH (port 22) from your IP / 0.0.0.0/0
- **Storage:** 100 GB gp3 (default, sufficient)
- **Launch!**

**Cost Estimate:** 
- Instance: $0.34/hr × 24 × 3 = ~$72
- Storage (100 GB, 3 days): ~$3–5
- Data transfer: ~$2–5
- **Total: ~$80–85** (safe within $100 budget)

---

### Step 2: SSH & Set Up Environment

Once instance is running, copy its **Public IPv4** address and:

```bash
# Open SSH connection
ssh -i /path/to/your-key.pem ubuntu@<PUBLIC_IP>

# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Python + build tools
sudo apt-get install -y python3.12 python3.12-venv python3-pip build-essential

# Create working directory
mkdir -p ~/ber && cd ~/ber

# Option A: If you have git access to your repo
git clone <YOUR_REPO_URL> .

# Option B: If uploading files manually
# Use scp from your local machine:
# scp -r -i key.pem ./student_resource/* ubuntu@<IP>:~/ber/
```

---

### Step 3: Run Pipeline

```bash
cd ~/ber/code/business_entity_resolution

# Create venv
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Go back to student_resource directory
cd ../../../

# ===== TRAINING (main entrypoint) =====
# This will:
#   1. Load all train sources (takes ~3 min)
#   2. Run blocking over all 2.2M S1 entities (takes ~45 min)
#   3. Engineer features (takes ~1.5 hours)
#   4. Train LightGBM (takes ~10 min)
#   5. Tune threshold on validation set (takes ~5 min)
# Total: ~2.5 hours

python code/business_entity_resolution/src/train.py

# ===== INFERENCE (produces submission files) =====
# This will:
#   1. Block the 1.7M test entities (takes ~30 min)
#   2. Score all candidates with the trained model (takes ~1 hour)
#   3. Write output/matching_results.tsv and output/candidate_pairs.tsv
# Total: ~1.5 hours

python code/business_entity_resolution/src/predict.py

# ===== VALIDATE OUTPUT =====
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test

# Should print: "PASS (exit 0)"
```

---

## Monitoring & Logs

While running, you can monitor progress:

```bash
# In a separate SSH session, watch the output files grow
watch -n 5 'wc -l output/*.tsv cache/*.tsv'

# Monitor memory/CPU
top
# Press 'q' to quit

# Monitor disk space
df -h
```

---

## After Training: Get Results

```bash
# Copy outputs back to your local machine
# (Run this on YOUR LOCAL machine, not on EC2)

scp -i key.pem -r ubuntu@<IP>:~/ber/output ~/local-results/

# Check the files
ls ~/local-results/
# Should see: matching_results.tsv, candidate_pairs.tsv
```

---

## Cost Control: STOP the Instance When Done!

**CRITICAL:** Turn off the instance when training finishes to avoid wasting credits.

```bash
# Option 1: Via AWS Console
# EC2 Dashboard → Instances → Select instance → Instance State → Stop

# Option 2: Via CLI
aws ec2 stop-instances --instance-ids i-0123456789abcdef --region us-east-1
```

A **stopped** instance costs $0.10/GB/month for storage only (~$3/month for 100 GB), NOT the $0.34/hr compute cost.

To resume: **Instance State → Start**

---

## Troubleshooting

**Q: "Connection timed out" when SSHing**
- Check security group allows inbound SSH
- Check instance is in "running" state
- Try waiting 30 seconds after launch (takes time to initialize)

**Q: Out of memory error**
- m6i.2xlarge has 32 GB; should be plenty
- If blocking fails, the `run_blocking()` function streams in 100k-record batches (memory-bounded)
- Check no other processes are running (`top`)

**Q: Training is slower than expected**
- Check CPU usage (`top`) — should be ~300-500% (multi-core usage)
- If stuck at one stage, check logs in the terminal

**Q: How do I know training is done?**
- Script will print "saved model to models/lgbm_model.txt" (training done)
- Then it will print "wrote output/candidate_pairs.tsv" (inference done)
- Then "wrote output/matching_results.tsv" (final results ready)

---

## Instance Specifications (Reference)

| Component | Spec |
|-----------|------|
| vCPU | 8 |
| Memory | 32 GB |
| Network | Up to 12.5 Gbps |
| Storage | 100 GB gp3 SSD |
| OS | Ubuntu 24.04 LTS |
| Hourly Cost | $0.34 (on-demand, us-east-1) |

This is enough for:
- Loading 2.2M + 10.3M entity records into RAM
- Parallel feature computation (8 cores)
- LightGBM training with 10M+ candidate pairs

---

## After You Get Results

1. **Validate** with the provided script (should print `PASS`)
2. **Download** the output TSVs to your local machine
3. **Upload** `matching_results.tsv` to the challenge leaderboard
4. **Create** the final submission zip (see Documentation_template.md)
5. **Stop** the EC2 instance to save credits

---

## Questions?

- AWS EC2 pricing: https://aws.amazon.com/ec2/pricing/on-demand/
- Instance types: https://aws.amazon.com/ec2/instance-types/m6i/
- Ubuntu AMI finder: https://cloud-images.ubuntu.com/locator/ec2/
