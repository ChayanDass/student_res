#!/bin/bash
# EC2 deployment script — run this ONCE after SSH'ing into the instance
# Usage: ssh into instance, then: bash deploy.sh

set -e

echo "=== BER Pipeline Setup on EC2 ==="

# Update system
echo "[1/6] Updating system packages..."
sudo apt-get update -qq
sudo apt-get upgrade -y -qq

# Install Python + dependencies
echo "[2/6] Installing Python 3.11 and build tools..."
sudo apt-get install -y -qq python3.11 python3.11-venv python3-pip build-essential git wget

# Create workspace
echo "[3/6] Creating workspace..."
mkdir -p ~/ber && cd ~/ber

# Clone repo or wait for manual upload
echo "[4/6] Setting up code directory..."
if [ ! -d "student_resource" ]; then
    echo "Note: Waiting for code upload or git clone..."
    # User will either:
    # A) scp -r student_resource/ ubuntu@<IP>:~/ber/
    # B) git clone <repo> student_resource
    echo "Please upload the code now (scp) or provide git repo URL"
    exit 0
fi

cd student_resource/code/business_entity_resolution

# Create venv
echo "[5/6] Creating Python venv..."
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
echo "[6/6] Installing Python packages..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "✓ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. cd ~/ber/student_resource"
echo "  2. source code/business_entity_resolution/venv/bin/activate"
echo "  3. python code/business_entity_resolution/src/train.py"
echo ""
