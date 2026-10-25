#!/usr/bin/env bash
# Quick setup script for Google Colab runtimes
set -e

echo "=== RepoTopology Google Colab Environment Setup ==="

# Check Python version
python3 --version

# Upgrade pip
python3 -m pip install -q -U pip

# Install repo with Colab and Dev dependencies
echo "Installing RepoTopology and dependencies..."
python3 -m pip install -q -e ".[colab,dev]"

# Hardware verification
echo "=== GPU Status ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "No NVIDIA GPU detected. Running in CPU mode."
fi

echo "=== Setup complete! ==="
