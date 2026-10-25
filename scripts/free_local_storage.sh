#!/usr/bin/env bash
# Script to safely inspect and reclaim 20+ GB of local disk space
# by removing heavy competition datasets (snapshots, raw graphs, wheels, venv)
# that should be hosted in Google Drive / Google Colab.

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==========================================================="
echo "RepoTopology Local Storage Inspector & Cleaner"
echo "==========================================================="
echo "Checking disk usage in: $REPO_ROOT"
echo ""

if [ -d "data" ]; then
    echo "Disk usage in data/:"
    du -sh data/* 2>/dev/null || du -sh data
    echo ""
fi

if [ -d ".venv" ]; then
    echo "Virtual environment size:"
    du -sh .venv 2>/dev/null
    echo ""
fi

TOTAL_SIZE=$(du -sh . 2>/dev/null | cut -f1)
echo "Total folder size: $TOTAL_SIZE"
echo "==========================================================="
echo ""
echo "Google Colab Note:"
echo "The heavy files (snapshots, embeddings, graphs) are designed"
echo "to live in Google Drive mounted at /content/drive/MyDrive/repology_data"
echo "so your local computer does NOT need to store 21+ GB."
echo ""

CONFIRM="$1"
if [ "$CONFIRM" != "--yes" ]; then
    echo "Options:"
    echo "  1) Delete data/snapshots (~20 GB) only"
    echo "  2) Delete all heavy data/ (snapshots, graphs, embeddings, wheels) (~21 GB)"
    echo "  3) Delete heavy data/ AND .venv (~22 GB)"
    echo "  4) Cancel / Do nothing"
    read -p "Select option (1-4): " CHOICE
else
    CHOICE="1"
fi

case "$CHOICE" in
    1)
        if [ -d "data/snapshots" ]; then
            echo "Removing data/snapshots..."
            rm -rf data/snapshots
            echo "[SUCCESS] Removed data/snapshots! Reclaimed ~20 GB."
        else
            echo "data/snapshots not found."
        fi
        ;;
    2)
        if [ -d "data" ]; then
            echo "Removing heavy data folders (snapshots, embeddings, graphs, wheels)..."
            rm -rf data/snapshots data/embeddings data/graphs data/wheels data/mlx data/sandbox
            echo "[SUCCESS] Removed heavy data! Preserved tasks.jsonl and sft_trajectories.jsonl."
        fi
        ;;
    3)
        if [ -d "data" ]; then
            echo "Removing heavy data folders..."
            rm -rf data/snapshots data/embeddings data/graphs data/wheels data/mlx data/sandbox
        fi
        if [ -d ".venv" ]; then
            echo "Removing .venv..."
            rm -rf .venv
        fi
        echo "[SUCCESS] Reclaimed ~22 GB!"
        ;;
    *)
        echo "Operation cancelled. No files were removed."
        exit 0
        ;;
esac

echo ""
NEW_SIZE=$(du -sh . 2>/dev/null | cut -f1)
echo "New total folder size: $NEW_SIZE"
