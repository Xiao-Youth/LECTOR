#!/bin/bash
# ============================================================
# LECTOR Evaluation Script
# ============================================================

set -e

# --- Configuration (modify these) ---
DATA_FILE="./data/NC_Physics_test.jsonl"
OUTPUT_DIR="./evaluation_results"
MAX_SAMPLES=100

# Model can be a local path or served via vLLM/OpenAI-compatible API
MODEL="./outputs/lector/LECTOR-Qwen3-4B/checkpoint"
API_KEY="test"
BASE_URL="http://127.0.0.1:8000/v1"

MAX_WORKERS=32
MAX_TOKENS=8192
TEMPERATURE=0.7
REWARD_TYPES="graph_quality writing writing_llm"
RUN_MODE="2step"

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_file) DATA_FILE="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --max_samples) MAX_SAMPLES="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --api_key) API_KEY="$2"; shift 2 ;;
        --base_url) BASE_URL="$2"; shift 2 ;;
        --max_workers) MAX_WORKERS="$2"; shift 2 ;;
        --max_tokens) MAX_TOKENS="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --reward_types) REWARD_TYPES="$2"; shift 2 ;;
        --run_mode) RUN_MODE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "  --data_file       Path to test jsonl data"
            echo "  --output_dir      Output directory"
            echo "  --max_samples     Max samples to evaluate"
            echo "  --model           Model path or name"
            echo "  --api_key         API key for model endpoint"
            echo "  --base_url        OpenAI-compatible API base URL"
            echo "  --max_workers     Parallel workers"
            echo "  --max_tokens      Max generation tokens"
            echo "  --temperature     Generation temperature"
            echo "  --reward_types    Reward types (graph_quality writing writing_llm)"
            echo "  --run_mode        Mode: 2step (default)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

python evaluation/evaluation_pipeline.py \
    --data_file "$DATA_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --max_samples $MAX_SAMPLES \
    --model "$MODEL" \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --max_workers $MAX_WORKERS \
    --max_tokens $MAX_TOKENS \
    --temperature $TEMPERATURE \
    --reward_types $REWARD_TYPES \
    --parallel_inference \
    --run_mode $RUN_MODE \
    --eval
