#!/bin/bash

# Configuration parameters
GPU_ID=3
PORT=2427
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
MAX_WAIT=500   # Maximum wait time in seconds
SLEEP_INTERVAL=20  # Interval between checks in seconds
LOG_FILE="vllm_server.log"

# Start the vLLM server
echo "Starting vLLM server on GPU $GPU_ID with port $PORT..."
CUDA_VISIBLE_DEVICES=$GPU_ID nohup python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --max_model_len 5000 \
    --tensor_parallel_size 1 \
    --dtype bfloat16 \
    --max-num-seqs 512 \
    --gpu_memory_utilization 0.7 \
    --port $PORT > $LOG_FILE 2>&1 &

# Wait for the vLLM server to be ready
echo "Waiting for vLLM server to be ready on port $PORT..."
elapsed=0

while ! nc -z localhost $PORT; do
    sleep $SLEEP_INTERVAL
    elapsed=$((elapsed + SLEEP_INTERVAL))
    echo "Still waiting for vLLM server... (${elapsed}s elapsed)"
    if [ $elapsed -ge $MAX_WAIT ]; then
        echo "[ERROR] vLLM server failed to start within ${MAX_WAIT}s. Check $LOG_FILE for details."
        exit 1
    fi
done

echo "✅ vLLM server is up after ${elapsed}s."

# Set GPU environment variable
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Run the generate script
echo "Starting generate.py with model $MODEL_NAME..."
python run_src/generate.py \
    --dataset_name MATH \
    --if_use_cards True \
    --test_json_filename test_all \
    --model_ckpt $MODEL_NAME \
    --attribute_type condition \
    --api gpt3.5-turbo

echo "✅ Generate execution completed."

# Kill the vLLM server after completion
echo "Shutting down vLLM server..."
kill $(pgrep -f "vllm.entrypoints.openai.api_server")
echo "✅ vLLM server stopped."
echo "✅ Script execution completed."