#!/bin/bash
# Test edge grasp 10 tasks with representative checkpoints
# Runs on GPU 1 to avoid interfering with GPU 0 training

# source /path/to/conda.sh  # Activate conda (adjust path as needed)
conda activate lerobot
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=1

CKPT_DIR="${CHECKPOINT_DIR:-./checkpoints}/pi05_edge_grasp_lora/checkpoints"
EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
TASKS="40,41,42,43,44,45,46,47,49,50"
EPISODES=10

# Representative checkpoints: test from newest to oldest
STEPS="048000 040000 030000 020000 015000"

cd "$EVAL_DIR"

for STEP in $STEPS; do
    MODEL_PATH="$CKPT_DIR/$STEP/pretrained_model"

    if [ ! -d "$MODEL_PATH" ]; then
        echo "$(date): Skipping $STEP - model not found"
        continue
    fi

    echo "============================================"
    echo "$(date): Starting evaluation for checkpoint $STEP"
    echo "============================================"

    python unified_eval.py \
        --model_path "$MODEL_PATH" \
        --suite libero_spatial \
        --type extended \
        --task_id "$TASKS" \
        --episodes "$EPISODES" \
        --save_video \
        --output_dir "results/edge_grasp_${STEP}"

    echo "$(date): Checkpoint $STEP evaluation complete"
    echo ""
done

echo "============================================"
echo "$(date): All evaluations complete!"
echo "============================================"

# Summarize results
echo ""
echo "=== Results Summary ==="
for STEP in $STEPS; do
    RESULT_FILE="results/edge_grasp_${STEP}/results.json"
    if [ -f "$RESULT_FILE" ]; then
        echo "--- $STEP ---"
        python3 -c "
import json
with open('$RESULT_FILE') as f:
    data = json.load(f)
total_success = 0
total_episodes = 0
for task_key, task_data in data.items():
    name = task_data.get('name', task_key)
    sr = task_data.get('success_rate', 0)
    n = task_data.get('n_episodes', 0)
    s = task_data.get('successes', 0)
    total_success += s
    total_episodes += n
    print(f'  {name}: {sr*100:.0f}% ({s}/{n})')
if total_episodes > 0:
    print(f'  Total: {total_success/total_episodes*100:.1f}% ({total_success}/{total_episodes})')
"
    fi
done
