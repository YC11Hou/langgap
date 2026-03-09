#!/bin/bash
# =============================================================================
# Auto-monitor checkpoints and run evaluations
# =============================================================================
set -e

DATE=0215
CKPT_DIR=${CHECKPOINT_DIR:-./checkpoints}/pi05_multispatial_v1/checkpoints
EVAL_DIR=$PROJECT_ROOT/eval_results_multispatial_${DATE}
GPU=1  # Use GPU 1 for evaluation, GPU 0 for training
EVAL_SCRIPT=$PROJECT_ROOT/eval/unified_eval.py

# List of already-tested checkpoints
TESTED_FILE=$PROJECT_ROOT/logs/multispatial_tested_ckpts.txt
touch "$TESTED_FILE"

mkdir -p "$EVAL_DIR"
mkdir -p $PROJECT_ROOT/logs

# source /path/to/conda.sh  # Activate conda (adjust path as needed)
conda activate lerobot
export MUJOCO_GL=egl

echo "========================================"
echo "Checkpoint Monitor & Auto-Evaluation"
echo "========================================"
echo "CKPT_DIR: $CKPT_DIR"
echo "EVAL_DIR: $EVAL_DIR"
echo "GPU: $GPU"
echo "========================================"

while true; do
    # Check if training is still running
    if ! tmux has-session -t multispatial_train 2>/dev/null; then
        echo "[$(date '+%H:%M:%S')] Training session has ended"
    fi

    # Scan all checkpoints (skip symlinks like "last")
    if [ -d "$CKPT_DIR" ]; then
        for STEP_DIR in $(find "$CKPT_DIR" -maxdepth 1 -mindepth 1 -type d ! -type l | sort); do
            STEP=$(basename "$STEP_DIR")
            CKPT_PATH="${STEP_DIR}/pretrained_model"

            # Skip directories without pretrained_model
            if [ ! -d "$CKPT_PATH" ]; then
                continue
            fi

            # Skip already tested
            if grep -q "^${STEP}$" "$TESTED_FILE" 2>/dev/null; then
                continue
            fi

            # Check if model files are complete (adapter_config.json exists means save is done)
            if [ ! -f "${CKPT_PATH}/adapter_config.json" ]; then
                echo "[$(date '+%H:%M:%S')] Checkpoint ${STEP} is still being saved, skipping..."
                continue
            fi

            echo ""
            echo "========================================"
            echo "[$(date '+%H:%M:%S')] Found new checkpoint: ${STEP}"
            echo "========================================"

            # 5a. Test Task 10 (original libero_spatial, benchmark_task_id=0)
            echo "[$(date '+%H:%M:%S')] Testing Task 10 (original)..."
            CUDA_VISIBLE_DEVICES=$GPU python "$EVAL_SCRIPT" \
                --model_path "${CKPT_PATH}" \
                --suite libero_spatial --type original --task_id 0 \
                --episodes 10 --save_video \
                --output_dir "${EVAL_DIR}/task10_ckpt${STEP}" 2>&1 | tee "${EVAL_DIR}/task10_ckpt${STEP}.log"

            # 5b. Test Task 41,44,47,49,50 (extended)
            echo "[$(date '+%H:%M:%S')] Testing Task 41,44,47,49,50 (extended)..."
            CUDA_VISIBLE_DEVICES=$GPU python "$EVAL_SCRIPT" \
                --model_path "${CKPT_PATH}" \
                --task_id 41,44,47,49,50 \
                --episodes 10 --save_video \
                --output_dir "${EVAL_DIR}/ext5_ckpt${STEP}" 2>&1 | tee "${EVAL_DIR}/ext5_ckpt${STEP}.log"

            # Mark as tested
            echo "$STEP" >> "$TESTED_FILE"

            # Print result summary
            echo ""
            echo "========================================"
            echo "===== Checkpoint ${STEP} evaluation complete ====="
            echo "========================================"
            echo "--- Task 10 (original) ---"
            cat "${EVAL_DIR}/task10_ckpt${STEP}/summary.md" 2>/dev/null || echo "(summary not found)"
            echo ""
            echo "--- Extended 5 Tasks ---"
            cat "${EVAL_DIR}/ext5_ckpt${STEP}/summary.md" 2>/dev/null || echo "(summary not found)"
            echo "========================================"
            echo ""
        done
    fi

    # Check every 60 seconds
    sleep 60
done
