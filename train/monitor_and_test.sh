#!/bin/bash
# Monitor training progress and automatically test every 1K checkpoint with video
# Usage: tmux new-session -d -s task10_eval "bash train/monitor_and_test.sh"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}/pi05_task10_scripted_lora/checkpoints"
EVAL_DIR="$PROJECT_ROOT/eval"
LOG_FILE="$PROJECT_ROOT/logs/task10_eval.log"
TESTED_FILE="/tmp/tested_task10_scripted.txt"
RESULT_DIR="$PROJECT_ROOT/eval_results_task10_0213"

# source /path/to/conda.sh  # Activate conda (adjust path as needed)
conda activate lerobot
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=1

touch "$TESTED_FILE"

echo "$(date): ========================================" >> "$LOG_FILE"
echo "$(date): Task10 scripted auto-monitor testing started" >> "$LOG_FILE"
echo "$(date): Checkpoint dir: $CHECKPOINT_DIR" >> "$LOG_FILE"
echo "$(date): ========================================" >> "$LOG_FILE"

while true; do
    # Check if the training process is alive
    if ! pgrep -f "pi05_task10_scripted_lora" > /dev/null; then
        echo "$(date): Warning - training process has stopped!" >> "$LOG_FILE"
    fi

    # Check for new checkpoints
    if [ -d "$CHECKPOINT_DIR" ]; then
        for ckpt in $(ls -d ${CHECKPOINT_DIR}/0* 2>/dev/null | sort -V); do
            ckpt_name=$(basename "$ckpt")

            # Skip already tested
            if grep -q "$ckpt_name" "$TESTED_FILE" 2>/dev/null; then
                continue
            fi

            # Check for complete model files (LoRA uses adapter_model.safetensors)
            if [ -f "$ckpt/pretrained_model/adapter_model.safetensors" ] || [ -f "$ckpt/pretrained_model/model.safetensors" ]; then
                echo "$(date): Found new checkpoint $ckpt_name, starting test..." >> "$LOG_FILE"

                OUTPUT="$RESULT_DIR/checkpoint_${ckpt_name}"
                TEST_LOG="/tmp/test_task10_${ckpt_name}.log"

                # Run test (10 episodes + save video)
                # Task 10 was collected from libero_spatial scene, corresponding to task_id=0
                cd "$EVAL_DIR"
                python unified_eval.py \
                    --model_path "$ckpt/pretrained_model" \
                    --suite libero_spatial \
                    --type original \
                    --task_id 0 \
                    --episodes 10 \
                    --save_video \
                    --output_dir "$OUTPUT" \
                    > "$TEST_LOG" 2>&1

                # Extract success rate
                success_rate=$(grep -oP "Success rate: \K[0-9.]+" "$TEST_LOG" | tail -1)
                if [ -z "$success_rate" ]; then
                    # Try extracting from summary.md
                    success_rate=$(grep -oP "\d+\.\d+%" "$OUTPUT/summary.md" 2>/dev/null | head -1)
                fi
                echo "$(date): $ckpt_name test complete, success rate: ${success_rate:-N/A}" >> "$LOG_FILE"

                # Mark as tested
                echo "$ckpt_name" >> "$TESTED_FILE"
            fi
        done
    fi

    # Check every 3 minutes
    sleep 180
done
