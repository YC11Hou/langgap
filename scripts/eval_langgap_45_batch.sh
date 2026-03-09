#!/bin/bash
# scripts/eval_langgap_45_batch.sh

# source /path/to/conda.sh  # Activate conda (adjust path as needed)
conda activate lerobot
export MUJOCO_GL=egl

DATE=0220
CKPT_DIR=${CHECKPOINT_DIR:-./checkpoints}/pi05_langgap_45/checkpoints
EVAL_DIR=$PROJECT_ROOT/eval_results_langgap_45_${DATE}
GPU=1
EVAL_INTERVAL=1  # Evaluate every 1K steps

mkdir -p $EVAL_DIR

while true; do
  STEPS=()
  for d in "$CKPT_DIR"/*/; do
    step=$(basename "$d")
    [[ "$step" == "last" ]] && continue
    [[ -d "$d/pretrained_model" ]] && STEPS+=("$step")
  done
  IFS=$'\n' STEPS=($(sort <<<"${STEPS[*]}")); unset IFS

  FOUND_NEW=false
  for STEP in "${STEPS[@]}"; do
    STEP_NUM=$((10#$STEP / 1000))
    if (( STEP_NUM % EVAL_INTERVAL != 0 )); then
      continue
    fi

    MODEL=${CKPT_DIR}/${STEP}/pretrained_model
    [[ ! -d "$MODEL" ]] && continue
    [[ -f "${EVAL_DIR}/ext5_ckpt${STEP}/results.json" ]] && continue

    FOUND_NEW=true
    echo "=== Checkpoint $STEP ==="

    CUDA_VISIBLE_DEVICES=$GPU python $PROJECT_ROOT/eval/unified_eval.py \
      --model_path $MODEL \
      --task_id 41,44,47,49,50 \
      --episodes 10 \
      --output_dir ${EVAL_DIR}/ext5_ckpt${STEP}
  done

  if [[ "$FOUND_NEW" == "false" ]]; then
    echo "$(date): No new checkpoints. Waiting 300s..."
    sleep 300
  fi
done
