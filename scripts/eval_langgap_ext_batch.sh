#!/bin/bash
# Batch evaluation template - evaluate every 5K steps (to keep up with training)
# 5 episodes x 16 tasks ~ 13min/ckpt, training 5K steps ~ 15min

# source /path/to/conda.sh  # Activate conda (adjust path as needed)
conda activate lerobot
export MUJOCO_GL=egl

DATE=0218
CKPT_DIR=${CHECKPOINT_DIR:-./checkpoints}/pi05_langgap_ext/checkpoints
EVAL_DIR=$PROJECT_ROOT/eval_results_langgap_ext_${DATE}
GPU=1
EVAL_INTERVAL=1  # Evaluate every 1K steps

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
    # Only evaluate at EVAL_INTERVAL multiples (e.g. 5K, 10K, 15K...)
    STEP_NUM=$((10#$STEP / 1000))
    if (( STEP_NUM % EVAL_INTERVAL != 0 )); then
      continue
    fi

    MODEL=${CKPT_DIR}/${STEP}/pretrained_model
    if [[ ! -d "$MODEL" ]]; then
      echo "Skip $STEP (not found)"
      continue
    fi

    # Skip already evaluated checkpoints
    if [[ -f "${EVAL_DIR}/ext16_ckpt${STEP}/results.json" ]]; then
      echo "Skip $STEP (already evaluated)"
      continue
    fi

    FOUND_NEW=true
    echo "=== Checkpoint $STEP ==="

    CUDA_VISIBLE_DEVICES=$GPU python $PROJECT_ROOT/eval/unified_eval.py \
      --model_path $MODEL \
      --task_id 40,41,42,43,44,45,49,50,51,54,78,79,59,62,63,64 \
      --episodes 5 \
      --output_dir ${EVAL_DIR}/ext16_ckpt${STEP}
  done

  if [[ "$FOUND_NEW" == "false" ]]; then
    echo "$(date): No new checkpoints. Waiting 300s..."
    sleep 300
  fi
done
