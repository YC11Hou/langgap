#!/bin/bash
# Batch evaluation template - evaluate every 4K steps (matching 56-task evaluation speed)
# 56 tasks: (40+16) x 5 eps = 280 eps/ckpt ~ 45min
# Training 4K steps ~ 60min -> EVAL_INTERVAL=4 leaves headroom

# source /path/to/conda.sh  # Activate conda (adjust path as needed)
conda activate lerobot
export MUJOCO_GL=egl

DATE=0218
CKPT_DIR=${CHECKPOINT_DIR:-./checkpoints}/pi05_langgap_full/checkpoints
EVAL_DIR=$PROJECT_ROOT/eval_results_langgap_full_${DATE}
GPU=0
EVAL_INTERVAL=4  # Evaluate every 4K steps (56-task eval is 3.5x heavier than 16-task)

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
    # Only evaluate at EVAL_INTERVAL multiples (4K, 8K, 12K...)
    STEP_NUM=$((10#$STEP / 1000))
    if (( STEP_NUM % EVAL_INTERVAL != 0 )); then
      continue
    fi

    MODEL=${CKPT_DIR}/${STEP}/pretrained_model
    if [[ ! -d "$MODEL" ]]; then
      echo "Skip $STEP (not found)"
      continue
    fi

    # Skip already evaluated checkpoints (both rounds must be complete)
    if [[ -f "${EVAL_DIR}/official40_ckpt${STEP}/results.json" && -f "${EVAL_DIR}/ext16_ckpt${STEP}/results.json" ]]; then
      echo "Skip $STEP (already evaluated)"
      continue
    fi

    FOUND_NEW=true
    echo "=== Checkpoint $STEP ==="

    # Test official 40 tasks
    if [[ ! -f "${EVAL_DIR}/official40_ckpt${STEP}/results.json" ]]; then
      CUDA_VISIBLE_DEVICES=$GPU python $PROJECT_ROOT/eval/unified_eval.py \
        --model_path $MODEL \
        --task_id 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39 \
        --episodes 5 \
        --output_dir ${EVAL_DIR}/official40_ckpt${STEP}
    fi

    # Test extended 16 tasks
    if [[ ! -f "${EVAL_DIR}/ext16_ckpt${STEP}/results.json" ]]; then
      CUDA_VISIBLE_DEVICES=$GPU python $PROJECT_ROOT/eval/unified_eval.py \
        --model_path $MODEL \
        --task_id 40,41,42,43,44,45,49,50,51,54,78,79,59,62,63,64 \
        --episodes 5 \
        --output_dir ${EVAL_DIR}/ext16_ckpt${STEP}
    fi
  done

  if [[ "$FOUND_NEW" == "false" ]]; then
    echo "$(date): No new checkpoints. Waiting 300s..."
    sleep 300
  fi
done
