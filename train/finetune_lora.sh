#!/bin/bash
# =============================================================================
# LoRA fine-tuning for π0.5 - supports single-task/multi-task training
# =============================================================================
#
# Usage:
#   bash finetune_lora.sh --model=<pretrained_model> [--task=<task_ids>] [options]
#
# Required arguments:
#   --model=<str>        - Pretrained model path
#
# Optional arguments:
#   --task=<str>         - Task ID (10, 10,50, 0-39, all); if not specified, uses all episodes in the dataset
#   --dataset=<str>      - Dataset repo_id, default <YOUR_HF_USERNAME>/langgap_full
#   --lora_r=<int>       - LoRA rank, default 8
#   --lr=<float>         - Learning rate, default 2.5e-05
#   --batch_size=<int>   - Batch size, default 4
#   --dtype=<str>        - Precision, default bfloat16
#   --steps=<int>        - Training steps, default 200000
#   --grad_ckpt=<bool>   - Gradient checkpointing, default True
#   --save_freq=<int>    - Checkpoint save frequency, default 1000
#   --output_dir=<str>   - Checkpoint save path, default auto-generated
#
# Examples:
#   bash finetune_lora.sh --task=10 --model=lerobot/pi05_libero_base
#   bash finetune_lora.sh --dataset=<YOUR_HF_USERNAME>/task50_ext --model=lerobot/pi05_libero_base  # all episodes
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAPPING_FILE="$SCRIPT_DIR/task_episodes.txt"

# Defaults
TASK_IDS_INPUT=""
PRETRAINED_MODEL=""
DATASET_REPO="<YOUR_HF_USERNAME>/langgap_full"
LORA_R=8
LR="2.5e-05"
BATCH_SIZE=4
DTYPE="bfloat16"
STEPS=200000
GRAD_CKPT="True"
EVAL_FREQ=1000
SAVE_FREQ=1000
OUTPUT_DIR_CUSTOM=""

# Parse arguments
for arg in "$@"; do
    case $arg in
        --task=*)
            TASK_IDS_INPUT="${arg#*=}"
            ;;
        --model=*)
            PRETRAINED_MODEL="${arg#*=}"
            ;;
        --dataset=*)
            DATASET_REPO="${arg#*=}"
            ;;
        --lora_r=*)
            LORA_R="${arg#*=}"
            ;;
        --lr=*)
            LR="${arg#*=}"
            ;;
        --batch_size=*)
            BATCH_SIZE="${arg#*=}"
            ;;
        --dtype=*)
            DTYPE="${arg#*=}"
            ;;
        --steps=*)
            STEPS="${arg#*=}"
            ;;
        --grad_ckpt=*)
            GRAD_CKPT="${arg#*=}"
            ;;
        --save_freq=*)
            SAVE_FREQ="${arg#*=}"
            ;;
        --output_dir=*)
            OUTPUT_DIR_CUSTOM="${arg#*=}"
            ;;
        --*)
            echo "Unknown argument: $arg"
            exit 1
            ;;
        *)
            echo "Unknown argument: $arg (please use --key=value format)"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$PRETRAINED_MODEL" ]; then
    echo "Usage: bash finetune_lora.sh --model=<pretrained_model> [--task=<task_ids>] [options]"
    echo ""
    echo "Required arguments:"
    echo "  --model=<str>        - Pretrained model path"
    echo ""
    echo "Optional arguments:"
    echo "  --task=<str>         - Task ID (10, 10,50, 0-39, all); if not specified, uses all episodes in the dataset"
    echo "  --dataset=<str>      - Dataset repo_id, default <YOUR_HF_USERNAME>/langgap_full"
    echo "  --lora_r=<int>       - LoRA rank, default 8"
    echo "  --lr=<float>         - Learning rate, default 2.5e-05"
    echo "  --batch_size=<int>   - Batch size, default 4"
    echo "  --dtype=<str>        - Precision, default bfloat16"
    echo "  --steps=<int>        - Training steps, default 200000"
    echo "  --grad_ckpt=<bool>   - Gradient checkpointing, default True"
    echo "  --save_freq=<int>    - Checkpoint save frequency, default 1000"
    echo "  --output_dir=<str>   - Checkpoint save path, default auto-generated"
    echo ""
    echo "Examples:"
    echo "  bash finetune_lora.sh --task=10 --model=lerobot/pi05_libero_base"
    echo "  bash finetune_lora.sh --dataset=<YOUR_HF_USERNAME>/task50_ext --model=lerobot/pi05_libero_base  # all episodes"
    exit 1
fi

# Handle episode filtering
if [ -z "$TASK_IDS_INPUT" ]; then
    # No --task specified, use all episodes in the dataset
    ALL_EPISODES=""
    TOTAL_EPISODES="all"
    TASK_COUNT="all"
    # Extract output directory name from dataset name
    OUTPUT_NAME=$(echo "$DATASET_REPO" | sed 's/.*\///' | sed 's/_ext$//')
else
    # Parse task_ids
    parse_task_ids() {
        local input=$1
        local result=""

        if [ "$input" = "all" ]; then
            result=$(seq 0 62 | tr '\n' ',' | sed 's/,$//')
        elif [[ "$input" =~ ^[0-9]+-[0-9]+$ ]]; then
            local start=$(echo "$input" | cut -d'-' -f1)
            local end=$(echo "$input" | cut -d'-' -f2)
            result=$(seq $start $end | tr '\n' ',' | sed 's/,$//')
        else
            result=$input
        fi

        echo "$result"
    }

    TASK_IDS=$(parse_task_ids "$TASK_IDS_INPUT")

    # Collect all episodes
    ALL_EPISODES=""
    TOTAL_EPISODES=0
    TASK_COUNT=0

    if [ "$DATASET_REPO" = "<YOUR_HF_USERNAME>/edge_grasp" ]; then
        # edge_grasp dataset: inline mapping table
        # Format: num_episodes|start_episode|end_episode
        declare -A EDGE_GRASP_MAP=(
            [40]="52|0|51"
            [41]="50|52|101"
            [42]="55|102|156"
            [43]="50|157|206"
            [44]="52|207|258"
            [45]="52|259|310"
            [46]="50|311|360"
            [47]="51|361|411"
            [49]="52|412|463"
            [50]="50|464|513"
        )

        IFS=',' read -ra TASK_ARRAY <<< "$TASK_IDS"
        for TASK_ID in "${TASK_ARRAY[@]}"; do
            if [ -z "${EDGE_GRASP_MAP[$TASK_ID]+x}" ]; then
                echo "Error: task $TASK_ID not found in edge_grasp dataset"
                exit 1
            fi
            TASK_LINE="${EDGE_GRASP_MAP[$TASK_ID]}"
            NUM_EP=$(echo "$TASK_LINE" | cut -d'|' -f1)
            EP_START=$(echo "$TASK_LINE" | cut -d'|' -f2)
            EP_END=$(echo "$TASK_LINE" | cut -d'|' -f3)
            EPISODES=$(seq $EP_START $EP_END | tr '\n' ',' | sed 's/,$//')

            if [ -z "$ALL_EPISODES" ]; then
                ALL_EPISODES="$EPISODES"
            else
                ALL_EPISODES="$ALL_EPISODES,$EPISODES"
            fi

            TOTAL_EPISODES=$((TOTAL_EPISODES + NUM_EP))
            TASK_COUNT=$((TASK_COUNT + 1))
        done
    else
        # Original logic: look up from task_episodes.txt
        if [ ! -f "$MAPPING_FILE" ]; then
            echo "Error: mapping file not found: $MAPPING_FILE"
            exit 1
        fi

        IFS=',' read -ra TASK_ARRAY <<< "$TASK_IDS"
        for TASK_ID in "${TASK_ARRAY[@]}"; do
            TASK_LINE=$(grep "^${TASK_ID}|" "$MAPPING_FILE")
            if [ -z "$TASK_LINE" ]; then
                echo "Error: task $TASK_ID not found"
                exit 1
            fi

            NUM_EP=$(echo "$TASK_LINE" | cut -d'|' -f2)
            EPISODES=$(echo "$TASK_LINE" | cut -d'|' -f3)

            if [ -z "$ALL_EPISODES" ]; then
                ALL_EPISODES="$EPISODES"
            else
                ALL_EPISODES="$ALL_EPISODES,$EPISODES"
            fi

            TOTAL_EPISODES=$((TOTAL_EPISODES + NUM_EP))
            TASK_COUNT=$((TASK_COUNT + 1))
        done
    fi

    # Set output directory name
    if [ "$TASK_IDS_INPUT" = "all" ]; then
        OUTPUT_NAME="all"
    elif [[ "$TASK_IDS_INPUT" =~ ^[0-9]+-[0-9]+$ ]]; then
        OUTPUT_NAME="range_${TASK_IDS_INPUT//-/_}"
    elif [[ "$TASK_IDS_INPUT" =~ , ]]; then
        OUTPUT_NAME="multi_${TASK_IDS_INPUT//,/_}"
    else
        OUTPUT_NAME="task${TASK_IDS_INPUT}"
    fi
fi

# Set output directory (custom or default)
if [ -n "$OUTPUT_DIR_CUSTOM" ]; then
    OUTPUT_DIR="$OUTPUT_DIR_CUSTOM"
else
    OUTPUT_DIR="${CHECKPOINT_DIR:-./checkpoints}/pi05_${OUTPUT_NAME}_lora"
fi
OUTPUT_REPO="<YOUR_HF_USERNAME>/pi05_${OUTPUT_NAME}_lora"

# source /path/to/conda.sh  # Activate conda (adjust path as needed)
conda activate lerobot

export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export HF_HOME=${HF_HOME:-~/.cache/huggingface}

echo "========================================"
echo "π0.5 LoRA Training"
echo "========================================"
if [ -z "$TASK_IDS_INPUT" ]; then
    echo "Tasks: all (using all episodes in the dataset)"
else
    echo "Tasks: $TASK_IDS_INPUT ($TASK_COUNT tasks)"
fi
echo "Training data: $TOTAL_EPISODES episodes"
echo "Dataset: $DATASET_REPO"
echo "Pretrained model: $PRETRAINED_MODEL"
echo "Output directory: $OUTPUT_DIR"
echo "----------------------------------------"
echo "LoRA R: $LORA_R"
echo "Learning rate: $LR"
echo "Batch Size: $BATCH_SIZE"
echo "Precision: $DTYPE"
echo "Training steps: $STEPS"
echo "Gradient Checkpointing: $GRAD_CKPT"
echo "Save frequency: $SAVE_FREQ"
echo "========================================"

cd "$SCRIPT_DIR/../lerobot"

# Build episodes argument
if [ -z "$ALL_EPISODES" ]; then
    EPISODES_ARG=""
else
    EPISODES_ARG="--dataset.episodes=[${ALL_EPISODES}]"
fi

python -u -c "
import sys
args = ['lerobot-train',
    '--policy.type=pi05',
    '--policy.pretrained_path=${PRETRAINED_MODEL}',
    '--policy.gradient_checkpointing=${GRAD_CKPT}',
    '--policy.use_amp=True',
    '--policy.dtype=${DTYPE}',
    '--policy.optimizer_lr=${LR}',
    '--peft.method_type=LORA',
    '--peft.r=${LORA_R}',
    '--dataset.repo_id=${DATASET_REPO}',
    '--dataset.video_backend=pyav',
    '--output_dir=${OUTPUT_DIR}',
    '--batch_size=${BATCH_SIZE}',
    '--steps=${STEPS}',
    '--save_freq=${SAVE_FREQ}',
    '--policy.push_to_hub=false',
    '--num_workers=8',
    '--seed=42',
]
if '${EPISODES_ARG}':
    args.append('${EPISODES_ARG}')
sys.argv = args
from lerobot.scripts.lerobot_train import main
main()
"

echo "Training complete!"
