#!/bin/bash
# Batch replay langgap_ext (16 extended tasks) and official tasks (2 examples)
# Replay 2 episodes per task

set -e

cd $PROJECT_ROOT
# source /path/to/conda.sh  # Activate conda (adjust path as needed)
conda activate lerobot
export MUJOCO_GL=egl

# Output directory
OUTPUT_BASE=$PROJECT_ROOT/replay_videos
mkdir -p $OUTPUT_BASE

# Source HDF5 directory
HDF5_BASE=$PROJECT_ROOT/data/source/langgap_ext

# BDDL directory
BDDL_BASE=$PROJECT_ROOT/data/bddl_files

# 16 extended task mappings: task_id -> (suite, dim, bddl)
declare -A TASK_INFO
TASK_INFO[40]="libero_spatial/extended/dim1_change_bowl:ext_01_task0_bowl2"
TASK_INFO[41]="libero_spatial/extended/dim1_change_bowl:ext_02_task2_bowl2"
TASK_INFO[42]="libero_spatial/extended/dim1_change_bowl:ext_04_task4_bowl2"
TASK_INFO[43]="libero_spatial/extended/dim2_change_target:ext_01_task0_to_stove"
TASK_INFO[44]="libero_spatial/extended/dim2_change_target:ext_02_task0_to_cabinet"
TASK_INFO[45]="libero_spatial/extended/dim2_change_target:ext_04_task2_to_ramekin"
TASK_INFO[49]="libero_spatial/extended/dim3_change_object:ext_01_ramekin_to_plate"
TASK_INFO[50]="libero_spatial/extended/dim3_change_object:ext_03_ramekin_to_cabinet"
TASK_INFO[51]="libero_spatial/extended/dim3_change_object:ext_05_cookie_box_to_plate"
TASK_INFO[54]="libero_goal/extended/dim1_change_object:ext_03_cream_cheese_to_stove"
TASK_INFO[59]="libero_object/extended/dim1_change_object:ext_03_scene1_alphabet_soup"
TASK_INFO[62]="libero_object/extended/dim1_change_object:ext_09_scene4_bbq_sauce"
TASK_INFO[63]="libero_object/extended/dim1_change_object:ext_06_scene2_tomato_sauce"
TASK_INFO[64]="libero_object/extended/dim1_change_object:ext_05_scene2_milk"
TASK_INFO[78]="libero_goal/extended/dim1_change_object:ext_05_cream_cheese_to_cabinet"
TASK_INFO[79]="libero_goal/extended/dim1_change_object:ext_07_cream_cheese_to_plate"

# Task index order in the dataset (langgap_ext task_index 0-15)
TASK_ORDER=(40 41 42 43 44 45 49 50 51 54 78 79 59 62 63 64)

echo "=== Starting replay of langgap_ext (16 extended tasks) ==="
echo "Replaying 2 episodes per task"
echo ""

for i in "${!TASK_ORDER[@]}"; do
    TASK=${TASK_ORDER[$i]}
    INFO=${TASK_INFO[$TASK]}
    DIR_PATH=${INFO%:*}
    BDDL_NAME=${INFO#*:}

    BDDL=$BDDL_BASE/$DIR_PATH/${BDDL_NAME}.bddl
    HDF5=$HDF5_BASE/task_${TASK}.hdf5

    echo "=== Task $TASK (index $i): $BDDL_NAME ==="

    if [ ! -f "$HDF5" ]; then
        echo "  SKIP: HDF5 not found: $HDF5"
        continue
    fi

    if [ ! -f "$BDDL" ]; then
        echo "  SKIP: BDDL not found: $BDDL"
        continue
    fi

    python process/replay_dataset.py \
        --dataset <YOUR_HF_USERNAME>/langgap_ext \
        --bddl "$BDDL" \
        --source_hdf5 "$HDF5" \
        --task_index $i \
        --episodes 2 \
        --output_dir $OUTPUT_BASE/ext_task_${TASK}

    echo ""
done

echo "=== Done! Videos saved to $OUTPUT_BASE ==="
ls -la $OUTPUT_BASE/*/
