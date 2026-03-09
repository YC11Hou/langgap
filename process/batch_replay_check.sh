#!/bin/bash
# Batch replay check for edge grasp tasks
# Outputs per-episode SUCCESS/FAIL to JSON

BDDL_BASE=$PROJECT_ROOT/data/bddl_files/libero_spatial/extended
HDF5_BASE=$PROJECT_ROOT/data/source/edge_grasp
DATASET=<YOUR_HF_USERNAME>/edge_grasp
OUTPUT_BASE=$PROJECT_ROOT/replay_results

mkdir -p $OUTPUT_BASE

# Task -> BDDL mapping
declare -A TASK_BDDL
TASK_BDDL[40]="dim1_change_bowl/ext_01_task0_bowl2.bddl"
TASK_BDDL[41]="dim1_change_bowl/ext_02_task2_bowl2.bddl"
TASK_BDDL[42]="dim1_change_bowl/ext_04_task4_bowl2.bddl"
TASK_BDDL[43]="dim2_change_target/ext_01_task0_to_stove.bddl"
TASK_BDDL[44]="dim2_change_target/ext_02_task0_to_cabinet.bddl"
TASK_BDDL[45]="dim2_change_target/ext_04_task2_to_ramekin.bddl"
TASK_BDDL[46]="dim2_change_target/ext_07_task7_to_cabinet.bddl"
TASK_BDDL[47]="dim2_change_target/ext_09_task8_to_stove.bddl"
TASK_BDDL[49]="dim3_change_object/ext_01_ramekin_to_plate.bddl"
TASK_BDDL[50]="dim3_change_object/ext_03_ramekin_to_cabinet.bddl"

# Task index in merged dataset (0-based relative position)
declare -A TASK_INDEX
TASK_INDEX[40]=0
TASK_INDEX[41]=1
TASK_INDEX[42]=2
TASK_INDEX[43]=3
TASK_INDEX[44]=4
TASK_INDEX[45]=5
TASK_INDEX[46]=6
TASK_INDEX[47]=7
TASK_INDEX[49]=8
TASK_INDEX[50]=9

run_task() {
    TASK=$1
    echo "=== Task $TASK ==="
    BDDL=$BDDL_BASE/${TASK_BDDL[$TASK]}
    HDF5=$HDF5_BASE/task_${TASK}.hdf5
    IDX=${TASK_INDEX[$TASK]}

    cd $PROJECT_ROOT
    python process/replay_with_results.py \
        --dataset $DATASET \
        --bddl $BDDL \
        --source_hdf5 $HDF5 \
        --task_index $IDX \
        --output_json $OUTPUT_BASE/task_${TASK}.json \
        --frame_skip 1
}

# Run specified task or all
if [ -n "$1" ]; then
    run_task $1
else
    for TASK in 40 41 42 43 44 45 46 47 49 50; do
        run_task $TASK
    done
fi
