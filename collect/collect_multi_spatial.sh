#!/bin/bash
# Batch collection for 6 libero_spatial tasks
#
# Usage:
#   ./collect_multi_spatial.sh          # Production: 6x50 episodes
#   ./collect_multi_spatial.sh --debug  # 1 demo per task + video verification
#
# Requires: conda activate lerobot && export MUJOCO_GL=egl

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BDDL_BASE="$(cd "$SCRIPT_DIR/../.." && pwd)/data/bddl_files"
OUTPUT_DIR="$SCRIPT_DIR/../data/source"
DEBUG_DIR="$SCRIPT_DIR/../debug_videos"
REPLAY_SCRIPT="$SCRIPT_DIR/../process/replay_dataset.py"

# Task definitions: task_id script bddl_relative_path
declare -A TASKS
TASKS[10]="scripted_collect_task10.py|libero_spatial/original/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl"
TASKS[41]="scripted_collect_task41.py|libero_spatial/extended/dim1_change_bowl/ext_02_task2_bowl2.bddl"
TASKS[44]="scripted_collect_task44.py|libero_spatial/extended/dim2_change_target/ext_02_task0_to_cabinet.bddl"
TASKS[47]="scripted_collect_task47.py|libero_spatial/extended/dim2_change_target/ext_09_task8_to_stove.bddl"
TASKS[49]="scripted_collect_task49.py|libero_spatial/extended/dim3_change_object/ext_01_ramekin_to_plate.bddl"
TASKS[50]="scripted_collect_task50.py|libero_spatial/extended/dim3_change_object/ext_03_ramekin_to_cabinet.bddl"

# Ordered task IDs
TASK_ORDER=(10 41 44 47 49 50)

# Parse args
DEBUG=false
if [[ "${1:-}" == "--debug" ]]; then
    DEBUG=true
fi

FAILED_TASKS=()

echo "============================================================"
if $DEBUG; then
    echo "  DEBUG MODE: 1 demo per task + video"
    mkdir -p "$DEBUG_DIR"
else
    echo "  PRODUCTION MODE: 50 demos per task"
    mkdir -p "$OUTPUT_DIR"
fi
echo "============================================================"

for task_id in "${TASK_ORDER[@]}"; do
    IFS='|' read -r script bddl <<< "${TASKS[$task_id]}"
    bddl_path="$BDDL_BASE/$bddl"
    script_path="$SCRIPT_DIR/$script"

    echo ""
    echo "========================================"
    echo "  Task $task_id: $bddl"
    echo "========================================"

    if [[ ! -f "$bddl_path" ]]; then
        echo "ERROR: BDDL not found: $bddl_path"
        exit 1
    fi

    if [[ ! -f "$script_path" ]]; then
        echo "ERROR: Script not found: $script_path"
        exit 1
    fi

    if $DEBUG; then
        output_file="/tmp/task_${task_id}_debug.hdf5"
        if python "$script_path" \
            --bddl "$bddl_path" \
            --output "$output_file" \
            --debug; then

            # Generate debug video
            if [[ -f "$output_file" ]]; then
                echo "  Generating debug video..."
                python "$REPLAY_SCRIPT" \
                    --dataset "$output_file" \
                    --bddl "$bddl_path" \
                    --output_dir "$DEBUG_DIR" \
                    --episodes 1
                # Rename output to task-specific name
                generated=$(ls -t "$DEBUG_DIR"/episode_*.mp4 2>/dev/null | head -1)
                if [[ -n "$generated" ]]; then
                    mv "$generated" "$DEBUG_DIR/task${task_id}_debug.mp4"
                    echo "  Video: $DEBUG_DIR/task${task_id}_debug.mp4"
                fi
            fi
        else
            echo "  FAILED: Task $task_id collection failed"
            FAILED_TASKS+=("$task_id")
        fi
    else
        output_file="$OUTPUT_DIR/task_${task_id}.hdf5"
        if python "$script_path" \
            --bddl "$bddl_path" \
            --output "$output_file" \
            --num 200 \
            --target 50; then

            if [[ -f "$output_file" ]]; then
                echo "  Saved: $output_file"
            else
                echo "  WARNING: No output for task $task_id"
                FAILED_TASKS+=("$task_id")
            fi
        else
            echo "  FAILED: Task $task_id collection failed"
            FAILED_TASKS+=("$task_id")
        fi
    fi
done

echo ""
echo "============================================================"
if [[ ${#FAILED_TASKS[@]} -eq 0 ]]; then
    echo "  All tasks complete!"
else
    echo "  Done with ${#FAILED_TASKS[@]} failure(s): ${FAILED_TASKS[*]}"
fi
if $DEBUG; then
    echo "  Debug videos: $DEBUG_DIR/"
    ls -la "$DEBUG_DIR"/task*_debug.mp4 2>/dev/null || echo "  (no videos generated)"
else
    echo "  Output files: $OUTPUT_DIR/"
    ls -la "$OUTPUT_DIR"/task_*.hdf5 2>/dev/null || echo "  (no files generated)"
fi
echo "============================================================"

# Exit with error if any task failed
if [[ ${#FAILED_TASKS[@]} -gt 0 ]]; then
    exit 1
fi
