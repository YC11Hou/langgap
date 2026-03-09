#!/bin/bash
# Batch collection for 16 extended tasks (spatial 9 + goal 3 + object 4)
#
# Usage:
#   ./collect_multi.sh --debug --episodes 5 --output_dir debug_videos/
#   ./collect_multi.sh --episodes 150 --output_dir ./data/source/
#
# Requires: conda activate lerobot && export MUJOCO_GL=egl

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BDDL_BASE="$(cd "$SCRIPT_DIR/../.." && pwd)/data/bddl_files"
REPLAY_SCRIPT="$SCRIPT_DIR/../process/replay_dataset.py"

# ---- Defaults ----
DEBUG=false
EPISODES=150
OUTPUT_DIR="$SCRIPT_DIR/../data/source"

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)
            DEBUG=true
            shift
            ;;
        --episodes)
            EPISODES="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown arg: $1"
            echo "Usage: $0 [--debug] [--episodes N] [--output_dir DIR]"
            exit 1
            ;;
    esac
done

# ---- Task definitions ----
# Format: task_id|script|bddl_relative_path
# 16 extended tasks (spatial 9 + goal 3 + object 4)
TASK_DEFS=(
    # === libero_spatial (9 tasks) ===
    "40|scripted_collect_task40.py|libero_spatial/extended/dim1_change_bowl/ext_01_task0_bowl2.bddl"
    "41|scripted_collect_task41.py|libero_spatial/extended/dim1_change_bowl/ext_02_task2_bowl2.bddl"
    "42|scripted_collect_task42.py|libero_spatial/extended/dim1_change_bowl/ext_04_task4_bowl2.bddl"
    "43|scripted_collect_task43.py|libero_spatial/extended/dim2_change_target/ext_01_task0_to_stove.bddl"
    "44|scripted_collect_task44.py|libero_spatial/extended/dim2_change_target/ext_02_task0_to_cabinet.bddl"
    "45|scripted_collect_task45.py|libero_spatial/extended/dim2_change_target/ext_04_task2_to_ramekin.bddl"
    "49|scripted_collect_task49.py|libero_spatial/extended/dim3_change_object/ext_01_ramekin_to_plate.bddl"
    "50|scripted_collect_task50.py|libero_spatial/extended/dim3_change_object/ext_03_ramekin_to_cabinet.bddl"
    "51|scripted_collect_task51.py|libero_spatial/extended/dim3_change_object/ext_05_cookie_box_to_plate.bddl"
    # === libero_goal (3 tasks) ===
    "54|scripted_collect_task54.py|libero_goal/extended/dim1_change_object/ext_03_cream_cheese_to_stove.bddl"
    "78|scripted_collect_task78.py|libero_goal/extended/dim1_change_object/ext_05_cream_cheese_to_cabinet.bddl"
    "79|scripted_collect_task79.py|libero_goal/extended/dim1_change_object/ext_07_cream_cheese_to_plate.bddl"
    # === libero_object (4 tasks) ===
    "59|scripted_collect_task59.py|libero_object/extended/dim1_change_object/ext_03_scene1_alphabet_soup.bddl"
    "62|scripted_collect_task62.py|libero_object/extended/dim1_change_object/ext_09_scene4_bbq_sauce.bddl"
    "63|scripted_collect_task63.py|libero_object/extended/dim1_change_object/ext_06_scene2_tomato_sauce.bddl"
    "64|scripted_collect_task64.py|libero_object/extended/dim1_change_object/ext_05_scene2_milk.bddl"
)

FAILED_TASKS=()
COMPLETED=0
TOTAL=${#TASK_DEFS[@]}

echo "============================================================"
if $DEBUG; then
    echo "  DEBUG MODE: $EPISODES episodes per task + video replay"
else
    echo "  PRODUCTION MODE: $EPISODES episodes per task"
fi
echo "  Output: $OUTPUT_DIR"
echo "  Tasks: $TOTAL"
echo "============================================================"

mkdir -p "$OUTPUT_DIR"

for task_def in "${TASK_DEFS[@]}"; do
    IFS='|' read -r task_id script bddl <<< "$task_def"
    bddl_path="$BDDL_BASE/$bddl"
    script_path="$SCRIPT_DIR/$script"

    echo ""
    echo "========================================"
    echo "  Task $task_id ($((COMPLETED + 1))/$TOTAL)"
    echo "  BDDL: $bddl"
    echo "========================================"

    if [[ ! -f "$bddl_path" ]]; then
        echo "ERROR: BDDL not found: $bddl_path"
        FAILED_TASKS+=("$task_id")
        COMPLETED=$((COMPLETED + 1))
        continue
    fi

    if [[ ! -f "$script_path" ]]; then
        echo "ERROR: Script not found: $script_path"
        FAILED_TASKS+=("$task_id")
        COMPLETED=$((COMPLETED + 1))
        continue
    fi

    if $DEBUG; then
        # Debug mode: collect to per-task subdirectory
        task_dir="$OUTPUT_DIR/task${task_id}"
        mkdir -p "$task_dir"
        output_file="$task_dir/task_${task_id}.hdf5"

        if python "$script_path" \
            --bddl "$bddl_path" \
            --output "$output_file" \
            --num $((EPISODES * 4)) \
            --target "$EPISODES"; then

            # Generate replay videos for each episode
            if [[ -f "$output_file" ]]; then
                echo "  Generating replay videos..."
                python "$REPLAY_SCRIPT" \
                    --dataset "$output_file" \
                    --bddl "$bddl_path" \
                    --output_dir "$task_dir" \
                    --episodes "$EPISODES" 2>&1 | tail -1
                echo "  Videos: $task_dir/"
            fi
        else
            echo "  FAILED: Task $task_id collection failed"
            FAILED_TASKS+=("$task_id")
        fi
    else
        # Production mode
        output_file="$OUTPUT_DIR/task_${task_id}.hdf5"

        if python "$script_path" \
            --bddl "$bddl_path" \
            --output "$output_file" \
            --num $((EPISODES * 4)) \
            --target "$EPISODES"; then

            if [[ -f "$output_file" ]]; then
                # Count demos in HDF5
                n_demos=$(python -c "import h5py; f=h5py.File('$output_file','r'); print(f['data'].attrs['total'])" 2>/dev/null || echo "?")
                echo "  Saved: $output_file ($n_demos demos)"
            else
                echo "  WARNING: No output for task $task_id"
                FAILED_TASKS+=("$task_id")
            fi
        else
            echo "  FAILED: Task $task_id collection failed"
            FAILED_TASKS+=("$task_id")
        fi
    fi

    COMPLETED=$((COMPLETED + 1))
done

echo ""
echo "============================================================"
echo "  Summary: $((TOTAL - ${#FAILED_TASKS[@]}))/$TOTAL tasks succeeded"
if [[ ${#FAILED_TASKS[@]} -gt 0 ]]; then
    echo "  Failed tasks: ${FAILED_TASKS[*]}"
fi
echo "  Output: $OUTPUT_DIR/"
if $DEBUG; then
    echo ""
    echo "  Debug videos per task:"
    for task_def in "${TASK_DEFS[@]}"; do
        IFS='|' read -r task_id _ _ <<< "$task_def"
        n_vids=$(ls "$OUTPUT_DIR/task${task_id}"/episode_*.mp4 2>/dev/null | wc -l)
        if [[ $n_vids -gt 0 ]]; then
            echo "    task${task_id}/: $n_vids videos"
        fi
    done
fi
echo "============================================================"

# Exit with error if any task failed
if [[ ${#FAILED_TASKS[@]} -gt 0 ]]; then
    exit 1
fi
