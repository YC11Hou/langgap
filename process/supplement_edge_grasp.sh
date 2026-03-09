#!/bin/bash
# Supplement edge_grasp demos: collect + verify + append all verified demos.
#
# Collects extra demos (oversampled by historical success rate), verifies them,
# and appends only passed demos to existing HDF5 files.
#
# Usage:
#     cd $PROJECT_ROOT
#     conda activate lerobot
#     export MUJOCO_GL=egl
#     bash process/supplement_edge_grasp.sh

set -e

SUPPLEMENT_DIR=/tmp/edge_grasp_supplement
HDF5_DIR=$PROJECT_ROOT/data/source/edge_grasp
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)

# Task -> collection count (oversampled based on historical success rates)
declare -A COLLECT_NUM
COLLECT_NUM[40]=12   # needs 7, success rate 86%
COLLECT_NUM[42]=36   # needs 18, success rate 64%
COLLECT_NUM[44]=4    # needs 2, success rate 96%
COLLECT_NUM[45]=4    # needs 2, success rate 96%
COLLECT_NUM[47]=3    # needs 1, success rate 98%
COLLECT_NUM[49]=5    # needs 3, success rate 94%

echo "============================================================"
echo "Edge Grasp Data Supplement"
echo "============================================================"
echo "Supplement dir: $SUPPLEMENT_DIR"
echo "Target HDF5 dir: $HDF5_DIR"
echo ""

# Show current status
echo "Current demo counts:"
for TASK_ID in 40 42 44 45 47 49; do
    HDF5=$HDF5_DIR/task_${TASK_ID}.hdf5
    if [ -f "$HDF5" ]; then
        COUNT=$(python3 -c "import h5py; f=h5py.File('$HDF5','r'); print(len([k for k in f['data'].keys() if k.startswith('demo')]))")
        echo "  Task $TASK_ID: $COUNT demos -> need $((50 - COUNT)), will collect ${COLLECT_NUM[$TASK_ID]}"
    else
        echo "  Task $TASK_ID: FILE NOT FOUND"
    fi
done
echo ""

rm -rf "$SUPPLEMENT_DIR"
mkdir -p "$SUPPLEMENT_DIR"

# Step 1: Collect supplement demos
echo "============================================================"
echo "Step 1: Collecting supplement demos"
echo "============================================================"

for TASK_ID in 40 42 44 45 47 49; do
    echo ""
    echo "--- Task $TASK_ID: collecting ${COLLECT_NUM[$TASK_ID]} demos ---"
    cd "$PROJECT_DIR"
    python collect/scripted_collect_edge_grasp.py \
        --task_id $TASK_ID \
        --num_episodes ${COLLECT_NUM[$TASK_ID]} \
        --output_dir "$SUPPLEMENT_DIR"
done

# Step 2: Verify + Append
echo ""
echo "============================================================"
echo "Step 2: Verify + Append"
echo "============================================================"

cd "$PROJECT_DIR"
for TASK_ID in 40 42 44 45 47 49; do
    echo ""
    echo "--- Task $TASK_ID ---"
    SUPPLEMENT_HDF5="$SUPPLEMENT_DIR/task_${TASK_ID}.hdf5"

    if [ ! -f "$SUPPLEMENT_HDF5" ]; then
        echo "  WARNING: $SUPPLEMENT_HDF5 not found, skipping"
        continue
    fi

    # Verify supplement
    python "$SCRIPT_DIR/verify_hdf5.py" \
        --hdf5 "$SUPPLEMENT_HDF5" \
        --task_id $TASK_ID \
        --output_json "$SUPPLEMENT_DIR/verify_${TASK_ID}.json" || true

    # Append verified demos
    python "$SCRIPT_DIR/append_demos.py" \
        --target "$HDF5_DIR/task_${TASK_ID}.hdf5" \
        --source "$SUPPLEMENT_HDF5" \
        --verified_json "$SUPPLEMENT_DIR/verify_${TASK_ID}.json"
done

# Step 3: Full verification of all tasks
echo ""
echo "============================================================"
echo "Step 3: Full HDF5 verification (all 10 tasks)"
echo "============================================================"

ALL_PASS=true
cd "$PROJECT_DIR"
for TASK_ID in 40 41 42 43 44 45 46 47 49 50; do
    echo ""
    HDF5="$HDF5_DIR/task_${TASK_ID}.hdf5"
    if [ ! -f "$HDF5" ]; then
        echo "Task $TASK_ID: FILE NOT FOUND"
        ALL_PASS=false
        continue
    fi

    COUNT=$(python3 -c "import h5py; f=h5py.File('$HDF5','r'); print(len([k for k in f['data'].keys() if k.startswith('demo')]))")
    echo "Task $TASK_ID: $COUNT demos"

    if [ "$COUNT" -lt 50 ]; then
        echo "  WARNING: only $COUNT demos (< 50)"
        ALL_PASS=false
    fi

    python "$SCRIPT_DIR/verify_hdf5.py" \
        --hdf5 "$HDF5" \
        --task_id $TASK_ID \
        --output_json "$SUPPLEMENT_DIR/fullcheck_${TASK_ID}.json" || ALL_PASS=false
done

# Summary
echo ""
echo "============================================================"
echo "Summary"
echo "============================================================"
echo ""
printf "%-8s %8s %8s\n" "Task" "Demos" "Status"
echo "--------------------------------"
TOTAL=0
for TASK_ID in 40 41 42 43 44 45 46 47 49 50; do
    HDF5="$HDF5_DIR/task_${TASK_ID}.hdf5"
    if [ -f "$HDF5" ]; then
        COUNT=$(python3 -c "import h5py; f=h5py.File('$HDF5','r'); print(len([k for k in f['data'].keys() if k.startswith('demo')]))")
        TOTAL=$((TOTAL + COUNT))

        # Check full verification result
        JSON="$SUPPLEMENT_DIR/fullcheck_${TASK_ID}.json"
        if [ -f "$JSON" ]; then
            PASS=$(python3 -c "import json; d=json.load(open('$JSON')); print(sum(r['success'] for r in d['results']))")
            FAIL=$((COUNT - PASS))
            if [ "$FAIL" -eq 0 ]; then
                STATUS="OK"
            else
                STATUS="${FAIL} FAIL"
            fi
        else
            STATUS="no verify"
        fi
        printf "%-8s %8s %8s\n" "$TASK_ID" "$COUNT" "$STATUS"
    else
        printf "%-8s %8s %8s\n" "$TASK_ID" "MISSING" "-"
    fi
done
echo "--------------------------------"
printf "%-8s %8s\n" "Total" "$TOTAL"

if $ALL_PASS; then
    echo ""
    echo "All tasks verified successfully!"
else
    echo ""
    echo "WARNING: Some tasks have issues. Check output above."
    exit 1
fi
