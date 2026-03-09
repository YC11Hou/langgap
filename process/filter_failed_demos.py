#!/usr/bin/env python3
"""Filter failed replay episodes from edge_grasp HDF5 files.

Reads replay result JSONs (from batch_replay_check.sh), identifies failed episodes,
and removes the corresponding demo groups from source HDF5 files. Remaining demos
are renumbered to be contiguous (demo_1, demo_2, ..., demo_N).

IMPORTANT: Episode indices in the JSON are 0-based and correspond to the sorted()
string order of HDF5 demo keys. E.g., episode 0 → demo_1, episode 1 → demo_10,
episode 11 → demo_2. This script handles the mapping correctly.

Usage:
    python filter_failed_demos.py \
        --hdf5_dir $PROJECT_ROOT/data/source/edge_grasp \
        --results_dir replay_results
"""

import argparse
import json
import shutil
from pathlib import Path

import h5py


def get_failed_demo_keys(json_path, hdf5_path):
    """Map failed episode indices from JSON to HDF5 demo key names.

    Returns (keys_to_delete, demo_keys_sorted) where demo_keys_sorted is the
    sorted list of all demo keys in the HDF5 file.
    """
    with open(json_path) as f:
        data = json.load(f)

    failed_indices = [r["episode"] for r in data["results"] if not r["success"]]
    if not failed_indices:
        return [], []

    with h5py.File(str(hdf5_path), "r") as f:
        demo_keys = sorted([k for k in f["data"].keys() if k.startswith("demo")])

    keys_to_delete = [demo_keys[i] for i in failed_indices]
    return keys_to_delete, demo_keys


def filter_hdf5(hdf5_path, keys_to_delete):
    """Remove failed demo groups and renumber remaining demos.

    Creates a .bak backup before modifying.
    Returns (original_count, new_count).
    """
    # Backup
    bak_path = Path(str(hdf5_path) + ".bak")
    if not bak_path.exists():
        shutil.copy2(hdf5_path, bak_path)
        print(f"  Backup: {bak_path}")
    else:
        print(f"  Backup already exists: {bak_path}")

    with h5py.File(str(hdf5_path), "a") as f:
        demo_keys = sorted([k for k in f["data"].keys() if k.startswith("demo")])
        original_count = len(demo_keys)

        # Delete failed demos
        for key in keys_to_delete:
            del f["data"][key]
            print(f"    Deleted data/{key}")

        # Get remaining keys and renumber
        remaining_keys = sorted([k for k in f["data"].keys() if k.startswith("demo")])
        new_count = len(remaining_keys)

        # Renumber: first rename to temp names to avoid collisions
        for i, key in enumerate(remaining_keys):
            f["data"].move(key, f"__temp_{i}")

        for i in range(new_count):
            f["data"].move(f"__temp_{i}", f"demo_{i + 1}")

        # Update total attribute
        f["data"].attrs["total"] = new_count

    return original_count, new_count


def main():
    parser = argparse.ArgumentParser(
        description="Filter failed replay episodes from edge_grasp HDF5 files."
    )
    parser.add_argument(
        "--hdf5_dir",
        required=True,
        help="Directory containing task_XX.hdf5 files",
    )
    parser.add_argument(
        "--results_dir",
        required=True,
        help="Directory containing task_XX.json replay results",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be deleted without modifying files",
    )
    args = parser.parse_args()

    hdf5_dir = Path(args.hdf5_dir).expanduser()
    results_dir = Path(args.results_dir).expanduser()

    # Find all result JSONs
    json_files = sorted(results_dir.glob("task_*.json"))
    if not json_files:
        print(f"No task_*.json files found in {results_dir}")
        return

    print(f"HDF5 dir: {hdf5_dir}")
    print(f"Results dir: {results_dir}")
    print(f"Found {len(json_files)} result files")
    print()

    summary = []
    total_original = 0
    total_deleted = 0

    for json_path in json_files:
        task_id = json_path.stem  # e.g. "task_40"
        hdf5_path = hdf5_dir / f"{task_id}.hdf5"

        if not hdf5_path.exists():
            print(f"WARNING: {hdf5_path} not found, skipping")
            continue

        keys_to_delete, demo_keys = get_failed_demo_keys(json_path, hdf5_path)
        n_delete = len(keys_to_delete)

        if n_delete == 0:
            print(f"{task_id}: 0 failures, skipping")
            summary.append((task_id, 50, 0, 50))
            total_original += 50
            continue

        print(f"{task_id}: {n_delete} failures -> deleting {keys_to_delete}")

        if args.dry_run:
            summary.append((task_id, len(demo_keys), n_delete, len(demo_keys) - n_delete))
            total_original += len(demo_keys)
            total_deleted += n_delete
        else:
            original, new = filter_hdf5(hdf5_path, keys_to_delete)
            summary.append((task_id, original, n_delete, new))
            total_original += original
            total_deleted += n_delete

    # Also count tasks without JSON (100% success)
    all_task_ids = [40, 41, 42, 43, 44, 45, 46, 47, 49, 50]
    json_task_ids = {json_path.stem for json_path in json_files}
    for tid in all_task_ids:
        task_name = f"task_{tid}"
        if task_name not in json_task_ids:
            hdf5_path = hdf5_dir / f"{task_name}.hdf5"
            if hdf5_path.exists():
                with h5py.File(str(hdf5_path), "r") as f:
                    n = len([k for k in f["data"].keys() if k.startswith("demo")])
                summary.append((task_name, n, 0, n))
                total_original += n

    # Print summary
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"{'Task':<10} {'Before':>8} {'Deleted':>8} {'After':>8}")
    print("-" * 50)
    for task_id, before, deleted, after in sorted(summary):
        print(f"{task_id:<10} {before:>8} {deleted:>8} {after:>8}")
    print("-" * 50)
    total_after = total_original - total_deleted
    print(f"{'TOTAL':<10} {total_original:>8} {total_deleted:>8} {total_after:>8}")

    if args.dry_run:
        print("\n(dry run - no files modified)")

    # Verify renumbering
    if not args.dry_run and total_deleted > 0:
        print("\nVerifying renumbered files...")
        all_ok = True
        for task_id, before, deleted, after in sorted(summary):
            if deleted == 0:
                continue
            hdf5_path = hdf5_dir / f"{task_id}.hdf5"
            with h5py.File(str(hdf5_path), "r") as f:
                demo_keys = set(
                    k for k in f["data"].keys() if k.startswith("demo")
                )
                expected = {f"demo_{i}" for i in range(1, after + 1)}
                attr_total = f["data"].attrs["total"]
                ok = demo_keys == expected and attr_total == after
                status = "OK" if ok else "MISMATCH"
                if not ok:
                    all_ok = False
                print(f"  {task_id}: {len(demo_keys)} demos, total={attr_total} [{status}]")
        if all_ok:
            print("All files verified OK.")
        else:
            print("WARNING: Some files have mismatched demo keys!")


if __name__ == "__main__":
    main()
