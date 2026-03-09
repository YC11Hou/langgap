#!/usr/bin/env python3
"""Append verified supplement demos to an existing HDF5 file.

Reads verification results (from verify_hdf5.py), takes only passed demos
from the source HDF5, and appends all of them to the target HDF5.

Usage:
    python process/append_demos.py \
        --target data/source/edge_grasp/task_40.hdf5 \
        --source /tmp/supplement/task_40.hdf5 \
        --verified_json /tmp/supplement/verify_40.json

    # Without pre-computed verification (will verify inline)
    python process/append_demos.py \
        --target data/source/edge_grasp/task_40.hdf5 \
        --source /tmp/supplement/task_40.hdf5 \
        --task_id 40
"""

import argparse
import json
import sys
from pathlib import Path

import h5py


def load_verified_indices(json_path):
    """Load passed episode indices from verification JSON."""
    with open(json_path) as f:
        data = json.load(f)
    return [r["episode"] for r in data["results"] if r["success"]]


def append_demos(target_path, source_path,
                 passed_indices=None, task_id=None):
    """Append passed demos from source to target HDF5.

    Args:
        target_path: Path to target HDF5 (will be modified in place)
        source_path: Path to source/supplement HDF5
        passed_indices: List of 0-based episode indices that passed verification.
                       If None, will run inline verification (requires task_id).
        task_id: Task ID for inline verification (only used if passed_indices is None)

    Returns:
        Number of demos actually appended.
    """
    # Run inline verification if no pre-computed results
    if passed_indices is None:
        if task_id is None:
            print("Error: must provide --verified_json or --task_id")
            sys.exit(1)
        print(f"Running inline verification for {source_path}...")
        from verify_hdf5 import verify_hdf5
        results = verify_hdf5(str(source_path), task_id)
        passed_indices = [r["episode"] for r in results if r["success"]]

    if not passed_indices:
        print("No passed demos in source, nothing to append.")
        return 0

    # Get source demo keys
    with h5py.File(str(source_path), "r") as fs:
        src_demo_keys = sorted(
            [k for k in fs["data"].keys() if k.startswith("demo")])

    passed_keys = [src_demo_keys[i] for i in passed_indices
                   if i < len(src_demo_keys)]

    # Get current target count
    with h5py.File(str(target_path), "r") as ft:
        existing_keys = [k for k in ft["data"].keys() if k.startswith("demo")]
        existing_count = len(existing_keys)

    to_append = passed_keys
    n_append = len(to_append)

    print(f"Appending {n_append} demos to {target_path} "
          f"({existing_count} -> {existing_count + n_append})")

    # Append demos
    with h5py.File(str(source_path), "r") as fs, \
         h5py.File(str(target_path), "a") as ft:
        for i, src_key in enumerate(to_append):
            new_name = f"demo_{existing_count + i + 1}"
            fs.copy(f"data/{src_key}", ft["data"], name=new_name)
            # Copy num_samples attribute if present
            if "num_samples" in fs[f"data/{src_key}"].attrs:
                ft[f"data/{new_name}"].attrs["num_samples"] = \
                    fs[f"data/{src_key}"].attrs["num_samples"]

        # Update total
        new_total = existing_count + n_append
        ft["data"].attrs["total"] = new_total
        print(f"Updated total: {new_total}")

    return n_append


def main():
    parser = argparse.ArgumentParser(
        description="Append verified supplement demos to target HDF5")
    parser.add_argument("--target", required=True,
                        help="Target HDF5 file (will be modified)")
    parser.add_argument("--source", required=True,
                        help="Source/supplement HDF5 file")
    parser.add_argument("--verified_json", default=None,
                        help="Pre-computed verification JSON from verify_hdf5.py")
    parser.add_argument("--task_id", type=int, default=None,
                        help="Task ID for inline verification (if no --verified_json)")
    args = parser.parse_args()

    target_path = Path(args.target).expanduser()
    source_path = Path(args.source).expanduser()

    if not target_path.exists():
        print(f"Error: target HDF5 not found: {target_path}")
        sys.exit(1)
    if not source_path.exists():
        print(f"Error: source HDF5 not found: {source_path}")
        sys.exit(1)

    # Load verification results
    passed_indices = None
    if args.verified_json:
        json_path = Path(args.verified_json).expanduser()
        if not json_path.exists():
            print(f"Error: verified JSON not found: {json_path}")
            sys.exit(1)
        passed_indices = load_verified_indices(str(json_path))
        print(f"Loaded verification: {len(passed_indices)} passed demos")

    n_appended = append_demos(
        target_path, source_path,
        passed_indices=passed_indices,
        task_id=args.task_id,
    )

    # Final check
    with h5py.File(str(target_path), "r") as f:
        final_keys = [k for k in f["data"].keys() if k.startswith("demo")]
        final_total = f["data"].attrs["total"]
    print(f"Final: {len(final_keys)} demos, total attr = {final_total}")


if __name__ == "__main__":
    main()
