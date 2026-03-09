#!/usr/bin/env python3
"""Rename manually collected demo files to a standardized naming convention.

Naming format: {suite}_{dimension}_{task_id}.hdf5

Examples:
- spatial_dim1_task0_bowl2.hdf5
- spatial_dim2_task0_to_stove.hdf5
- goal_dim1_cream_cheese_to_stove.hdf5
- object_dim1_scene0_salad_dressing.hdf5
"""

import os
import re
import shutil
from pathlib import Path

# Source directory
SOURCE_DIR = Path(__file__).resolve().parent.parent / "data" / "source"

# Task mapping table (extracted from TRAINING_TASK_SELECTION.md)
TASK_MAPPING = {
    # libero_spatial - dim1_change_bowl
    "ext_01_task0_bowl2": ("spatial", "dim1", "task0_bowl2"),
    "ext_02_task2_bowl2": ("spatial", "dim1", "task2_bowl2"),
    "ext_04_task4_bowl2": ("spatial", "dim1", "task4_bowl2"),

    # libero_spatial - dim2_change_target
    "ext_01_task0_to_stove": ("spatial", "dim2", "task0_to_stove"),
    "ext_02_task0_to_cabinet": ("spatial", "dim2", "task0_to_cabinet"),
    "ext_04_task2_to_ramekin": ("spatial", "dim2", "task2_to_ramekin"),
    "ext_07_task7_to_cabinet": ("spatial", "dim2", "task7_to_cabinet"),
    "ext_09_task8_to_stove": ("spatial", "dim2", "task8_to_stove"),
    "ext_11_task9_to_stove": ("spatial", "dim2", "task9_to_stove"),

    # libero_spatial - dim3_change_object
    "ext_01_ramekin_to_plate": ("spatial", "dim3", "ramekin_to_plate"),
    "ext_03_ramekin_to_cabinet": ("spatial", "dim3", "ramekin_to_cabinet"),
    "ext_05_cookie_box_to_plate": ("spatial", "dim3", "cookie_box_to_plate"),
    "ext_07_cookie_box_to_cabinet": ("spatial", "dim3", "cookie_box_to_cabinet"),

    # libero_spatial - dim5_drawer_action
    "ext_03_open_middle": ("spatial", "dim5", "open_middle"),

    # libero_goal - dim1_change_object
    "ext_03_cream_cheese_to_stove": ("goal", "dim1", "cream_cheese_to_stove"),
    "ext_06_wine_bottle_to_plate": ("goal", "dim1", "wine_bottle_to_plate"),
    "ext_08_bowl_to_rack": ("goal", "dim1", "bowl_to_rack"),
    "ext_11_bottom_drawer": ("goal", "dim1", "bottom_drawer"),

    # libero_object - dim1_change_object
    "ext_01_scene0_salad_dressing": ("object", "dim1", "scene0_salad_dressing"),
    "ext_03_scene1_alphabet_soup": ("object", "dim1", "scene1_alphabet_soup"),
    "ext_05_scene2_milk": ("object", "dim1", "scene2_milk"),
    "ext_07_scene3_ketchup": ("object", "dim1", "scene3_ketchup"),
    "ext_09_scene4_bbq_sauce": ("object", "dim1", "scene4_bbq_sauce"),
    "ext_11_scene5_bbq_sauce": ("object", "dim1", "scene5_bbq_sauce"),
    "ext_13_scene6_ketchup": ("object", "dim1", "scene6_ketchup"),
    "ext_15_scene7_butter": ("object", "dim1", "scene7_butter"),
    "ext_17_scene8_orange_juice": ("object", "dim1", "scene8_orange_juice"),
    "ext_20_scene9_chocolate_pudding": ("object", "dim1", "scene9_chocolate_pudding"),
    "ext_22_scene9_salad_dressing": ("object", "dim1", "scene9_salad_dressing"),
}


def extract_task_key(filename: str) -> str | None:
    """Extract task key from filename (remove timestamp suffix)."""
    # Remove .hdf5 suffix
    name = filename.replace(".hdf5", "")
    # Remove timestamp suffix _YYYYMMDD_HHMMSS
    match = re.match(r"(.+)_\d{8}_\d{6}$", name)
    if match:
        return match.group(1)
    return name


def get_new_filename(task_key: str) -> str | None:
    """Generate new filename based on task key."""
    if task_key not in TASK_MAPPING:
        return None
    suite, dim, task_id = TASK_MAPPING[task_key]
    return f"{suite}_{dim}_{task_id}.hdf5"


def main():
    """Main function: rename all demo files."""
    print("=" * 60)
    print("Demo File Rename Tool")
    print("=" * 60)

    if not SOURCE_DIR.exists():
        print(f"Error: source directory does not exist {SOURCE_DIR}")
        return

    hdf5_files = list(SOURCE_DIR.glob("*.hdf5"))
    print(f"\nFound {len(hdf5_files)} HDF5 files")

    # Analyze files
    rename_plan = []
    unmatched = []

    for f in sorted(hdf5_files):
        task_key = extract_task_key(f.name)
        new_name = get_new_filename(task_key)

        if new_name:
            rename_plan.append((f, new_name, task_key))
        else:
            unmatched.append((f.name, task_key))

    # Display rename plan
    print(f"\nMatched: {len(rename_plan)} files")
    print("-" * 60)
    for old_file, new_name, task_key in rename_plan:
        print(f"  {old_file.name}")
        print(f"    → {new_name}")
        print()

    if unmatched:
        print(f"\nUnmatched: {len(unmatched)} files")
        print("-" * 60)
        for fname, task_key in unmatched:
            print(f"  {fname}")
            print(f"    task_key: {task_key}")
            print()

    # Confirm execution
    print("=" * 60)
    response = input("Execute rename? (y/n): ").strip().lower()

    if response != "y":
        print("Cancelled")
        return

    # Execute rename
    print("\nRenaming...")
    for old_file, new_name, _ in rename_plan:
        new_path = SOURCE_DIR / new_name
        if new_path.exists():
            print(f"  SKIP (target exists): {new_name}")
            continue
        shutil.move(old_file, new_path)
        print(f"  OK {old_file.name} -> {new_name}")

    print("\nDone!")

    # Display final statistics
    print("\n" + "=" * 60)
    print("Statistics by suite:")
    final_files = list(SOURCE_DIR.glob("*.hdf5"))
    by_suite = {"spatial": [], "goal": [], "object": [], "other": []}
    for f in final_files:
        for suite in ["spatial", "goal", "object"]:
            if f.name.startswith(suite):
                by_suite[suite].append(f.name)
                break
        else:
            by_suite["other"].append(f.name)

    for suite, files in by_suite.items():
        if files:
            print(f"\n{suite}: {len(files)} files")
            for fname in sorted(files):
                print(f"  - {fname}")


if __name__ == "__main__":
    main()
