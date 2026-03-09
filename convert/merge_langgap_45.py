#!/usr/bin/env python3
"""
Merge lerobot/libero (40 tasks) + <YOUR_HF_USERNAME>/langgap_6 (6 tasks) = 45 tasks.

Deduplication: langgap_6 task 0 ("Pick the akita black bowl...") is the same
physical task as official task 34 ("pick up the black bowl between the plate
and the ramekin and place it on the plate"). We rename langgap_6's task 0 to
match the official description so aggregate_datasets merges them automatically.

Result: 40 + 6 - 1 duplicate = 45 unique tasks, ~1993 episodes.

Usage:
    conda activate lerobot
    python convert/merge_langgap_45.py
"""

import json
import logging
import shutil
import tempfile
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
LIBERO_CACHE = Path.home() / ".cache/huggingface/lerobot/lerobot/libero"
LANGGAP6_CACHE = Path.home() / ".cache/huggingface/lerobot/<YOUR_HF_USERNAME>/langgap_6"
OUTPUT_DIR = Path("./data/langgap_45")
REPO_ID = "<YOUR_HF_USERNAME>/langgap_45"

# The duplicate task: langgap_6 task 0 → official task 34
DUPLICATE_OLD = "Pick the akita black bowl between the plate and the ramekin and place it on the plate"
DUPLICATE_NEW = "pick up the black bowl between the plate and the ramekin and place it on the plate"


def fix_features_and_fps(info_path: Path):
    """Fix features names format and fps type to match between datasets."""
    with open(info_path) as f:
        info = json.load(f)

    fps = float(info["fps"])
    info["fps"] = fps

    if "observation.state" in info["features"]:
        feat = info["features"]["observation.state"]
        if isinstance(feat.get("names"), dict):
            feat["names"] = ["state"]
        feat["fps"] = fps

    if "action" in info["features"]:
        feat = info["features"]["action"]
        if isinstance(feat.get("names"), dict):
            feat["names"] = ["actions"]
        feat["fps"] = fps

    for key, feat in info["features"].items():
        if feat.get("dtype") == "video":
            feat["fps"] = fps

    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)


def add_missing_episode_columns(dataset_path: Path):
    """Add v3.0 episode columns (tasks, meta/episodes/chunk_index, etc.) if missing."""
    episodes_dir = dataset_path / "meta" / "episodes"
    tasks_df = pd.read_parquet(dataset_path / "meta" / "tasks.parquet")
    # Build task_index → description mapping
    task_idx_to_desc = {row["task_index"]: desc for desc, row in tasks_df.iterrows()}

    # Build episode_index → task_index mapping from data parquet
    data_dir = dataset_path / "data"
    ep_to_task = {}
    for pf in sorted(data_dir.rglob("*.parquet")):
        df = pd.read_parquet(pf, columns=["episode_index", "task_index"])
        for ep_idx, grp in df.groupby("episode_index"):
            ep_to_task[int(ep_idx)] = int(grp["task_index"].iloc[0])

    for ep_file in sorted(episodes_dir.rglob("*.parquet")):
        df = pd.read_parquet(ep_file)
        modified = False

        if "meta/episodes/chunk_index" not in df.columns:
            # All episodes in chunk-000/file-000 → both are 0
            chunk_idx = int(ep_file.parent.name.split("-")[-1])
            file_idx = int(ep_file.stem.split("-")[-1])
            df["meta/episodes/chunk_index"] = chunk_idx
            df["meta/episodes/file_index"] = file_idx
            modified = True

        if "tasks" not in df.columns:
            df["tasks"] = df["episode_index"].apply(
                lambda ep: [task_idx_to_desc.get(ep_to_task.get(int(ep), 0), "unknown")]
            )
            modified = True

        if modified:
            df.to_parquet(ep_file, index=False)
            logger.info(f"Added missing columns to {ep_file}")


def prepare_langgap6_copy(src: Path, dst: Path):
    """Copy langgap_6 to temp dir with feature fixes and task dedup rename."""
    logger.info(f"Copying langgap_6 to {dst}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    fix_features_and_fps(dst / "meta" / "info.json")
    logger.info("Fixed langgap_6 features compatibility")

    # Rename duplicate task in tasks.parquet
    tasks_path = dst / "meta" / "tasks.parquet"
    tasks = pd.read_parquet(tasks_path)
    old_index = tasks.index.tolist()
    new_index = [DUPLICATE_NEW if t == DUPLICATE_OLD else t for t in old_index]
    assert sum(1 for t in old_index if t == DUPLICATE_OLD) == 1, "Expected exactly 1 duplicate task"
    tasks.index = new_index
    tasks.to_parquet(tasks_path)
    logger.info(f"Renamed duplicate task: '{DUPLICATE_OLD[:50]}...' → '{DUPLICATE_NEW[:50]}...'")

    # Also rename in episodes parquet (the 'tasks' column is a list of task descriptions)
    episodes_dir = dst / "meta" / "episodes"
    for ep_file in sorted(episodes_dir.rglob("*.parquet")):
        df = pd.read_parquet(ep_file)
        if "tasks" in df.columns:
            df["tasks"] = df["tasks"].apply(
                lambda task_list: [DUPLICATE_NEW if t == DUPLICATE_OLD else t for t in task_list]
            )
            df.to_parquet(ep_file, index=False)
    logger.info("Renamed duplicate task in episodes metadata")


def prepare_libero_copy(src: Path, dst: Path):
    """Copy libero to temp dir and add missing v3.0 episode columns."""
    logger.info(f"Copying libero to {dst}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    fix_features_and_fps(dst / "meta" / "info.json")
    add_missing_episode_columns(dst)
    logger.info("Patched libero with missing v3.0 columns")


def merge(libero_path: Path, langgap6_path: Path, output_path: Path, repo_id: str):
    """Merge using LeRobot's aggregate_datasets."""
    from lerobot.datasets.aggregate import aggregate_datasets

    if output_path.exists():
        logger.warning(f"Output exists, removing: {output_path}")
        shutil.rmtree(output_path)

    logger.info("Running aggregate_datasets...")
    aggregate_datasets(
        repo_ids=["lerobot/libero", "<YOUR_HF_USERNAME>/langgap_6"],
        aggr_repo_id=repo_id,
        roots=[libero_path, langgap6_path],
        aggr_root=output_path,
    )
    logger.info(f"Merge complete: {output_path}")


def verify(output_path: Path, repo_id: str):
    """Verify the merged dataset."""
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id, root=output_path)

    print("\n" + "=" * 60)
    print("MERGED DATASET: langgap_45")
    print("=" * 60)
    print(f"Total Episodes: {meta.total_episodes}")
    print(f"Total Frames:   {meta.total_frames}")
    print(f"Total Tasks:    {len(meta.tasks)}")
    print(f"FPS:            {meta.fps}")

    print("\nTasks:")
    for task_name, row in meta.tasks.iterrows():
        print(f"  [{row['task_index']:2d}] {task_name}")

    # Assertions from prompt
    n_eps = meta.total_episodes
    n_tasks = len(meta.tasks)
    assert 1900 <= n_eps <= 2000, f"Expected ~1993 episodes, got {n_eps}"
    assert n_tasks == 45, f"Expected 45 tasks, got {n_tasks}"

    print(f"\nVerification PASSED: {n_tasks} tasks, {n_eps} episodes")
    print("=" * 60)


def main():
    # Check source datasets exist
    assert LIBERO_CACHE.exists(), f"libero not cached: {LIBERO_CACHE}"
    assert LANGGAP6_CACHE.exists(), f"langgap_6 not cached: {LANGGAP6_CACHE}"

    # Step 1: Prepare patched copies
    tmp_base = Path(tempfile.mkdtemp(prefix="langgap45_merge_"))
    tmp_libero = tmp_base / "libero"
    tmp_lg6 = tmp_base / "langgap_6"
    try:
        prepare_libero_copy(LIBERO_CACHE, tmp_libero)
        prepare_langgap6_copy(LANGGAP6_CACHE, tmp_lg6)

        # Step 2: Merge
        merge(tmp_libero, tmp_lg6, OUTPUT_DIR, REPO_ID)

        # Step 3: Verify
        verify(OUTPUT_DIR, REPO_ID)
    finally:
        shutil.rmtree(tmp_base, ignore_errors=True)
        logger.info(f"Cleaned up temp dir: {tmp_base}")

    logger.info("All done!")


if __name__ == "__main__":
    main()
