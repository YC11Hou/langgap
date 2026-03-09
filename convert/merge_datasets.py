#!/usr/bin/env python3
"""
Merge the new multi-task dataset with official lerobot/libero dataset.

Steps:
1. Fix features compatibility (names format)
2. Download official dataset to local
3. Merge using aggregate_datasets
4. Verify the merged dataset

Usage:
    conda activate lerobot
    python convert/merge_datasets.py --output /path/to/merged_dataset
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fix_features_compatibility(dataset_path: Path):
    """
    Fix the features names format to match official dataset.

    Official format: {"names": ["state"], "fps": 10.0}
    Our format: {"names": {"axes": [...]}}

    We need to convert our format to match official.
    """
    info_path = dataset_path / "meta" / "info.json"

    with open(info_path, 'r') as f:
        info = json.load(f)

    # Backup original (only if no backup exists yet)
    backup_path = info_path.with_suffix('.json.bak')
    if not backup_path.exists():
        shutil.copy(info_path, backup_path)
        logger.info(f"Backed up info.json to {backup_path}")

    # Ensure fps is float first (used below)
    fps = float(info['fps'])
    info['fps'] = fps

    # Fix observation.state
    if 'observation.state' in info['features']:
        state_feature = info['features']['observation.state']
        if isinstance(state_feature.get('names'), dict):
            state_feature['names'] = ['state']
        state_feature['fps'] = fps

    # Fix action
    if 'action' in info['features']:
        action_feature = info['features']['action']
        if isinstance(action_feature.get('names'), dict):
            action_feature['names'] = ['actions']
        action_feature['fps'] = fps

    # Fix video features: add fps if missing
    for key, feat in info['features'].items():
        if feat.get('dtype') == 'video':
            feat['fps'] = fps

    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)

    logger.info(f"Fixed features compatibility in {info_path}")


def download_official_dataset(local_dir: Path) -> Path:
    """
    Download the official lerobot/libero dataset to local.

    Args:
        local_dir: Path to download the dataset to

    Returns:
        Path to the downloaded dataset
    """
    logger.info("Downloading official lerobot/libero dataset (~34 GB)...")
    logger.info(f"Target directory: {local_dir}")

    if local_dir.exists() and (local_dir / "meta" / "info.json").exists():
        logger.info(f"Official dataset already exists at {local_dir}")
        return local_dir

    local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id="lerobot/libero",
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )

    logger.info(f"Downloaded official dataset to {local_dir}")
    return local_dir


def merge_datasets(
    official_path: Path,
    new_data_path: Path,
    output_path: Path,
    repo_id: str = "<YOUR_HF_USERNAME>/langgap_full",
):
    """
    Merge official and new datasets using LeRobot's aggregate_datasets.
    """
    from lerobot.datasets.aggregate import aggregate_datasets

    logger.info("Starting dataset merge...")
    logger.info(f"  Official dataset: {official_path}")
    logger.info(f"  New dataset: {new_data_path}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Repo ID: {repo_id}")

    # Clean output directory if exists
    if output_path.exists():
        logger.warning(f"Output directory exists, removing: {output_path}")
        shutil.rmtree(output_path)

    # Use aggregate_datasets
    # Note: repo_ids are just identifiers, roots point to actual data
    aggregate_datasets(
        repo_ids=["lerobot/libero", "<YOUR_HF_USERNAME>/langgap_ext"],
        aggr_repo_id=repo_id,
        roots=[official_path, new_data_path],
        aggr_root=output_path,
    )

    logger.info(f"Merge complete! Output at: {output_path}")


def verify_merged_dataset(merged_path: Path, repo_id: str = "<YOUR_HF_USERNAME>/langgap_full"):
    """
    Verify the merged dataset.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    logger.info("Verifying merged dataset...")

    # Load metadata
    meta = LeRobotDatasetMetadata(repo_id, root=merged_path)

    print("\n" + "="*60)
    print("MERGED DATASET SUMMARY")
    print("="*60)
    print(f"Total Episodes: {meta.total_episodes}")
    print(f"Total Frames: {meta.total_frames}")
    print(f"Total Tasks: {len(meta.tasks)}")
    print(f"FPS: {meta.fps}")
    print(f"Robot Type: {meta.robot_type}")
    print("\nTasks:")
    for task_name, row in meta.tasks.iterrows():
        print(f"  [{row['task_index']:2d}] {task_name}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Merge datasets for LIBERO multi-task training")
    parser.add_argument(
        "--new-data",
        type=Path,
        default=Path("/tmp/langgap_ext"),
        help="Path to the new multi-task dataset (our 16-task extension)"
    )
    parser.add_argument(
        "--official-data",
        type=Path,
        default=Path("./data/lerobot-libero-official"),
        help="Path to download/store official lerobot/libero dataset (~34 GB)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/langgap_full"),
        help="Output path for merged dataset"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="<YOUR_HF_USERNAME>/langgap_full",
        help="HuggingFace repo ID for the merged dataset"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading official dataset (use if already downloaded)"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify the merged dataset"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually doing it"
    )

    args = parser.parse_args()

    if args.dry_run:
        print("\n" + "="*60)
        print("DRY RUN - Data Merge Plan")
        print("="*60)
        print(f"\nNew data (our extension): {args.new_data}")
        print(f"  - 16 tasks, 2400 episodes, 269,490 frames")
        print(f"\nOfficial data: {args.official_data}")
        print(f"  - 40 tasks, 1693 episodes, 273,465 frames")
        print(f"\nMerged output: {args.output}")
        print(f"  Repo ID: {args.repo_id}")
        print(f"  - 56 tasks, 4093 episodes, 542,955 frames")
        print("\nSteps:")
        print("  1. Fix features compatibility in new data")
        print("  2. Download official dataset (if not exists)")
        print("  3. Merge using LeRobot aggregate_datasets")
        print("  4. Verify merged dataset")
        print("="*60)
        return

    if args.verify_only:
        verify_merged_dataset(args.output, repo_id=args.repo_id)
        return

    # Step 1: Fix features compatibility
    logger.info("Step 1: Fix features compatibility")
    fix_features_compatibility(args.new_data)

    # Step 2: Download official dataset
    if not args.skip_download:
        logger.info("Step 2: Download official dataset (~34 GB)")
        official_path = download_official_dataset(args.official_data)
    else:
        official_path = args.official_data
        if not official_path.exists():
            raise ValueError(f"Official dataset not found at {official_path}")

    # Step 3: Merge datasets
    logger.info("Step 3: Merge datasets")
    merge_datasets(
        official_path=official_path,
        new_data_path=args.new_data,
        output_path=args.output,
        repo_id=args.repo_id,
    )

    # Step 4: Verify
    logger.info("Step 4: Verify merged dataset")
    verify_merged_dataset(args.output, repo_id=args.repo_id)

    logger.info("All done!")


if __name__ == "__main__":
    main()
