#!/usr/bin/env python3
"""
Convert manually collected source HDF5 files to LeRobot format and upload to HuggingFace.

Supports single-file and multi-file input (multi-task joint training).

Usage:
    conda activate lerobot
    export MUJOCO_GL=egl

    # Single task
    python convert/source_to_lerobot.py \
        --input data/source/task50.hdf5 \
        --bddl data/bddl_files/.../ext_03_ramekin_to_cabinet.bddl \
        --repo_id <YOUR_HF_USERNAME>/task50_ext \
        --push

    # Multi-task (--input and --bddl must match one-to-one)
    python convert/source_to_lerobot.py \
        --input task_10.hdf5 task_41.hdf5 task_44.hdf5 \
        --bddl bddl_10.bddl bddl_41.bddl bddl_44.bddl \
        --repo_id <YOUR_HF_USERNAME>/scripted_multispatial \
        --push --private
"""

import os
import sys
import argparse
import re
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

# LeRobot imports
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# LIBERO imports
from libero.libero.envs import OffScreenRenderEnv

# Features consistent with official lerobot/libero
LIBERO_FEATURES = {
    "observation.images.image": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.images.image2": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (8,),
        "names": {"axes": ["eef_x", "eef_y", "eef_z", "aa_x", "aa_y", "aa_z", "gripper_0", "gripper_1"]},
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": {"axes": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]},
    },
}

LIBERO_FPS = 10
LIBERO_ROBOT_TYPE = "panda"


def create_env(bddl_path):
    """Create a LIBERO environment for rendering."""
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=256,
        camera_widths=256,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="agentview",
        use_camera_obs=True,
    )
    return env


def get_instruction_from_bddl(bddl_path: str) -> str:
    """Extract task instruction from BDDL file."""
    with open(bddl_path, "r") as f:
        content = f.read()
    match = re.search(r"\(:language ([^)]+)\)", content)
    if match:
        return match.group(1).strip()
    return "Unknown instruction"


def quat2axisangle(quat):
    """Quaternion (x,y,z,w) -> axis-angle (3,)."""
    x, y, z, w = quat
    w = np.clip(w, -1.0, 1.0)
    den = np.sqrt(max(1.0 - w * w, 0.0))
    if den < 1e-10:
        return np.zeros(3)
    angle = 2.0 * np.arccos(w)
    axis = np.array([x, y, z]) / den
    return axis * angle


def get_observation_state(obs):
    """Construct observation.state (8-dim) -- consistent with official lerobot and LiberoProcessorStep."""
    eef_pos = obs['robot0_eef_pos']  # (3,)
    eef_quat = obs['robot0_eef_quat']  # (4,) robosuite format (x,y,z,w)
    gripper_qpos = obs['robot0_gripper_qpos']  # (2,) two joints
    # quaternion -> axis-angle (consistent with lerobot/src/lerobot/processor/env_processor.py)
    eef_axisangle = quat2axisangle(eef_quat)  # (3,)
    state = np.concatenate([eef_pos, eef_axisangle, gripper_qpos]).astype(np.float32)
    return state


def trim_idle_actions(states, actions, max_idle=5):
    """Trim leading/trailing idle (zero-action) segments. Truncate consecutive
    idle segments in the middle to max_idle steps.

    states and actions are truncated in sync, maintaining 1:1 correspondence.
    """
    is_idle = np.abs(actions).sum(axis=1) <= 1e-3

    # 1. Find first/last non-zero boundaries
    nonzero_indices = np.where(~is_idle)[0]
    if len(nonzero_indices) == 0:
        return states, actions  # All zeros, skip processing
    first, last = nonzero_indices[0], nonzero_indices[-1]

    # 2. Trim to [first, last] range (remove leading/trailing idle)
    states = states[first:last+1]
    actions = actions[first:last+1]
    is_idle = is_idle[first:last+1]

    # 3. Compress long pauses in the middle: truncate consecutive zeros beyond max_idle
    keep = np.ones(len(actions), dtype=bool)
    idle_count = 0
    for i in range(len(actions)):
        if is_idle[i]:
            idle_count += 1
            if idle_count > max_idle:
                keep[i] = False
        else:
            idle_count = 0

    return states[keep], actions[keep]


def render_trajectory(env, states, actions, task_instruction, frame_skip=1):
    """Render the full trajectory from states."""
    frames = []

    env.reset()
    obs = env.set_init_state(states[0])

    # Initial frame
    frame = {
        "observation.images.image": obs["agentview_image"][::-1].copy(),
        "observation.images.image2": obs["robot0_eye_in_hand_image"][::-1].copy(),
        "observation.state": get_observation_state(obs),
        "action": actions[0].astype(np.float32),
        "task": task_instruction,
    }
    frames.append(frame)

    # Subsequent frames
    for i in range(1, len(actions)):
        if i % frame_skip == 0:
            obs = env.set_init_state(states[i])
            frame = {
                "observation.images.image": obs["agentview_image"][::-1].copy(),
                "observation.images.image2": obs["robot0_eye_in_hand_image"][::-1].copy(),
                "observation.state": get_observation_state(obs),
                "action": actions[i].astype(np.float32),
                "task": task_instruction,
            }
            frames.append(frame)

    return frames


def process_one_hdf5(hdf5_path, bddl_path, lerobot_dataset, frame_skip=1,
                     trim_zeros=False, max_idle=5, max_demos=None):
    """Process all demos from one HDF5 file and add them to the dataset."""
    instruction = get_instruction_from_bddl(str(bddl_path))

    with h5py.File(hdf5_path, 'r') as f:
        # Fallback: try reading instruction from HDF5 if BDDL parsing fails
        if instruction == "Unknown instruction":
            instruction = f['data'].attrs.get('instruction', 'unknown task')
        demo_keys = sorted([k for k in f['data'].keys() if k.startswith('demo')])
        total_demos = len(demo_keys)

    if max_demos:
        demo_keys = demo_keys[:max_demos]

    print(f"\n  HDF5: {hdf5_path}")
    print(f"  BDDL: {bddl_path}")
    print(f"  Instruction: {instruction}")
    print(f"  Demos: {len(demo_keys)}/{total_demos}")

    # Create rendering environment
    env = create_env(str(Path(bddl_path).absolute()))

    success_count = 0
    with h5py.File(hdf5_path, 'r') as f:
        for demo_key in tqdm(demo_keys, desc=f"  {Path(hdf5_path).stem}"):
            demo = f[f'data/{demo_key}']
            states = demo['states'][:]
            actions = demo['actions'][:]

            if trim_zeros:
                orig_len = len(actions)
                states, actions = trim_idle_actions(states, actions, max_idle)
                if len(actions) < orig_len:
                    tqdm.write(f"    {demo_key}: {orig_len} -> {len(actions)} steps (trimmed)")

            try:
                frames = render_trajectory(
                    env=env,
                    states=states,
                    actions=actions,
                    task_instruction=instruction,
                    frame_skip=frame_skip,
                )

                for frame in frames:
                    lerobot_dataset.add_frame(frame)

                lerobot_dataset.save_episode()
                success_count += 1

            except Exception as e:
                tqdm.write(f"    WARNING: {demo_key} failed: {e}")
                continue

    env.close()
    return success_count


def main():
    parser = argparse.ArgumentParser(description="Convert source HDF5 to LeRobot format")
    parser.add_argument("--input", type=str, nargs='+', required=True,
                        help="Input HDF5 file(s) (multiple for multi-task)")
    parser.add_argument("--bddl", type=str, nargs='+', required=True,
                        help="BDDL file(s) for rendering (must match --input count)")
    parser.add_argument("--repo_id", type=str, required=True,
                        help="HuggingFace repo ID (e.g., <YOUR_HF_USERNAME>/scripted_multispatial)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: /tmp/<repo_name>)")
    parser.add_argument("--frame_skip", type=int, default=1,
                        help="Sim steps per stored frame (1 = keep every action)")
    parser.add_argument("--max_demos", type=int, default=None,
                        help="Max demos per HDF5 (for testing)")
    parser.add_argument("--trim_zeros", action="store_true",
                        help="Trim leading/trailing zero-action segments and compress long pauses")
    parser.add_argument("--max_idle", type=int, default=5,
                        help="Max consecutive zero-action steps to keep in the middle (default 5)")
    parser.add_argument("--push", action="store_true",
                        help="Push to HuggingFace after conversion")
    parser.add_argument("--private", action="store_true",
                        help="Make HuggingFace repo private")

    args = parser.parse_args()

    input_paths = [Path(p) for p in args.input]
    bddl_paths = [Path(p) for p in args.bddl]

    if len(input_paths) != len(bddl_paths):
        print(f"Error: --input ({len(input_paths)}) and --bddl ({len(bddl_paths)}) count mismatch")
        return

    # Validate all files exist
    for p in input_paths:
        if not p.exists():
            print(f"Error: input file not found: {p}")
            return
    for p in bddl_paths:
        if not p.exists():
            print(f"Error: BDDL file not found: {p}")
            return

    print("=" * 60)
    print("Source HDF5 -> LeRobot Conversion")
    print("=" * 60)
    print(f"Input files: {len(input_paths)}")
    for i, (inp, bddl) in enumerate(zip(input_paths, bddl_paths)):
        instruction = get_instruction_from_bddl(str(bddl))
        print(f"  [{i}] {inp.name} -> {instruction}")
    print(f"Repo ID: {args.repo_id}")
    print("=" * 60)

    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        repo_name = args.repo_id.split('/')[-1]
        output_dir = Path(f"/tmp/{repo_name}")

    # Clean existing data (LeRobotDataset.create requires directory to not exist)
    if output_dir.exists():
        print(f"Cleaning existing data: {output_dir}")
        import shutil
        shutil.rmtree(output_dir)

    # Create LeRobot dataset
    print(f"\nCreating LeRobot dataset...")
    lerobot_dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=LIBERO_FPS,
        features=LIBERO_FEATURES,
        robot_type=LIBERO_ROBOT_TYPE,
        root=output_dir,
        use_videos=True,
    )

    # Process each HDF5 file
    print(f"\nConverting {len(input_paths)} file(s)...")
    total_episodes = 0

    for hdf5_path, bddl_path in zip(input_paths, bddl_paths):
        count = process_one_hdf5(
            hdf5_path=hdf5_path,
            bddl_path=bddl_path,
            lerobot_dataset=lerobot_dataset,
            frame_skip=args.frame_skip,
            trim_zeros=args.trim_zeros,
            max_idle=args.max_idle,
            max_demos=args.max_demos,
        )
        total_episodes += count
        print(f"  -> {count} episodes added (cumulative: {total_episodes})")

    # Finalize
    print(f"\nFinalizing dataset...")
    lerobot_dataset.finalize()

    print(f"\nConversion complete!")
    print(f"  Episodes: {lerobot_dataset.meta.total_episodes}")
    print(f"  Frames: {lerobot_dataset.meta.total_frames}")
    print(f"  Tasks: {lerobot_dataset.meta.total_tasks}")
    print(f"  Output directory: {output_dir}")

    # Upload to HuggingFace
    if args.push:
        print(f"\nUploading to HuggingFace: {args.repo_id}")
        lerobot_dataset.push_to_hub(
            repo_id=args.repo_id,
            private=args.private,
        )
        print(f"Upload complete! https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
