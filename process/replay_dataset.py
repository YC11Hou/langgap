#!/usr/bin/env python3
"""Replay a dataset through the LIBERO environment and save agentview videos.

Pure action replay: loads actions from the dataset, executes them through env.step(),
and records agentview frames. Useful for verifying that dataset actions produce
the intended robot behavior.

Supports two input modes (auto-detected by file extension):
  - HDF5 file (*.hdf5 / *.h5): reads actions and initial states directly
  - LeRobot repo_id: loads from HuggingFace, REQUIRES --source_hdf5 for init states

IMPORTANT: When replaying LeRobot datasets, you MUST provide --source_hdf5.
Without initial states, env.reset() randomizes object positions and the replay
will fail because actions are computed for specific initial configurations.

Usage:
    export MUJOCO_GL=egl
    BDDL=data/bddl_files/libero_spatial/extended/dim3_change_object/ext_03_ramekin_to_cabinet.bddl
    HDF5=data/source/task50_edge_grasp.hdf5

    # Replay HDF5 directly (actions + init states from same file)
    python replay_dataset.py \
        --dataset $HDF5 \
        --bddl $BDDL --output_dir replay_videos/raw --episodes 3

    # Replay LeRobot dataset (action replay) - MUST use --source_hdf5
    python replay_dataset.py \
        --dataset <YOUR_HF_USERNAME>/task50_scripted \
        --bddl $BDDL --output_dir replay_videos/scripted --episodes 56 \
        --source_hdf5 $HDF5
"""

import argparse
from pathlib import Path

import imageio
import numpy as np

# from compare_datasets import load_dataset  # DISABLED: use HDF5 mode only


def create_env(bddl_path):
    """Create LIBERO OffScreenRenderEnv for replay."""
    from libero.libero.envs import OffScreenRenderEnv

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


def replay_episode(env, actions, frame_skip=2, init_state=None):
    """Replay actions through env.step() and collect agentview frames.

    Args:
        env: LIBERO OffScreenRenderEnv instance.
        actions: Array of actions to replay (10Hz from converted dataset).
        frame_skip: Sim steps per dataset action (default 2, 10Hz -> 20Hz env).
        init_state: Optional 92-dim MuJoCo state to restore initial conditions.
    """
    obs = env.reset()
    if init_state is not None:
        obs = env.set_init_state(init_state)
        # Reset controller state to match the restored MuJoCo state
        # This is crucial for deterministic replay - without this, the controller
        # targets remain from reset() and cause trajectory divergence
        robot = env.robots[0]
        controller = robot.controller
        controller.update_initial_joints(robot.init_qpos)
        env.sim.forward()
        controller.reset_goal()

    frames = [obs["agentview_image"][::-1].copy()]
    for action in actions:
        for _ in range(frame_skip):
            obs, _, _, _ = env.step(action.astype(np.float32))
        frames.append(obs["agentview_image"][::-1].copy())

    success = env.check_success()
    return frames, success


def save_video(frames, path, fps=10):
    """Save frames as MP4 video."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def main():
    parser = argparse.ArgumentParser(
        description="Replay a LeRobot dataset through LIBERO and save agentview videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", required=True,
                        help="HDF5 file (*.hdf5/*.h5) or LeRobot repo_id (e.g. <YOUR_HF_USERNAME>/task50_ext)")
    parser.add_argument("--bddl", required=True, help="BDDL file path for the task environment")
    parser.add_argument("--output_dir", required=True, help="Directory to save replay videos")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to replay (default: 3)")
    parser.add_argument("--task_index", type=int, default=None, help="Filter by task index")
    parser.add_argument("--fps", type=int, default=10, help="Video FPS (default: 10)")
    parser.add_argument("--source_hdf5", default=None, help="Source HDF5 for initial states (loads states[0] only)")
    parser.add_argument("--frame_skip", type=int, default=1, help="Sim steps per action (default: 1)")

    args = parser.parse_args()

    bddl_path = Path(args.bddl)
    if not bddl_path.exists():
        print(f"Error: BDDL file not found: {bddl_path}")
        return

    # Auto-detect input mode
    input_is_hdf5 = args.dataset.endswith(('.hdf5', '.h5'))

    if input_is_hdf5:
        import h5py

        hdf5_path = Path(args.dataset)
        if not hdf5_path.exists():
            print(f"Error: HDF5 file not found: {hdf5_path}")
            return
        with h5py.File(str(hdf5_path), 'r') as f:
            demo_keys = sorted([k for k in f['data'].keys() if k.startswith('demo')])
            episodes = []
            hdf5_init_states = []
            for dk in demo_keys:
                actions = f[f'data/{dk}/actions'][:]
                episodes.append({'actions': actions})
                hdf5_init_states.append(f[f'data/{dk}/states'][0])
        n_total = len(episodes)
        n_replay = min(args.episodes, n_total)
        print(f"Loading HDF5: {hdf5_path}")
        print(f"  {n_total} episodes, replaying {n_replay}")
    else:
        # LeRobot mode
        task_indices = [args.task_index] if args.task_index is not None else None
        print(f"Loading dataset: {args.dataset}")
        print("Error: LeRobot mode disabled. Use HDF5 file directly."); return
        episodes = dataset["episodes"]
        n_total = len(episodes)
        n_replay = min(args.episodes, n_total)
        print(f"  {n_total} episodes, replaying {n_replay}")

        # Load initial states from source HDF5 (LeRobot mode)
        hdf5_init_states = None
        if args.source_hdf5:
            import h5py

            hdf5_path = Path(args.source_hdf5)
            if not hdf5_path.exists():
                print(f"Error: Source HDF5 not found: {hdf5_path}")
                return
            with h5py.File(str(hdf5_path), "r") as f:
                # Use string sort to match convert script order
                demo_keys = sorted([k for k in f["data"].keys() if k.startswith("demo")])
                hdf5_init_states = [f[f"data/{dk}/states"][0] for dk in demo_keys]
            print(f"  Loaded {len(hdf5_init_states)} initial states from {hdf5_path}")
        else:
            print("  WARNING: No --source_hdf5 provided. Using random initial states from env.reset().")
            print("           Results will NOT be meaningful - actions are specific to initial states!")

    print("  Mode: action replay")

    # Create environment
    print(f"Creating environment from: {bddl_path}")
    env = create_env(str(bddl_path.absolute()))

    # Replay episodes
    output_dir = Path(args.output_dir)
    results = []
    for i in range(n_replay):
        ep = episodes[i]
        actions = ep["actions"]
        init_state = hdf5_init_states[i] if hdf5_init_states else None
        print(f"  Replaying episode {i} ({len(actions)} actions)...")
        frames, success = replay_episode(env, actions, args.frame_skip, init_state)
        results.append(success)

        video_path = output_dir / f"episode_{i:03d}.mp4"
        save_video(frames, video_path, fps=args.fps)
        status = "SUCCESS" if success else "FAIL"
        print(f"    {status} | Saved: {video_path} ({len(frames)} frames)")

    env.close()
    n_success = sum(results)
    print(f"\nDone. {n_replay} videos saved to {output_dir}/")
    print(f"Success: {n_success}/{n_replay}")


if __name__ == "__main__":
    main()
