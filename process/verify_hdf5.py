#!/usr/bin/env python3
"""Fast HDF5 action replay verification (no rendering).

Replays actions through env.step() and checks success, without creating an
offscreen renderer. Much faster than replay_dataset.py for bulk validation.

Output JSON format is compatible with replay_with_results.py and
filter_failed_demos.py.

Usage:
    export MUJOCO_GL=egl

    # Verify a single HDF5 file
    python process/verify_hdf5.py \
        --hdf5 data/source/edge_grasp/task_40.hdf5 \
        --task_id 40 \
        --output_json /tmp/verify_40.json

    # Exit code: 0 = all passed, 1 = some failed
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from robosuite import load_controller_config
import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import TASK_MAPPING

# Import TASK_CONFIG from the collection script
sys.path.insert(0, str(Path(__file__).parent.parent / "collect"))
from scripted_collect_edge_grasp import TASK_CONFIG, BDDL_ROOT


def create_env_no_render(bddl_path):
    """Create LIBERO env without offscreen renderer (fast)."""
    problem_info = BDDLUtils.get_problem_info(bddl_path)
    problem_name = problem_info["problem_name"]

    controller_config = load_controller_config(default_controller="OSC_POSE")
    env = TASK_MAPPING[problem_name](
        bddl_file_name=bddl_path,
        robots=["Panda"],
        controller_configs=[controller_config],
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )
    return env


def verify_episode(env, init_state, actions):
    """Replay actions from init_state and return success status.

    Mirrors the controller reset logic from replay_dataset.py to ensure
    deterministic replay.
    """
    env.reset()
    # Restore MuJoCo state directly (env is raw robosuite, no wrapper)
    env.sim.set_state_from_flattened(init_state)
    env.sim.forward()

    # Reset controller state (crucial for deterministic replay)
    robot = env.robots[0]
    controller = robot.controller
    controller.update_initial_joints(robot.init_qpos)
    env.sim.forward()
    controller.reset_goal()

    for action in actions:
        env.step(action.astype(np.float32))

    return env._check_success()


def verify_hdf5(hdf5_path, task_id, output_json=None):
    """Verify all demos in an HDF5 file.

    Returns list of {"episode": int, "success": bool} dicts.
    """
    task_cfg = TASK_CONFIG[task_id]
    bddl_path = str((BDDL_ROOT / task_cfg["bddl"]).absolute())

    if not Path(bddl_path).exists():
        print(f"Error: BDDL file not found: {bddl_path}")
        sys.exit(1)

    # Load demos
    with h5py.File(str(hdf5_path), "r") as f:
        demo_keys = sorted([k for k in f["data"].keys() if k.startswith("demo")])
        n_demos = len(demo_keys)
        print(f"Verifying {hdf5_path}: {n_demos} demos (task {task_id})")

        demos = []
        for dk in demo_keys:
            init_state = f[f"data/{dk}/states"][0]
            actions = f[f"data/{dk}/actions"][:]
            demos.append((init_state, actions))

    # Create env
    env = create_env_no_render(bddl_path)

    # Verify each demo
    results = []
    n_success = 0
    for i, (init_state, actions) in enumerate(demos):
        success = verify_episode(env, init_state, actions)
        results.append({"episode": i, "success": bool(success)})
        if success:
            n_success += 1
        status = "OK" if success else "FAIL"
        print(f"  demo {i+1}/{n_demos}: {status} ({len(actions)} steps)")

    env.close()

    print(f"Result: {n_success}/{n_demos} passed")

    # Save JSON
    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump({"task_id": task_id, "results": results}, f, indent=2)
        print(f"Saved: {output_json}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Fast HDF5 action replay verification (no rendering)")
    parser.add_argument("--hdf5", required=True,
                        help="HDF5 file to verify")
    parser.add_argument("--task_id", type=int, required=True,
                        choices=list(TASK_CONFIG.keys()),
                        help="Task ID (for BDDL lookup)")
    parser.add_argument("--output_json", default=None,
                        help="Output JSON path (optional)")
    args = parser.parse_args()

    results = verify_hdf5(args.hdf5, args.task_id, args.output_json)

    n_failed = sum(1 for r in results if not r["success"])
    sys.exit(1 if n_failed > 0 else 0)


if __name__ == "__main__":
    main()
