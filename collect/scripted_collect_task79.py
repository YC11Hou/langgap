#!/usr/bin/env python3
"""
Scripted waypoint policy for automated demo collection.

Uses a sequence of waypoints to pick up the cream cheese and place it on the plate.
Based on scripted_collect_task51.py (Task 51: cookie box → plate), modified for Task 79
(cream_cheese_to_plate, libero_goal). Note: previously misnamed as task64.
Key difference: cream cheese is a tall narrow box (1.8×4.3×8.1cm).
Action computation follows MimicGen's target_pose_to_action approach.
Output HDF5 format is compatible with collect_for_mimicgen.py.
"""

import argparse
import json
import re
import sys
import h5py
import numpy as np
from datetime import datetime
from pathlib import Path

import robosuite as suite
from robosuite import load_controller_config
import robosuite.utils.transform_utils as T

import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import TASK_MAPPING


# ========== Pose utilities (MimicGen approach) ==========

def make_pose(pos, rot):
    """Create 4x4 pose matrix from position and rotation matrix."""
    pose = np.zeros((4, 4))
    pose[:3, :3] = rot
    pose[:3, 3] = pos
    pose[3, 3] = 1.0
    return pose


def unmake_pose(pose):
    """Extract position and rotation matrix from 4x4 pose."""
    return pose[:3, 3].copy(), pose[:3, :3].copy()


def get_eef_pose(env):
    """Get current EEF 4x4 pose matrix."""
    eef_site = env.robots[0].controller.eef_name
    site_id = env.sim.model.site_name2id(eef_site)
    pos = np.array(env.sim.data.site_xpos[site_id])
    rot = np.array(env.sim.data.site_xmat[site_id].reshape(3, 3))
    return make_pose(pos, rot)


# Fixed gripper rotation for top-down grasp.
# Columns: X=[0,1,0] (fingers close along world +Y, spanning box short axis),
# Y=[1,0,0], Z=[0,0,-1] (down). Matches initial EEF orientation (≈270° approach).
GRASP_ROT = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)


def target_pose_to_action(env, target_pose):
    """
    Convert target 4x4 pose to 6-dim action (MimicGen approach).

    Uses axis-angle representation for rotation delta, compatible with OSC_POSE.
    """
    curr_pose = get_eef_pose(env)
    curr_pos, curr_rot = unmake_pose(curr_pose)
    target_pos, target_rot = unmake_pose(target_pose)

    max_dpos = env.robots[0].controller.output_max[0]
    max_drot = env.robots[0].controller.output_max[3]

    # Position delta
    delta_pos = np.clip((target_pos - curr_pos) / max_dpos, -1., 1.)

    # Rotation delta (axis-angle)
    delta_rot_mat = target_rot @ curr_rot.T
    delta_quat = T.mat2quat(delta_rot_mat)
    delta_rotation = T.quat2axisangle(delta_quat)
    delta_rotation = np.clip(delta_rotation / max_drot, -1., 1.)

    return np.concatenate([delta_pos, delta_rotation])


def get_instruction_from_bddl(bddl_path: str) -> str:
    with open(bddl_path, "r") as f:
        content = f.read()
    match = re.search(r"\(:language ([^)]+)\)", content)
    if match:
        return match.group(1).strip()
    return "Unknown instruction"


def discover_names(env, debug=False):
    """Find MuJoCo body names for cream cheese and plate."""
    body_names = [env.sim.model.body_id2name(i)
                  for i in range(env.sim.model.nbody)]

    if debug:
        print("\n=== MuJoCo Body Names ===")
        for n in sorted(body_names):
            print(f"  {n}")

    # Find cream cheese body
    cheese_body = None
    for n in body_names:
        if "cream_cheese" in n.lower():
            cheese_body = n
            break

    # Find plate body
    plate_body = None
    for n in body_names:
        if "plate" in n.lower():
            plate_body = n
            break

    if cheese_body is None:
        raise RuntimeError(f"Could not find cream cheese body in: {body_names}")
    if plate_body is None:
        raise RuntimeError(f"Could not find plate body in: {body_names}")

    return cheese_body, plate_body


def get_body_pos(env, body_name):
    body_id = env.sim.model.body_name2id(body_name)
    return np.array(env.sim.data.body_xpos[body_id]).copy()


def get_site_pos(env, site_name):
    site_id = env.sim.model.site_name2id(site_name)
    return np.array(env.sim.data.site_xpos[site_id]).copy()


def compute_action(env, target_pose, gripper):
    """Compute 7-dim action from target pose and gripper state."""
    arm_action = target_pose_to_action(env, target_pose)
    return np.concatenate([arm_action, [gripper]])



def build_waypoints(object_pos, plate_pos, table_height,
                    place_offset=0.02, grasp_offset=0.01):
    """
    Build waypoint sequence with full 4x4 pose matrices.

    Center top-down grasp strategy: gripper descends straight onto the cream cheese
    with fixed orientation (GRASP_ROT matching initial EEF pose).

    Each waypoint: (target_pose_4x4, gripper, dist_threshold, hold_steps)
        - target_pose_4x4: 4x4 homogeneous transformation matrix
        - gripper: -1 open, +1 close (robosuite PandaGripper convention)
        - dist_threshold: distance to target to consider waypoint reached
        - hold_steps: extra steps to hold at this position (for gripper actions)

    Args:
        object_pos: cream cheese center position
        plate_pos: plate center position
        grasp_offset: height offset above object body center for grasping
    """
    ox, oy, oz = object_pos
    px, py, pz = plate_pos

    # Center grasp: gripper goes to object center XY
    grasp_x = ox
    grasp_y = oy
    grasp_z = oz + grasp_offset

    # Place: EEF goes directly to plate center (no edge compensation needed)
    place_x = px
    place_y = py

    waypoints = [
        # 1. Move above cream cheese center + rotate gripper (gripper open)
        (make_pose(np.array([grasp_x, grasp_y, grasp_z + 0.10]), GRASP_ROT), -1, 0.01, 0),
        # 2. Descend to grasp position (gripper open, tight threshold)
        (make_pose(np.array([grasp_x, grasp_y, grasp_z]), GRASP_ROT), -1, 0.005, 0),
        # 3. Close gripper (extra hold)
        (make_pose(np.array([grasp_x, grasp_y, grasp_z]), GRASP_ROT), +1, 999, 10),
        # 4. Lift up high (gripper closed) — higher to avoid overshooting during carry
        (make_pose(np.array([grasp_x, grasp_y, grasp_z + 0.18]), GRASP_ROT), +1, 0.01, 0),
        # 5. Move above plate center (higher approach for convergence)
        (make_pose(np.array([place_x, place_y, pz + 0.18]), GRASP_ROT), +1, 0.015, 0),
        # 6. Descend to place position (gripper closed)
        (make_pose(np.array([place_x, place_y, pz + place_offset]), GRASP_ROT), +1, 0.015, 0),
        # 7. Open gripper (release, hold 5 steps)
        (make_pose(np.array([place_x, place_y, pz + place_offset]), GRASP_ROT), -1, 999, 5),
    ]
    return waypoints


def collect_one_demo(env, cheese_body, plate_body, max_steps, debug=False,
                     place_offset=0.02):
    """Collect a single demo using scripted waypoints. Returns (states, actions, success)."""
    obs = env.reset()

    states = []
    actions = []

    # Let physics settle (objects may be spawned slightly above surfaces)
    # Do NOT record settling steps — they cause visible pauses in playback
    settle_action = np.zeros(7)
    settle_action[6] = -1.0  # gripper open during settling
    for _ in range(10):
        obs, _, _, _ = env.step(settle_action)

    # Get current object positions after settling
    object_pos = get_body_pos(env, cheese_body)
    plate_pos = get_body_pos(env, plate_body)

    # Get table height
    table_top_pos = get_site_pos(env, "table_top")
    table_height = table_top_pos[2]

    if debug:
        eef_pose = get_eef_pose(env)
        eef_pos, _ = unmake_pose(eef_pose)
        # Also check cream cheese default_site position (object center after settling)
        cheese_site_pos = get_site_pos(env, "cream_cheese_1_default_site")
        print(f"\n  EEF position:         {eef_pos}")
        print(f"  Cream cheese body:    {object_pos}")
        print(f"  Cream cheese site:    {cheese_site_pos}")
        print(f"  Plate pos:            {plate_pos}")
        print(f"  Cream cheese Z (body): {object_pos[2]:.4f}")
        print(f"  Cream cheese Z (site): {cheese_site_pos[2]:.4f}")
        print(f"  Plate Z:              {plate_pos[2]:.4f}")
        print(f"  Table top Z:          {table_height:.4f}")

    waypoints = build_waypoints(object_pos, plate_pos, table_height,
                                place_offset=place_offset)

    total_steps = len(actions)  # account for settling steps

    for wp_idx, (target_pose, gripper, dist_thresh, hold_steps) in enumerate(waypoints):
        target_pos, _ = unmake_pose(target_pose)
        if debug:
            print(f"  Waypoint {wp_idx + 1}: target={target_pos}, "
                  f"gripper={'close' if gripper > 0 else 'open'}, "
                  f"hold={hold_steps}")

        steps_at_wp = 0
        held = 0

        while total_steps < max_steps:
            state = env.sim.get_state().flatten()
            action = compute_action(env, target_pose, gripper)
            obs, reward, done, info = env.step(action)

            states.append(state)
            actions.append(action)
            total_steps += 1
            steps_at_wp += 1

            eef_pose = get_eef_pose(env)
            eef_pos, _ = unmake_pose(eef_pose)
            dist = np.linalg.norm(eef_pos - target_pos)

            # For hold waypoints (gripper open/close), just count steps
            if hold_steps > 0:
                held += 1
                if held >= hold_steps:
                    break
            else:
                if dist < dist_thresh:
                    break

            # Safety: don't spend too long on one waypoint
            if steps_at_wp > 150:
                if debug:
                    print(f"    Waypoint {wp_idx + 1} timeout at step {total_steps}, "
                          f"dist={dist:.4f}")
                break

        if debug:
            eef_pose_after = get_eef_pose(env)
            eef_after, _ = unmake_pose(eef_pose_after)
            cheese_after = get_body_pos(env, cheese_body)
            print(f"    WP{wp_idx+1} done: eef={eef_after}, "
                  f"cheese={cheese_after}, steps={steps_at_wp}")

        if total_steps >= max_steps:
            if debug:
                print(f"  Max steps reached at waypoint {wp_idx + 1}")
            break

    # Remove last state to align states/actions (states[i] -> action[i] -> states[i+1])
    if len(states) > len(actions):
        del states[-1]
    assert len(states) == len(actions), \
        f"states ({len(states)}) != actions ({len(actions)})"

    success = env._check_success()
    if debug:
        print(f"  Total steps: {total_steps}, Success: {success}")

    return states, actions, success


def save_hdf5(demos, output_path, env_info, bddl_path, instruction):
    """Save collected demos to HDF5 format."""
    with h5py.File(output_path, "w") as f:
        grp = f.create_group("data")

        for i, (states, actions) in enumerate(demos):
            ep_grp = grp.create_group(f"demo_{i + 1}")
            ep_grp.create_dataset("states", data=np.array(states))
            ep_grp.create_dataset("actions", data=np.array(actions))
            ep_grp.attrs["num_samples"] = len(actions)

        now = datetime.now()
        grp.attrs["date"] = f"{now.month}-{now.day}-{now.year}"
        grp.attrs["time"] = f"{now.hour}:{now.minute}:{now.second}"
        grp.attrs["repository_version"] = suite.__version__
        grp.attrs["env"] = "Libero_Tabletop_Manipulation"
        grp.attrs["env_info"] = env_info
        grp.attrs["bddl_file"] = bddl_path
        grp.attrs["instruction"] = instruction
        grp.attrs["total"] = len(demos)


EXPECTED_MAX_STEPS = 95  # Calibrated: max=86, ceil/5×5+5=95

def main():
    parser = argparse.ArgumentParser(
        description="Scripted waypoint demo collection (Task 64: cream cheese → plate)")
    parser.add_argument("--bddl", type=str, required=True,
                        help="BDDL file path")
    parser.add_argument("--num", type=int, default=200,
                        help="Max attempts (upper bound)")
    parser.add_argument("--target", type=int, default=50,
                        help="Stop after this many successes")
    parser.add_argument("--output", type=str, default=None,
                        help="Output HDF5 file path")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Max steps per demo")
    parser.add_argument("--place_offset", type=float, default=0.02,
                        help="Z offset above plate for place (tune this)")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: 1 demo, print all names/coords")
    args = parser.parse_args()

    # Resolve BDDL path
    bddl_path = Path(args.bddl)
    if not bddl_path.exists():
        bddl_path = Path(__file__).parent.parent / args.bddl
    if not bddl_path.exists():
        print(f"Error: BDDL file not found: {args.bddl}")
        sys.exit(1)

    bddl_path = str(bddl_path.absolute())
    problem_info = BDDLUtils.get_problem_info(bddl_path)
    problem_name = problem_info["problem_name"]
    instruction = get_instruction_from_bddl(bddl_path)

    num_demos = 1 if args.debug else args.num
    target = 1 if args.debug else args.target

    print("=" * 60)
    print("Scripted Waypoint Demo Collection (Task 64)")
    print("=" * 60)
    print(f"Task: {instruction}")
    print(f"Max attempts: {num_demos}")
    print(f"Target successes: {target}")
    print(f"Max steps/demo: {args.max_steps}")
    print(f"Place offset: {args.place_offset}")
    print(f"Debug: {args.debug}")
    print("=" * 60)

    # Create environment
    print("\nCreating environment...")
    controller_config = load_controller_config(default_controller="OSC_POSE")
    env = TASK_MAPPING[problem_name](
        bddl_file_name=bddl_path,
        robots=["Panda"],
        controller_configs=[controller_config],
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    # First reset to discover MuJoCo names
    print("Discovering object names...")
    env.reset()
    cheese_body, plate_body = discover_names(env, debug=args.debug)
    print(f"  Cream cheese body: {cheese_body}")
    print(f"  Plate body:        {plate_body}")

    # Collect demos
    demos = []
    successes = 0
    failures = 0

    for i in range(num_demos):
        print(f"\n--- Demo {i + 1}/{num_demos} ---")
        states, actions, success = collect_one_demo(
            env, cheese_body, plate_body,
            max_steps=args.max_steps,
            debug=args.debug,
            place_offset=args.place_offset,
        )

        if success:
            if len(actions) <= EXPECTED_MAX_STEPS:
                demos.append((states, actions))
                successes += 1
                print(f"  SUCCESS ({len(actions)} steps) "
                      f"[{successes}/{i + 1} successful]")
                if successes >= target:
                    print(f"\n  Reached target of {target} successes, stopping early.")
                    break
            else:
                print(f"  DISCARDED ({len(actions)} steps > {EXPECTED_MAX_STEPS})")
        else:
            failures += 1
            print(f"  FAILED ({len(actions)} steps) "
                  f"[{successes}/{i + 1} successful]")

    env.close()

    print(f"\n{'=' * 60}")
    print(f"Collection complete: {successes}/{num_demos} successful")
    print(f"{'=' * 60}")

    if not demos:
        print("No demos to save.")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_dir = Path(__file__).parent.parent / "data" / "source"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_name = Path(args.bddl).stem
        output_path = str(output_dir / f"{task_name}_scripted_{timestamp}.hdf5")

    env_info = json.dumps({
        "env_name": problem_name,
        "type": 1,
        "env_kwargs": {
            "bddl_file_name": bddl_path,
            "robots": ["Panda"],
        }
    })

    save_hdf5(demos, output_path, env_info, bddl_path, instruction)
    print(f"Saved {len(demos)} demos to: {output_path}")


if __name__ == "__main__":
    main()
