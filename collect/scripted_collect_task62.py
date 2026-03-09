#!/usr/bin/env python3
"""
Scripted waypoint policy for automated demo collection.

Task 62: pick up the bbq sauce and place it in the basket.
Suite: libero_object (LIBERO_Floor_Manipulation — objects on floor, not table).
Strategy: center top-down grasp with fixed GRASP_ROT orientation.
Target: basket contain_region.
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
    pose = np.zeros((4, 4))
    pose[:3, :3] = rot
    pose[:3, 3] = pos
    pose[3, 3] = 1.0
    return pose


def unmake_pose(pose):
    return pose[:3, 3].copy(), pose[:3, :3].copy()


def get_eef_pose(env):
    eef_site = env.robots[0].controller.eef_name
    site_id = env.sim.model.site_name2id(eef_site)
    pos = np.array(env.sim.data.site_xpos[site_id])
    rot = np.array(env.sim.data.site_xmat[site_id].reshape(3, 3))
    return make_pose(pos, rot)


# Fixed gripper rotation for top-down grasp.
GRASP_ROT = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)


def target_pose_to_action(env, target_pose):
    curr_pose = get_eef_pose(env)
    curr_pos, curr_rot = unmake_pose(curr_pose)
    target_pos, target_rot = unmake_pose(target_pose)

    max_dpos = env.robots[0].controller.output_max[0]
    max_drot = env.robots[0].controller.output_max[3]

    delta_pos = np.clip((target_pos - curr_pos) / max_dpos, -1., 1.)

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


# Object name for this task
OBJECT_KEYWORD = "bbq_sauce"
TASK_LABEL = "Task 62"


def discover_names(env, debug=False):
    """Find MuJoCo body name for target object and basket site."""
    body_names = [env.sim.model.body_id2name(i)
                  for i in range(env.sim.model.nbody)]
    site_names = [env.sim.model.site_id2name(i)
                  for i in range(env.sim.model.nsite)]

    if debug:
        print("\n=== MuJoCo Body Names ===")
        for n in sorted(body_names):
            print(f"  {n}")
        print(f"\n=== MuJoCo Site Names ===")
        for n in sorted(site_names):
            print(f"  {n}")

    object_body = None
    for n in body_names:
        if OBJECT_KEYWORD in n.lower():
            object_body = n
            break

    # Find basket body (for position reading)
    basket_body = None
    for n in body_names:
        if "basket" in n.lower():
            basket_body = n
            break

    # Find basket contain_region site (for placement target)
    basket_site = None
    for n in site_names:
        if "basket" in n.lower() and "contain" in n.lower():
            basket_site = n
            break
    if basket_site is None:
        for n in site_names:
            if "basket" in n.lower() and "default" in n.lower():
                basket_site = n
                break

    if object_body is None:
        raise RuntimeError(f"Could not find {OBJECT_KEYWORD} body in: {body_names}")
    if basket_body is None:
        raise RuntimeError(f"Could not find basket body in: {body_names}")

    return object_body, basket_body, basket_site


def get_body_pos(env, body_name):
    body_id = env.sim.model.body_name2id(body_name)
    return np.array(env.sim.data.body_xpos[body_id]).copy()


def get_site_pos(env, site_name):
    site_id = env.sim.model.site_name2id(site_name)
    return np.array(env.sim.data.site_xpos[site_id]).copy()


def compute_action(env, target_pose, gripper):
    arm_action = target_pose_to_action(env, target_pose)
    return np.concatenate([arm_action, [gripper]])


def build_waypoints(object_pos, basket_pos, place_offset=0.02, grasp_offset=0.01):
    """
    Build waypoint sequence for center top-down grasp, place in basket.

    Args:
        object_pos: target object center position
        basket_pos: basket center position (body pos or contain_region site)
        grasp_offset: height offset above object body center for grasping
        place_offset: height offset above basket for release
    """
    ox, oy, oz = object_pos
    bx, by, bz = basket_pos

    grasp_x = ox
    grasp_y = oy
    grasp_z = oz + grasp_offset

    place_x = bx
    place_y = by

    waypoints = [
        # 1. Move above object center (gripper open)
        (make_pose(np.array([grasp_x, grasp_y, grasp_z + 0.10]), GRASP_ROT), -1, 0.01, 0),
        # 2. Descend to grasp position (gripper open, relaxed for floor-level reach limit)
        (make_pose(np.array([grasp_x, grasp_y, grasp_z]), GRASP_ROT), -1, 0.025, 0),
        # 3. Close gripper (hold 10 steps)
        (make_pose(np.array([grasp_x, grasp_y, grasp_z]), GRASP_ROT), +1, 999, 10),
        # 4. Lift high (gripper closed) — high altitude for workspace freedom
        (make_pose(np.array([grasp_x, grasp_y, 0.30]), GRASP_ROT), +1, 0.01, 0),
        # 5. Transit to above basket at high altitude (arm has more freedom here)
        (make_pose(np.array([place_x, place_y, 0.30]), GRASP_ROT), +1, 0.01, 0),
        # 6. Descend directly above basket to release height
        (make_pose(np.array([place_x, place_y, bz + 0.12]), GRASP_ROT), +1, 0.01, 0),
        # 7. Stabilize over basket (hold 10 steps — object centers under EEF)
        (make_pose(np.array([place_x, place_y, bz + 0.12]), GRASP_ROT), +1, 999, 10),
        # 8. Open gripper (release into basket, hold 25 steps — gravity settle)
        (make_pose(np.array([place_x, place_y, bz + 0.12]), GRASP_ROT), -1, 999, 25),
    ]
    return waypoints


def collect_one_demo(env, object_body, basket_body, basket_site, max_steps,
                     debug=False, place_offset=0.02):
    """Collect a single demo using scripted waypoints. Returns (states, actions, success)."""
    obs = env.reset()

    states = []
    actions = []

    # Let physics settle — not recorded
    settle_action = np.zeros(7)
    settle_action[6] = -1.0
    for _ in range(10):
        obs, _, _, _ = env.step(settle_action)

    object_pos = get_body_pos(env, object_body)
    # Use basket site (contain_region) if available, else body pos
    if basket_site:
        basket_pos = get_site_pos(env, basket_site)
    else:
        basket_pos = get_body_pos(env, basket_body)

    if debug:
        eef_pose = get_eef_pose(env)
        eef_pos, _ = unmake_pose(eef_pose)
        print(f"\n  EEF position:         {eef_pos}")
        print(f"  Object body:          {object_pos}")
        print(f"  Basket pos:           {basket_pos}")
        print(f"  Object Z:             {object_pos[2]:.4f}")
        print(f"  Basket Z:             {basket_pos[2]:.4f}")

    waypoints = build_waypoints(object_pos, basket_pos, place_offset=place_offset)

    total_steps = 0

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

            if hold_steps > 0:
                held += 1
                if held >= hold_steps:
                    break
            else:
                if dist < dist_thresh:
                    break

            if steps_at_wp > 300:
                if debug:
                    print(f"    Waypoint {wp_idx + 1} timeout at step {total_steps}, "
                          f"dist={dist:.4f}")
                break

        if debug:
            eef_pose_after = get_eef_pose(env)
            eef_after, _ = unmake_pose(eef_pose_after)
            obj_after = get_body_pos(env, object_body)
            print(f"    WP{wp_idx+1} done: eef={eef_after}, "
                  f"object={obj_after}, steps={steps_at_wp}")

        if total_steps >= max_steps:
            if debug:
                print(f"  Max steps reached at waypoint {wp_idx + 1}")
            break

    if len(states) > len(actions):
        del states[-1]
    assert len(states) == len(actions), \
        f"states ({len(states)}) != actions ({len(actions)})"

    success = env._check_success()
    if debug:
        print(f"  Total steps: {total_steps}, Success: {success}")

    return states, actions, success


def save_hdf5(demos, output_path, env_info, bddl_path, instruction):
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
        grp.attrs["env"] = "Libero_Floor_Manipulation"
        grp.attrs["env_info"] = env_info
        grp.attrs["bddl_file"] = bddl_path
        grp.attrs["instruction"] = instruction
        grp.attrs["total"] = len(demos)


EXPECTED_MAX_STEPS = 140  # Calibrated: max=134, ceil/5×5+5=140

def main():
    parser = argparse.ArgumentParser(
        description=f"Scripted waypoint demo collection ({TASK_LABEL}: {OBJECT_KEYWORD} → basket)")
    parser.add_argument("--bddl", type=str, required=True)
    parser.add_argument("--num", type=int, default=200, help="Max attempts")
    parser.add_argument("--target", type=int, default=50, help="Stop after N successes")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--place_offset", type=float, default=0.02)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

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
    print(f"Scripted Waypoint Demo Collection ({TASK_LABEL})")
    print("=" * 60)
    print(f"Task: {instruction}")
    print(f"Max attempts: {num_demos}")
    print(f"Target successes: {target}")
    print(f"Max steps/demo: {args.max_steps}")
    print(f"Place offset: {args.place_offset}")
    print(f"Debug: {args.debug}")
    print("=" * 60)

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

    print("Discovering object names...")
    env.reset()
    object_body, basket_body, basket_site = discover_names(env, debug=args.debug)
    print(f"  Object body:  {object_body}")
    print(f"  Basket body:  {basket_body}")
    print(f"  Basket site:  {basket_site}")

    demos = []
    successes = 0
    failures = 0

    for i in range(num_demos):
        print(f"\n--- Demo {i + 1}/{num_demos} ---")
        states, actions, success = collect_one_demo(
            env, object_body, basket_body, basket_site,
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
