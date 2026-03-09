#!/usr/bin/env python3
"""
Scripted waypoint policy for automated demo collection.

Task 45: pick up the black bowl from the middle of the table and place it on the ramekin.
Strategy: edge grasp on bowl rim with automatic direction selection (avoids nearby obstacles).
  Prefers Y-axis (no rotation), falls back to X-axis (90 Z rotation) if blocked.
Tuned params: bowl_radius=0.048, grasp_offset=0.03, place_offset=0.02.
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
    """Find MuJoCo body name for akita_black_bowl_1 and ramekin default site."""
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

    # Find akita_black_bowl_1 body
    bowl_body = None
    for n in body_names:
        if "akita_black_bowl_1" in n.lower():
            bowl_body = n
            break

    # Find ramekin default site
    ramekin_site = None
    for n in site_names:
        if "ramekin" in n.lower() and "default" in n.lower():
            ramekin_site = n
            break

    # Fallback: search for ramekin site
    if ramekin_site is None:
        for n in site_names:
            if "ramekin" in n.lower():
                ramekin_site = n
                break

    if bowl_body is None:
        raise RuntimeError(f"Could not find akita_black_bowl_1 body in: {body_names}")
    if ramekin_site is None:
        raise RuntimeError(f"Could not find ramekin site in: {site_names}")

    return bowl_body, ramekin_site


def choose_grasp_direction(env, bowl_pos, bowl_radius=0.048, debug=False):
    """
    Choose safe grasp direction from {+Y, -Y, +X, -X}.

    Detection: obstacles within bowl_radius*2 of grasp point block that direction.
    Selection: Y-axis first (no rotation), then closest to robot.
    """
    detection_radius = bowl_radius * 2  # one diameter

    # Gather obstacle positions (table-height bodies, excluding bowl_1/robot/fixtures)
    obstacle_positions = []
    for i in range(env.sim.model.nbody):
        name = env.sim.model.body_id2name(i)
        if any(skip in name.lower() for skip in
               ['robot', 'gripper', 'mount', 'world', 'table',
                'cabinet', 'stove_1_base', 'stove_1_main']):
            continue
        if 'akita_black_bowl_1' in name.lower():
            continue
        pos = env.sim.data.body_xpos[i]
        if pos[2] > 0.85:
            obstacle_positions.append(pos[:2])

    bx, by = bowl_pos[0], bowl_pos[1]
    r = bowl_radius

    candidates = [
        (0, +r, False, "+Y"),
        (0, -r, False, "-Y"),
        (+r, 0, True, "+X"),
        (-r, 0, True, "-X"),
    ]

    safe = []
    for dx, dy, needs_rot, label in candidates:
        rim_x, rim_y = bx + dx, by + dy
        blocked = any(
            np.sqrt((rim_x - ox)**2 + (rim_y - oy)**2) < detection_radius
            for ox, oy in obstacle_positions
        )
        if blocked:
            if debug:
                print(f"    {label}: BLOCKED")
            continue
        dist_to_robot = np.sqrt(rim_x**2 + rim_y**2)
        safe.append((dx, dy, needs_rot, label, dist_to_robot))
        if debug:
            print(f"    {label}: safe, dist_to_robot={dist_to_robot:.4f}")

    safe.sort(key=lambda x: (x[2], x[4]))  # Y-axis first, then closest to robot

    if not safe:
        if debug:
            print("    WARNING: No safe direction, fallback -Y")
        return 0, -r, False, "-Y"

    chosen = safe[0]
    if debug:
        print(f"    Chosen: {chosen[3]}")
    return chosen[0], chosen[1], chosen[2], chosen[3]


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


def build_grasp_waypoints(bowl_pos, init_rot, rim_offset_x=0, rim_offset_y=None,
                          bowl_radius=0.048, grasp_offset=0.03, needs_rotation=False):
    """
    Build grasp waypoints (1-4): approach, descend, close, lift.

    rim_offset_x/y: offset from bowl center to rim grasp point.
    needs_rotation: if True, rotate init_rot 90 around Z for X-axis grasps.
    Each waypoint: (target_pose_4x4, gripper, dist_threshold, hold_steps)
    """
    bx, by, bz = bowl_pos
    if rim_offset_y is None:
        rim_offset_y = -bowl_radius  # default: -Y for backwards compat
    rim_x = bx + rim_offset_x
    rim_y = by + rim_offset_y
    grasp_z = bz + grasp_offset

    if needs_rotation:
        # Rotate 90 around Z axis for X-axis grasps
        Rz_90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        rot = Rz_90 @ init_rot
    else:
        rot = init_rot

    return [
        # 1. Move above bowl rim (gripper open)
        (make_pose(np.array([rim_x, rim_y, grasp_z + 0.10]), rot), -1, 0.01, 0),
        # 2. Descend to grasp position (gripper open)
        (make_pose(np.array([rim_x, rim_y, grasp_z]), rot), -1, 0.01, 0),
        # 3. Close gripper (hold 10 steps for secure grip)
        (make_pose(np.array([rim_x, rim_y, grasp_z]), rot), +1, 999, 10),
        # 4. Lift up (gripper closed)
        (make_pose(np.array([rim_x, rim_y, grasp_z + 0.18]), rot), +1, 0.01, 0),
    ]


def build_place_waypoints(place_xy, target_z, init_rot, place_offset=0.02):
    """
    Build place waypoints (5-8): move above, descend, open, lift.

    place_xy is adjusted so the bowl center (not gripper) lands on ramekin center.
    Each waypoint: (target_pose_4x4, gripper, dist_threshold, hold_steps)
    """
    px, py = place_xy
    sz = target_z
    rot = init_rot

    return [
        # 5. Move above ramekin (gripper closed) - relaxed threshold for long travel
        (make_pose(np.array([px, py, sz + 0.10]), rot), +1, 0.03, 0),
        # 6. Descend to place position (gripper closed)
        (make_pose(np.array([px, py, sz + place_offset]), rot), +1, 0.02, 0),
        # 7. Open gripper (release, hold 5 steps)
        (make_pose(np.array([px, py, sz + place_offset]), rot), -1, 999, 5),
    ]


def collect_one_demo(env, bowl_body, ramekin_site, max_steps, debug=False,
                     place_offset=0.02, bowl_radius=0.048, grasp_offset=0.03):
    """Collect a single demo using scripted waypoints. Returns (states, actions, success)."""
    obs = env.reset()

    states = []
    actions = []

    # Let physics settle (10 steps sufficient per MEMORY.md; not recorded as training data)
    settle_action = np.zeros(7)
    settle_action[6] = -1.0  # gripper open during settling
    for _ in range(10):
        obs, _, _, _ = env.step(settle_action)

    # Get current object positions and EEF orientation after settling
    bowl_pos = get_body_pos(env, bowl_body)
    ramekin_pos = get_site_pos(env, ramekin_site)
    init_rot = np.array(env.robots[0].controller.ee_ori_mat).copy()

    # Choose grasp direction
    rim_dx, rim_dy, needs_rot, dir_label = choose_grasp_direction(
        env, bowl_pos, bowl_radius=bowl_radius, debug=debug)

    if debug:
        eef_pose = get_eef_pose(env)
        eef_pos, _ = unmake_pose(eef_pose)
        bowl_site_pos = get_site_pos(env, "akita_black_bowl_1_default_site")
        table_top_pos = get_site_pos(env, "table_top")
        print(f"\n  EEF position:         {eef_pos}")
        print(f"  EEF init rotation:\n{init_rot}")
        print(f"  Bowl body pos:        {bowl_pos}")
        print(f"  Bowl site (dflt):     {bowl_site_pos}")
        print(f"  Ramekin pos:          {ramekin_pos}")
        print(f"  Table top Z:          {table_top_pos[2]:.4f}")
        print(f"  Bowl Z (body):        {bowl_pos[2]:.4f}")
        print(f"  Bowl Z (site):        {bowl_site_pos[2]:.4f}")
        print(f"  Ramekin Z:            {ramekin_pos[2]:.4f}")
        print(f"  Bowl radius:          {bowl_radius}")
        print(f"  Grasp offset:         {grasp_offset}")
        rim_x = bowl_pos[0] + rim_dx
        rim_y = bowl_pos[1] + rim_dy
        grasp_z = bowl_pos[2] + grasp_offset
        print(f"  Grasp direction:      {dir_label} (rotation={'yes' if needs_rot else 'no'})")
        print(f"  Rim edge ({dir_label}):       ({rim_x:.4f}, {rim_y:.4f})")
        print(f"  Grasp Z target:       {grasp_z:.4f}")

    # --- Phase 1a: WP1 only (approach above bowl) ---
    # Use initial bowl_pos for WP1 (move above bowl, no contact)
    wp1_waypoints = build_grasp_waypoints(bowl_pos, init_rot,
                                          rim_offset_x=rim_dx, rim_offset_y=rim_dy,
                                          bowl_radius=bowl_radius,
                                          grasp_offset=grasp_offset,
                                          needs_rotation=needs_rot)[:1]

    total_steps = 0

    for wp_idx, (target_pose, gripper, dist_thresh, hold_steps) in enumerate(wp1_waypoints):
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
            bowl_after = get_body_pos(env, bowl_body)
            print(f"    WP{wp_idx+1} done: eef={eef_after}, "
                  f"bowl={bowl_after}, steps={steps_at_wp}")

        if total_steps >= max_steps:
            if debug:
                print(f"  Max steps reached at waypoint {wp_idx + 1}")
            break

    # --- Re-read bowl position after WP1 (bowl has had 10+~20 steps to settle) ---
    bowl_pos = get_body_pos(env, bowl_body)
    if debug:
        print(f"  Bowl pos after WP1: {bowl_pos} (z={bowl_pos[2]:.4f})")

    # --- Phase 1b: WP2-4 (descend, close, lift) with updated bowl_pos ---
    wp234_waypoints = build_grasp_waypoints(bowl_pos, init_rot,
                                            rim_offset_x=rim_dx, rim_offset_y=rim_dy,
                                            bowl_radius=bowl_radius,
                                            grasp_offset=grasp_offset,
                                            needs_rotation=needs_rot)[1:]

    if total_steps < max_steps:
        for wp_idx_offset, (target_pose, gripper, dist_thresh, hold_steps) in enumerate(wp234_waypoints):
            wp_idx = wp_idx_offset + 1  # WP2=1, WP3=2, WP4=3
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
                bowl_after = get_body_pos(env, bowl_body)
                print(f"    WP{wp_idx+1} done: eef={eef_after}, "
                      f"bowl={bowl_after}, steps={steps_at_wp}")

            if total_steps >= max_steps:
                if debug:
                    print(f"  Max steps reached at waypoint {wp_idx + 1}")
                break

    # --- Phase 2: Compute grasp offset and build place waypoints (5-8) ---
    bowl_after_grasp = get_body_pos(env, bowl_body)
    eef_after_grasp, _ = unmake_pose(get_eef_pose(env))
    grasp_offset_xy = bowl_after_grasp[:2] - eef_after_grasp[:2]
    place_xy = ramekin_pos[:2] - grasp_offset_xy  # bowl center lands on ramekin center

    if debug:
        print(f"\n  Grasp offset (bowl-eef): [{grasp_offset_xy[0]:.4f}, {grasp_offset_xy[1]:.4f}]")
        print(f"  Adjusted place XY: [{place_xy[0]:.4f}, {place_xy[1]:.4f}] "
              f"(ramekin center: [{ramekin_pos[0]:.4f}, {ramekin_pos[1]:.4f}])")

    place_waypoints = build_place_waypoints(place_xy, ramekin_pos[2], init_rot,
                                            place_offset=place_offset)

    if total_steps < max_steps:
        for wp_idx_offset, (target_pose, gripper, dist_thresh, hold_steps) in enumerate(place_waypoints):
            wp_idx = wp_idx_offset + 4  # 4 grasp waypoints (WP1-4)
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
                bowl_after = get_body_pos(env, bowl_body)
                print(f"    WP{wp_idx+1} done: eef={eef_after}, "
                      f"bowl={bowl_after}, steps={steps_at_wp}")

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


EXPECTED_MAX_STEPS = 80  # Calibrated: max=74, ceil/5×5+5=80

def main():
    parser = argparse.ArgumentParser(
        description="Scripted waypoint demo collection (Task 45: black bowl → ramekin)")
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
    parser.add_argument("--place_offset", type=float, default=0.06,
                        help="Z offset above ramekin for place (0.06: avoids physical block on ramekin)")
    parser.add_argument("--bowl_radius", type=float, default=0.048,
                        help="Bowl radius for edge grasp offset (default 0.048)")
    parser.add_argument("--grasp_offset", type=float, default=0.03,
                        help="Z offset above bowl body center for grasp (default 0.03)")
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
    print("Scripted Waypoint Demo Collection (Task 45)")
    print("=" * 60)
    print(f"Task: {instruction}")
    print(f"Max attempts: {num_demos}")
    print(f"Target successes: {target}")
    print(f"Max steps/demo: {args.max_steps}")
    print(f"Place offset: {args.place_offset}")
    print(f"Bowl radius: {args.bowl_radius}")
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
    bowl_body, ramekin_site = discover_names(env, debug=args.debug)
    print(f"  Bowl body:    {bowl_body}")
    print(f"  Ramekin site: {ramekin_site}")

    # Collect demos
    demos = []
    successes = 0
    failures = 0

    for i in range(num_demos):
        print(f"\n--- Demo {i + 1}/{num_demos} ---")
        states, actions, success = collect_one_demo(
            env, bowl_body, ramekin_site,
            max_steps=args.max_steps,
            debug=args.debug,
            place_offset=args.place_offset,
            bowl_radius=args.bowl_radius,
            grasp_offset=args.grasp_offset,
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
