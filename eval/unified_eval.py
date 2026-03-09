#!/usr/bin/env python3
"""
Unified evaluation script - supports official tasks and extended tasks.

Features:
- Suite selection: libero_goal, libero_object, libero_spatial, libero_10
- Task type selection: original, extended, all
- Configurable number of episodes
- Parallel execution (via multiprocessing)
- Outputs results.json, summary.md, and videos

Usage:
  # Dry run: list all tasks
  python unified_eval.py --suite all --type all --dry_run

  # Run official tasks (1 episode)
  python unified_eval.py --suite all --type original --episodes 1

  # Run extended tasks and save videos
  python unified_eval.py --suite all --type extended --episodes 5 --save_video

  # Specify a single suite
  python unified_eval.py --suite libero_goal --type all --episodes 20
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict
import glob

os.environ['MUJOCO_GL'] = 'egl'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import torch
import imageio

# LIBERO imports
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import get_libero_path
from libero.libero import benchmark as libero_benchmark

# LeRobot imports
script_dir = Path(__file__).parent.absolute()
lerobot_path = script_dir.parent / 'lerobot' / 'src'
sys.path.insert(0, str(lerobot_path))

from lerobot.envs.utils import preprocess_observation
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor.env_processor import LiberoProcessorStep
from lerobot.processor.pipeline import PolicyProcessorPipeline


# ============================================================================
# Configuration
# ============================================================================

HF_TOKEN = os.environ.get("HF_TOKEN", "")
DEFAULT_PRETRAINED_PATH = "lerobot/pi05_libero_finetuned"
FINETUNED_PATH = "<YOUR_HF_USERNAME>/pi05_langgap_finetuned"  # Our fine-tuned model

# Data directory (under project root)
PROJECT_ROOT = script_dir.parent
DATA_DIR = PROJECT_ROOT / "data"
BDDL_DIR = DATA_DIR / "bddl_files"
INIT_DIR = DATA_DIR / "init_files"

# Extended task_id -> BDDL filename mapping (from task_registry.py)
sys.path.insert(0, str(Path(__file__).parent.parent))
from task_registry import get_id_to_name
EXTENDED_TASK_ID_TO_NAME = get_id_to_name()


def parse_task_ids(task_id_str: str) -> List[int]:
    """
    Parse task_id string, supporting multiple formats:
    - Single: "3" -> [3]
    - Multiple: "3,4,5" -> [3, 4, 5]
    - Range: "3-7" -> [3, 4, 5, 6, 7]
    """
    if not task_id_str:
        return []

    result = []
    task_id_str = task_id_str.strip()

    # Range format: 3-7
    if '-' in task_id_str and ',' not in task_id_str:
        parts = task_id_str.split('-')
        if len(parts) == 2:
            start, end = int(parts[0]), int(parts[1])
            result = list(range(start, end + 1))
    # Multiple format: 3,4,5
    elif ',' in task_id_str:
        result = [int(x.strip()) for x in task_id_str.split(',')]
    # Single format: 3
    else:
        result = [int(task_id_str)]

    return result


@dataclass
class TaskConfig:
    """Task configuration"""
    suite: str
    task_type: str  # original or extended
    task_id: str
    name: str
    instruction: str
    bddl_path: Path
    init_path: Path
    benchmark_task_id: int = -1  # LIBERO benchmark task_id (numeric index)


def get_instruction_from_bddl(bddl_path: Path) -> str:
    """Extract instruction from BDDL file"""
    with open(bddl_path, 'r') as f:
        content = f.read()
    import re
    match = re.search(r'\(:language ([^)]+)\)', content)
    if match:
        return match.group(1).strip()
    return "Unknown instruction"


def paraphrase_instruction(instruction: str, suite: str) -> str:
    """
    Paraphrase instructions following the official train->test convention.

    Official transformation patterns:
    - libero_spatial: pick up -> Pick, black bowl -> akita black bowl, cookie box -> cookies box
    - libero_goal: capitalize first letter
    - libero_object: pick up -> Pick, put it -> place it
    """
    s = instruction

    if suite == "libero_spatial":
        # pick up the black bowl → Pick the akita black bowl
        s = s.replace('pick up the black bowl', 'Pick the akita black bowl')
        # pick up the cookie box → Pick the cookies box
        s = s.replace('pick up the cookie box', 'Pick the cookies box')
        # pick up the ramekin → Pick the ramekin
        s = s.replace('pick up the ramekin', 'Pick the ramekin')
        # on top of the wooden cabinet → in the top layer of the wooden cabinet
        s = s.replace('on top of the wooden cabinet', 'in the top layer of the wooden cabinet')
        # If not already processed, capitalize first letter
        if s and s[0].islower():
            s = s[0].upper() + s[1:]

    elif suite == "libero_goal":
        # Capitalize first letter
        if s and s[0].islower():
            s = s[0].upper() + s[1:]

    elif suite == "libero_object":
        # pick up the -> Pick the
        s = s.replace('pick up the', 'Pick the')
        # put it in the basket -> place it in the basket
        s = s.replace('put it in the basket', 'place it in the basket')
        # If not already processed, capitalize first letter
        if s and s[0].islower():
            s = s[0].upper() + s[1:]

    return s


def get_benchmark_task_order(suite: str) -> Dict[str, int]:
    """Get benchmark task order mapping: task_name -> benchmark_task_id"""
    try:
        benchmark_dict = libero_benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[suite]()
        return {task.name: i for i, task in enumerate(task_suite.tasks)}
    except Exception as e:
        print(f"Warning: Could not get benchmark order for {suite}: {e}")
        return {}


def discover_tasks(suite: str, task_type: str) -> List[TaskConfig]:
    """Discover all tasks for the specified suite and type, ordered by benchmark index"""
    tasks = []

    suite_bddl_dir = BDDL_DIR / suite
    suite_init_dir = INIT_DIR / suite

    if task_type in ["original", "all"]:
        # Original tasks - use benchmark order
        orig_bddl_dir = suite_bddl_dir / "original"
        orig_init_dir = suite_init_dir / "original"

        # Get benchmark task order
        benchmark_order = get_benchmark_task_order(suite)

        if orig_bddl_dir.exists():
            orig_tasks = []
            for bddl_file in orig_bddl_dir.glob("*.bddl"):
                name = bddl_file.stem
                init_file = orig_init_dir / f"{name}.pruned_init"
                instruction = get_instruction_from_bddl(bddl_file)

                # Get benchmark task_id (numeric index)
                benchmark_task_id = benchmark_order.get(name, 999)

                orig_tasks.append(TaskConfig(
                    suite=suite,
                    task_type="original",
                    task_id=f"orig_{name}",
                    name=name,
                    instruction=instruction,
                    bddl_path=bddl_file,
                    init_path=init_file,
                    benchmark_task_id=benchmark_task_id,  # Store benchmark index
                ))

            # Sort by benchmark order
            orig_tasks.sort(key=lambda t: t.benchmark_task_id)
            tasks.extend(orig_tasks)

    if task_type in ["extended", "all"]:
        # Extended tasks
        ext_dir = suite_bddl_dir / "extended"
        ext_init_dir = suite_init_dir / "extended"

        if ext_dir.exists():
            for dim_dir in sorted(ext_dir.iterdir()):
                if dim_dir.is_dir():
                    for bddl_file in sorted(dim_dir.glob("*.bddl")):
                        name = bddl_file.stem
                        init_file = ext_init_dir / dim_dir.name / f"{name}.pruned_init"
                        instruction = get_instruction_from_bddl(bddl_file)

                        # Use full name to avoid truncation collision
                        tasks.append(TaskConfig(
                            suite=suite,
                            task_type=f"extended/{dim_dir.name}",
                            task_id=f"ext_{dim_dir.name}_{name}",
                            name=name,
                            instruction=instruction,
                            bddl_path=bddl_file,
                            init_path=init_file,
                        ))

    return tasks


# ============================================================================
# Environment creation
# ============================================================================

def create_env_from_bddl(bddl_path: Path, init_path: Path) -> tuple:
    """Create environment from BDDL and init files"""
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=256,
        camera_widths=256,
    )

    # Load init states
    if init_path.exists():
        init_states = torch.load(init_path, weights_only=False)
        if isinstance(init_states, dict):
            init_states = [init_states]
    else:
        init_states = [None]

    return env, init_states


def format_obs(env, raw_obs: dict) -> dict:
    """Format observations"""
    def ensure_batch(arr):
        if arr is None:
            return None
        if isinstance(arr, np.ndarray):
            if arr.ndim == 1:
                return arr[np.newaxis, :]
            elif arr.ndim == 2:
                return arr[np.newaxis, :, :]
        return arr

    return {
        "pixels": {
            "image": raw_obs.get("agentview_image"),
            "image2": raw_obs.get("robot0_eye_in_hand_image"),
        },
        "robot_state": {
            "eef": {
                "pos": ensure_batch(raw_obs.get("robot0_eef_pos")),
                "quat": ensure_batch(raw_obs.get("robot0_eef_quat")),
                "mat": ensure_batch(env.robots[0].controller.ee_ori_mat),
            },
            "gripper": {
                "qpos": ensure_batch(raw_obs.get("robot0_gripper_qpos")),
                "qvel": ensure_batch(raw_obs.get("robot0_gripper_qvel")),
            },
            "joints": {
                "pos": ensure_batch(raw_obs.get("robot0_joint_pos")),
                "vel": ensure_batch(raw_obs.get("robot0_joint_vel")),
            },
        },
    }


# ============================================================================
# Evaluation logic
# ============================================================================

def run_episode(
    env,
    init_states,
    init_state_id: int,
    policy,
    libero_pre,
    pol_pre,
    pol_post,
    task_description: str,
    max_steps: int = 300,
    save_video: Optional[str] = None,
) -> dict:
    """Run a single episode"""

    policy.reset()

    # Reset to specified initial state
    if init_states[init_state_id] is not None:
        env.set_init_state(init_states[init_state_id])
    raw_obs = env.reset()

    # Wait for physics to stabilize
    dummy_action = [0, 0, 0, 0, 0, 0, -1]
    for _ in range(10):
        raw_obs, _, _, _ = env.step(dummy_action)

    frames = []
    total_reward = 0

    for step in range(max_steps):
        obs = preprocess_observation(format_obs(env, raw_obs))
        obs["task"] = [task_description]

        obs = libero_pre(obs)
        obs = pol_pre(obs)

        with torch.inference_mode():
            action = policy.select_action(obs)
        action = pol_post(action)
        action_np = action.cpu().numpy()
        if action_np.ndim == 1:
            action_np = action_np[np.newaxis, :]
        action_np = action_np[0]

        raw_obs, reward, done, info = env.step(action_np)
        total_reward = reward
        is_success = env.check_success()

        if save_video:
            frame = raw_obs.get("agentview_image")
            frames.append(frame[::-1, ::-1])

        if done or is_success:
            break

    if save_video and frames:
        imageio.mimsave(save_video, frames, fps=30)

    return {
        "success": bool(is_success),
        "reward": float(total_reward),
        "steps": step + 1,
    }


def load_policy(model_path: str = None, use_fp16: bool = True):
    """Load π0.5 model (supports base models and PEFT fine-tuned models)"""
    from peft import PeftModel, PeftConfig

    pretrained_path = model_path or DEFAULT_PRETRAINED_PATH
    print(f"Loading π0.5 model from: {pretrained_path}")

    # Check if this is a PEFT model
    is_peft = False
    try:
        peft_config = PeftConfig.from_pretrained(pretrained_path, token=HF_TOKEN)
        is_peft = True
        base_model_path = peft_config.base_model_name_or_path
        print(f"Detected PEFT model, base model: {base_model_path}")
    except Exception:
        base_model_path = pretrained_path
        print("Loading as regular model (not PEFT)")

    # Load config
    config = PreTrainedConfig.from_pretrained(pretrained_path, token=HF_TOKEN)
    config.n_action_steps = 10

    # Select Policy class based on model path
    def get_policy_class(model_path: str):
        path_lower = model_path.lower()
        if "smolvla" in path_lower:
            return SmolVLAPolicy
        elif "pi0fast" in path_lower or "pi0_fast" in path_lower:
            return PI0FastPolicy
        elif "pi05" in path_lower:
            return PI05Policy
        else:
            return PI0Policy

    PolicyClass = get_policy_class(base_model_path)
    print(f"Using policy class: {PolicyClass.__name__}")

    # Load base model
    if use_fp16:
        print("Loading model in float16 for lower memory usage...")
        policy = PolicyClass.from_pretrained(
            base_model_path, config=config, token=HF_TOKEN,
            torch_dtype=torch.float16
        )
    else:
        policy = PolicyClass.from_pretrained(base_model_path, config=config, token=HF_TOKEN)

    # If PEFT model, load adapter
    if is_peft:
        print("Loading PEFT adapter...")
        policy = PeftModel.from_pretrained(policy, pretrained_path, token=HF_TOKEN)

    policy.eval()
    policy.to("cuda")

    libero_pre = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])
    pol_pre, pol_post = make_pre_post_processors(policy.config, pretrained_path=pretrained_path)

    return policy, libero_pre, pol_pre, pol_post


def main():
    parser = argparse.ArgumentParser(description="Unified evaluation script - supports official and extended tasks")
    parser.add_argument("--suite", type=str, default="all",
                        choices=["libero_goal", "libero_object", "libero_spatial", "libero_10", "all"],
                        help="Suite to evaluate")
    parser.add_argument("--type", type=str, default="all",
                        choices=["original", "extended", "all"],
                        help="Task type to evaluate")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes per task (official standard is 20)")
    parser.add_argument("--max_steps", type=int, default=300,
                        help="Maximum steps per episode")
    parser.add_argument("--save_video", action="store_true",
                        help="Save videos")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only list tasks without running")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Model path (supports HuggingFace repo or local path)")
    parser.add_argument("--finetuned", action="store_true",
                        help="Use fine-tuned model (<YOUR_HF_USERNAME>/pi05_langgap_finetuned)")
    parser.add_argument("--paraphrase", action="store_true",
                        help="Use official-style paraphrased instructions (simulates train->test variation)")
    parser.add_argument("--task_id", type=str, default=None,
                        help="Task ID, supports: single(3), multiple(3,4,5), range(3-7), all if not specified")
    parser.add_argument("--task_name", type=str, default=None,
                        help="Only test the task with specified name (for extended tasks, e.g. ext_03_ramekin_to_cabinet)")
    args = parser.parse_args()

    # Determine model path
    if args.finetuned:
        args.model_path = FINETUNED_PATH
    elif args.model_path is None:
        args.model_path = DEFAULT_PRETRAINED_PATH

    # Determine suites to evaluate
    if args.suite == "all":
        suites = ["libero_goal", "libero_object", "libero_spatial", "libero_10"]
    else:
        suites = [args.suite]

    # Discover all tasks
    all_tasks = []
    for suite in suites:
        tasks = discover_tasks(suite, args.type)
        all_tasks.extend(tasks)
        print(f"Found {len(tasks)} tasks in {suite} ({args.type})")

    # If task_name is specified, keep only that task (for extended tasks)
    if args.task_name is not None:
        filtered = [t for t in all_tasks if args.task_name in t.name]
        if filtered:
            all_tasks = filtered
            print(f"Filtered to task_name={args.task_name}: {len(all_tasks)} tasks")

    # If task_id is specified, keep only the specified tasks
    # task_id corresponds to LIBERO benchmark task index (0-9 for original, 40-98 for extended)
    if args.task_id is not None:
        task_ids = parse_task_ids(args.task_id)
        if task_ids:
            filtered = []
            for tid in task_ids:
                # 1. Match by benchmark_task_id (original tasks)
                matches = [t for t in all_tasks if t.benchmark_task_id == tid]
                # 2. If no match, try extended mapping
                if not matches and tid in EXTENDED_TASK_ID_TO_NAME:
                    ext_name = EXTENDED_TASK_ID_TO_NAME[tid]
                    matches = [t for t in all_tasks if t.name == ext_name]
                filtered.extend(matches)
            if filtered:
                all_tasks = filtered
                print(f"Filtered to task_id={args.task_id}: {len(all_tasks)} tasks")
            else:
                print(f"\nError: task_id={args.task_id} not found in suite={args.suite}, type={args.type}")
                for tid in task_ids:
                    if tid in EXTENDED_TASK_ID_TO_NAME:
                        print(f"  task_id={tid} -> {EXTENDED_TASK_ID_TO_NAME[tid]} (requires --suite all or matching suite)")
                    else:
                        print(f"  task_id={tid} not in extended mapping (original tasks use benchmark_task_id 0-9)")
                return

    print(f"\nTotal tasks to evaluate: {len(all_tasks)}")

    # Dry run mode: only list tasks
    if args.dry_run:
        print("\n" + "=" * 70)
        print("DRY RUN - Task List (sorted by benchmark task_id)")
        print("=" * 70)
        for i, task in enumerate(all_tasks):
            tid_str = f"task_id={task.benchmark_task_id}" if task.benchmark_task_id >= 0 else "extended"
            print(f"{i+1:3d}. [{task.suite}] [{tid_str}] {task.instruction[:50]}...")
        return

    # Output directory: auto-named based on model
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Extract name from model path (e.g. pi05_libero_base or checkpoints/005000)
        model_name = Path(args.model_path).name
        if model_name == "pretrained_model":
            # If checkpoint path, use parent directory name (e.g. 005000)
            model_name = Path(args.model_path).parent.name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(__file__).parent / "results" / f"{model_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("Unified Evaluation")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Suites: {suites}")
    print(f"Task type: {args.type}")
    print(f"Total tasks: {len(all_tasks)}")
    print(f"Episodes per task: {args.episodes}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)

    # Load model
    policy, libero_pre, pol_pre, pol_post = load_policy(model_path=args.model_path)

    # Run evaluation
    all_results = {}

    for task_idx, task in enumerate(all_tasks):
        task_key = f"{task.suite}_{task.task_id}"
        print(f"\n{'='*70}")
        print(f"[{task_idx+1}/{len(all_tasks)}] Testing: {task_key}")
        print(f"Type: {task.task_type}")
        print(f"Instruction: {task.instruction}")

        # If paraphrase is enabled, transform instruction
        test_instruction = task.instruction
        if args.paraphrase:
            test_instruction = paraphrase_instruction(task.instruction, task.suite)
            if test_instruction != task.instruction:
                print(f"Paraphrased: {test_instruction}")
        print("=" * 70)

        try:
            env, init_states = create_env_from_bddl(task.bddl_path, task.init_path)
            task_results = []

            for ep in range(args.episodes):
                video_path = None
                if args.save_video:
                    video_dir = output_dir / "videos" / task.suite
                    video_dir.mkdir(parents=True, exist_ok=True)
                    # Video naming: {task_type}_{task_name}_ep{ep}.mp4
                    safe_name = task.name.replace(" ", "_").replace("/", "_")[:50]
                    safe_type = task.task_type.replace("/", "_")
                    video_path = str(video_dir / f"{safe_type}_{safe_name}_ep{ep}.mp4")

                init_id = ep % len(init_states)
                print(f"  Episode {ep+1}/{args.episodes} (init_state={init_id})...", end=" ")

                result = run_episode(
                    env, init_states, init_id,
                    policy, libero_pre, pol_pre, pol_post,
                    task_description=test_instruction,
                    max_steps=args.max_steps,
                    save_video=video_path,
                )

                result["episode"] = ep
                result["init_state_id"] = init_id
                if video_path:
                    result["video_path"] = video_path
                task_results.append(result)

                status = "SUCCESS" if result["success"] else "FAIL"
                print(f"[{status}] steps={result['steps']}")

            env.close()

            # Statistics
            successes = sum(1 for r in task_results if r["success"])
            success_rate = successes / len(task_results) if task_results else 0

            result_entry = {
                "suite": task.suite,
                "task_type": task.task_type,
                "task_id": task.task_id,
                "name": task.name,
                "instruction": task.instruction,
                "n_episodes": args.episodes,
                "success_rate": success_rate,
                "successes": successes,
                "failures": args.episodes - successes,
                "episodes": task_results,
            }
            if args.paraphrase and test_instruction != task.instruction:
                result_entry["test_instruction"] = test_instruction
            all_results[task_key] = result_entry

            print(f"  Summary: {successes}/{args.episodes} = {success_rate*100:.1f}%")

        except Exception as e:
            print(f"  [ERROR] {e}")
            all_results[task_key] = {
                "suite": task.suite,
                "task_type": task.task_type,
                "task_id": task.task_id,
                "name": task.name,
                "instruction": task.instruction,
                "error": str(e),
            }

    # Save results
    result_file = output_dir / "results.json"
    with open(result_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {result_file}")

    # Generate summary
    generate_summary(all_results, output_dir, suites, args)


def generate_summary(all_results: Dict, output_dir: Path, suites: List[str], args):
    """Generate summary report"""
    summary_file = output_dir / "summary.md"

    with open(summary_file, 'w') as f:
        f.write("# Unified Evaluation Results\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Episodes per task**: {args.episodes}\n")
        f.write(f"**Task type**: {args.type}\n\n")

        # Statistics by suite
        for suite in suites:
            suite_results = {k: v for k, v in all_results.items() if v.get("suite") == suite}
            if not suite_results:
                continue

            f.write(f"## {suite}\n\n")

            # Group by type
            original_results = {k: v for k, v in suite_results.items() if v.get("task_type") == "original"}
            extended_results = {k: v for k, v in suite_results.items() if "extended" in str(v.get("task_type", ""))}

            if original_results:
                successes = sum(v.get("successes", 0) for v in original_results.values())
                total = sum(v.get("n_episodes", 0) for v in original_results.values())
                rate = successes / total * 100 if total > 0 else 0
                f.write(f"### Original Tasks\n")
                f.write(f"- Success Rate: **{rate:.1f}%** ({successes}/{total})\n")
                f.write(f"- Tasks: {len(original_results)}\n\n")

            if extended_results:
                successes = sum(v.get("successes", 0) for v in extended_results.values())
                total = sum(v.get("n_episodes", 0) for v in extended_results.values())
                rate = successes / total * 100 if total > 0 else 0
                f.write(f"### Extended Tasks\n")
                f.write(f"- Success Rate: **{rate:.1f}%** ({successes}/{total})\n")
                f.write(f"- Tasks: {len(extended_results)}\n\n")

        # Detailed results table
        f.write("---\n\n## Detailed Results\n\n")
        f.write("| Suite | Type | Task | Success Rate | Instruction |\n")
        f.write("|-------|------|------|-------------|-------------|\n")

        for k, v in sorted(all_results.items()):
            if "error" in v:
                f.write(f"| {v.get('suite', 'N/A')} | {v.get('task_type', 'N/A')} | {v.get('task_id', 'N/A')} | ERROR | {v.get('instruction', 'N/A')[:40]}... |\n")
            else:
                rate = v.get("success_rate", 0) * 100
                f.write(f"| {v.get('suite', 'N/A')} | {v.get('task_type', 'N/A')} | {v.get('task_id', 'N/A')} | {rate:.0f}% | {v.get('instruction', 'N/A')[:40]}... |\n")

    print(f"Summary saved to: {summary_file}")

    # Print summary to console
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for suite in suites:
        suite_results = {k: v for k, v in all_results.items() if v.get("suite") == suite}
        if not suite_results:
            continue

        print(f"\n{suite}:")

        original = {k: v for k, v in suite_results.items() if v.get("task_type") == "original"}
        extended = {k: v for k, v in suite_results.items() if "extended" in str(v.get("task_type", ""))}

        if original:
            successes = sum(v.get("successes", 0) for v in original.values())
            total = sum(v.get("n_episodes", 0) for v in original.values())
            print(f"  Original:  {successes}/{total} = {successes/total*100:.1f}%" if total > 0 else "  Original: N/A")

        if extended:
            successes = sum(v.get("successes", 0) for v in extended.values())
            total = sum(v.get("n_episodes", 0) for v in extended.values())
            print(f"  Extended:  {successes}/{total} = {successes/total*100:.1f}%" if total > 0 else "  Extended: N/A")


if __name__ == "__main__":
    main()
