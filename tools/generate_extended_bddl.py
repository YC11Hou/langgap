#!/usr/bin/env python3
"""
Generate extended BDDL files.

Based on the extension design document, generate extended BDDL files for each suite.
"""

import os
import re
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

# Path configuration
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BDDL_DIR = DATA_DIR / "bddl_files"
INIT_DIR = DATA_DIR / "init_files"


# ============================================================================
# libero_object extension configuration
# ============================================================================

# Task ID to filename mapping
LIBERO_OBJECT_TASKS = {
    0: "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
    1: "pick_up_the_cream_cheese_and_place_it_in_the_basket",
    2: "pick_up_the_salad_dressing_and_place_it_in_the_basket",
    3: "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
    4: "pick_up_the_ketchup_and_place_it_in_the_basket",
    5: "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
    6: "pick_up_the_butter_and_place_it_in_the_basket",
    7: "pick_up_the_milk_and_place_it_in_the_basket",
    8: "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
    9: "pick_up_the_orange_juice_and_place_it_in_the_basket",
}

# Dimension 1: Change manipulated object (22 entries) - use objects from different scenes
# Format: (ext_id, test_scene_task_id, target_object)
# Corrected based on actual scene objects (some objects in the original design doc were not in the corresponding scene)
LIBERO_OBJECT_DIM1 = [
    (1, 0, "salad_dressing"),      # scene0: salad_dressing, cream_cheese, milk, tomato_sauce, butter
    (2, 0, "cream_cheese"),
    (3, 1, "alphabet_soup"),       # scene1: alphabet_soup, milk, tomato_sauce, butter, orange_juice
    (4, 1, "butter"),
    (5, 2, "milk"),                # scene2: ketchup, alphabet_soup, cream_cheese, milk, tomato_sauce
    (6, 2, "tomato_sauce"),
    (7, 3, "ketchup"),             # scene3: chocolate_pudding, ketchup, salad_dressing, alphabet_soup, cream_cheese
    (8, 3, "chocolate_pudding"),
    (9, 4, "bbq_sauce"),           # scene4: bbq_sauce, salad_dressing, alphabet_soup, cream_cheese, milk
    (10, 4, "milk"),
    (11, 5, "bbq_sauce"),          # scene5: milk, butter, orange_juice, chocolate_pudding, bbq_sauce (alphabet_soup not present)
    (12, 5, "orange_juice"),       # (salad_dressing not present)
    (13, 6, "ketchup"),            # scene6: tomato_sauce, orange_juice, chocolate_pudding, bbq_sauce, ketchup (cream_cheese not present)
    (14, 6, "tomato_sauce"),
    (15, 7, "butter"),             # scene7: cream_cheese, tomato_sauce, butter, orange_juice, chocolate_pudding
    (16, 7, "cream_cheese"),       # (salad_dressing not present)
    (17, 8, "orange_juice"),       # scene8: orange_juice, bbq_sauce, ketchup, salad_dressing, alphabet_soup
    (18, 8, "bbq_sauce"),
    (19, 8, "ketchup"),
    (20, 9, "chocolate_pudding"),  # scene9: butter, chocolate_pudding, bbq_sauce, ketchup, salad_dressing
    (21, 9, "butter"),
    (22, 9, "salad_dressing"),     # (milk not present)
]

# Dimension 2: Paraphrase (8 entries)
# Format: (ext_id, original_task_id, new_instruction)
LIBERO_OBJECT_DIM2 = [
    (1, 0, "grab the alphabet soup and put it into the basket"),
    (2, 0, "take the alphabet soup and set it in the basket"),
    (3, 1, "grab the cream cheese and put it into the basket"),
    (4, 1, "take the cream cheese and set it in the basket"),
    (5, 7, "grab the milk and put it into the basket"),
    (6, 7, "take the milk and set it in the basket"),
    (7, 9, "grab the orange juice and put it into the basket"),
    (8, 9, "take the orange juice and set it in the basket"),
]

# Object name mapping (for references in BDDL files)
OBJECT_NAMES = {
    "alphabet_soup": "alphabet_soup_1",
    "cream_cheese": "cream_cheese_1",
    "salad_dressing": "salad_dressing_1",
    "bbq_sauce": "bbq_sauce_1",
    "ketchup": "ketchup_1",
    "tomato_sauce": "tomato_sauce_1",
    "butter": "butter_1",
    "milk": "milk_1",
    "chocolate_pudding": "chocolate_pudding_1",
    "orange_juice": "orange_juice_1",
}


# ============================================================================
# libero_goal extension configuration
# ============================================================================

LIBERO_GOAL_TASKS = {
    0: "open_the_middle_drawer_of_the_cabinet",
    1: "put_the_bowl_on_the_stove",
    2: "put_the_wine_bottle_on_top_of_the_cabinet",
    3: "open_the_top_drawer_and_put_the_bowl_inside",
    4: "put_the_bowl_on_top_of_the_cabinet",
    5: "push_the_plate_to_the_front_of_the_stove",
    6: "put_the_cream_cheese_in_the_bowl",
    7: "turn_on_the_stove",
    8: "put_the_bowl_on_the_plate",
    9: "put_the_wine_bottle_on_the_rack",
}

# libero_goal object and target location mapping
GOAL_OBJECTS = {
    "bowl": "akita_black_bowl_1",
    "plate": "plate_1",
    "wine_bottle": "wine_bottle_1",
    "cream_cheese": "cream_cheese_1",
}

GOAL_TARGETS = {
    "stove": "flat_stove_1_cook_region",
    "plate": "plate_1",
    "cabinet": "wooden_cabinet_1_top_side",
    "rack": "wine_rack_1_top_region",
    "bowl": "akita_black_bowl_1",
}

# Dimension 1: Change manipulated object (11 entries)
# Format: (ext_id, base_task_id, instruction, object, target_location)
LIBERO_GOAL_DIM1 = [
    (1, 1, "put the plate on the stove", "plate", "stove"),
    (2, 1, "put the wine bottle on the stove", "wine_bottle", "stove"),
    (3, 1, "put the cream cheese on the stove", "cream_cheese", "stove"),
    (4, 4, "put the plate on top of the cabinet", "plate", "cabinet"),
    (5, 4, "put the cream cheese on top of the cabinet", "cream_cheese", "cabinet"),
    (6, 8, "put the wine bottle on the plate", "wine_bottle", "plate"),
    (7, 8, "put the cream cheese on the plate", "cream_cheese", "plate"),
    (8, 9, "put the bowl on the rack", "bowl", "rack"),
    (9, 9, "put the plate on the rack", "plate", "rack"),
    (10, 9, "put the cream cheese on the rack", "cream_cheese", "rack"),
    (11, 0, "open the bottom drawer of the cabinet", None, None),  # special case
]

# Dimension 4: Paraphrase (7 entries)
# Format: (ext_id, base_task_id, new_instruction)
LIBERO_GOAL_DIM4 = [
    (1, 1, "place the bowl onto the stove"),
    (2, 1, "move the bowl to the stove"),
    (3, 8, "place the bowl onto the plate"),
    (4, 8, "move the bowl to the plate"),
    (5, 9, "place the wine bottle onto the rack"),
    (6, 9, "move the wine bottle to the rack"),
    (7, 7, "switch on the stove"),
]


# ============================================================================
# libero_spatial extension configuration
# ============================================================================

LIBERO_SPATIAL_TASKS = {
    0: "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    1: "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
    2: "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    3: "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
    4: "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
    5: "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
    6: "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
    7: "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
    8: "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
    9: "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
}

# libero_spatial target location mapping
SPATIAL_TARGETS = {
    "plate": "plate_1",
    "stove": "flat_stove_1_cook_region",
    "cabinet": "wooden_cabinet_1_top_side",
    "ramekin": "glazed_rim_porcelain_ramekin_1",
    "cookie_box": "cookies_1",
}

# Dimension 1: Change bowl (5 entries) - manipulate bowl_2 instead of bowl_1
# Format: (ext_id, base_task_id, new_instruction)
LIBERO_SPATIAL_DIM1 = [
    (1, 0, "pick up the black bowl on the right of the ramekin and place it on the plate"),
    (2, 2, "pick up the black bowl on the right of the plate and place it on the plate"),
    (3, 3, "pick up the black bowl on top of the wooden cabinet and place it on the plate"),
    (4, 4, "pick up the black bowl on top of the wooden cabinet and place it on the plate"),
    (5, 8, "pick up the black bowl next to the ramekin and place it on the plate"),
]

# Dimension 2: Change target location (12 entries) - change target from plate to other locations
# Format: (ext_id, base_task_id, new_instruction, new_target)
LIBERO_SPATIAL_DIM2 = [
    (1, 0, "pick up the black bowl between the plate and the ramekin and place it on the stove", "stove"),
    (2, 0, "pick up the black bowl between the plate and the ramekin and place it on the cabinet", "cabinet"),
    (3, 2, "pick up the black bowl from the middle of the table and place it on the stove", "stove"),
    (4, 2, "pick up the black bowl from the middle of the table and place it on the ramekin", "ramekin"),
    (5, 3, "pick up the black bowl on the cookie box and place it on the stove", "stove"),
    (6, 3, "pick up the black bowl on the cookie box and place it on the cabinet", "cabinet"),
    (7, 7, "pick up the black bowl on the stove and place it on the cabinet", "cabinet"),
    (8, 7, "pick up the black bowl on the stove and place it on the ramekin", "ramekin"),
    (9, 8, "pick up the black bowl next to the plate and place it on the stove", "stove"),
    (10, 8, "pick up the black bowl next to the plate and place it on the cabinet", "cabinet"),
    (11, 9, "pick up the black bowl on the wooden cabinet and place it on the stove", "stove"),
    (12, 9, "pick up the black bowl on the wooden cabinet and place it on the ramekin", "ramekin"),
]

# Dimension 3: Change manipulated object (10 entries) - manipulate ramekin, cookie_box, plate
# Format: (ext_id, any_base_task_id, new_instruction, manipulated_object, target_location)
LIBERO_SPATIAL_DIM3 = [
    (1, 0, "pick up the ramekin and place it on the plate", "ramekin", "plate"),
    (2, 0, "pick up the ramekin and place it on the stove", "ramekin", "stove"),
    (3, 0, "pick up the ramekin and place it on the cabinet", "ramekin", "cabinet"),
    (4, 0, "pick up the ramekin and place it on the cookie box", "ramekin", "cookie_box"),
    (5, 0, "pick up the cookie box and place it on the plate", "cookie_box", "plate"),
    (6, 0, "pick up the cookie box and place it on the stove", "cookie_box", "stove"),
    (7, 0, "pick up the cookie box and place it on the cabinet", "cookie_box", "cabinet"),
    (8, 0, "pick up the cookie box and place it on the ramekin", "cookie_box", "ramekin"),
    (9, 0, "pick up the plate and place it on the stove", "plate", "stove"),
    (10, 0, "pick up the plate and place it on the cabinet", "plate", "cabinet"),
]

# Dimension 4: Paraphrase (8 entries)
# Format: (ext_id, base_task_id, new_instruction)
LIBERO_SPATIAL_DIM4 = [
    (1, 0, "grab the bowl from between the plate and the ramekin and put it onto the plate"),
    (2, 0, "take the bowl from between the plate and the ramekin and set it on the plate"),
    (3, 3, "grab the bowl from the cookie box and put it onto the plate"),
    (4, 3, "take the bowl off the cookie box and set it on the plate"),
    (5, 7, "grab the bowl from the stove and put it onto the plate"),
    (6, 7, "take the bowl off the stove and set it on the plate"),
    (7, 9, "grab the bowl from the cabinet and put it onto the plate"),
    (8, 9, "take the bowl off the cabinet and set it on the plate"),
]

# Dimension 5: Drawer actions (4 entries)
# Format: (ext_id, base_task_id, instruction, action_type)
LIBERO_SPATIAL_DIM5 = [
    (1, 4, "close the top drawer of the cabinet", "close_top"),  # Task 4 drawer is open
    (2, 0, "open the top drawer of the cabinet", "open_top"),
    (3, 1, "open the middle drawer of the cabinet", "open_middle"),
    (4, 2, "open the bottom drawer of the cabinet", "open_bottom"),
]

# libero_spatial object mapping
SPATIAL_OBJECTS = {
    "bowl": "akita_black_bowl_1",
    "bowl_2": "akita_black_bowl_2",
    "ramekin": "glazed_rim_porcelain_ramekin_1",
    "cookie_box": "cookies_1",
    "plate": "plate_1",
}


def read_bddl(file_path: Path) -> str:
    """Read BDDL file content."""
    with open(file_path, 'r') as f:
        return f.read()


def write_bddl(file_path: Path, content: str):
    """Write BDDL file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(content)


def modify_bddl_language(content: str, new_language: str) -> str:
    """Modify the language field of a BDDL file."""
    # Match the (:language ...) section
    pattern = r'\(:language [^)]+\)'
    replacement = f'(:language {new_language})'
    return re.sub(pattern, replacement, content)


def modify_bddl_obj_of_interest(content: str, target_obj: str) -> str:
    """Modify the obj_of_interest field of a BDDL file."""
    obj_name = OBJECT_NAMES.get(target_obj, f"{target_obj}_1")
    # Match and replace the (:obj_of_interest ...) section
    pattern = r'\(:obj_of_interest\s+\w+_1\s+basket_1\s*\)'
    replacement = f'(:obj_of_interest\n    {obj_name}\n    basket_1\n  )'
    return re.sub(pattern, replacement, content)


def modify_bddl_goal(content: str, target_obj: str) -> str:
    """Modify the goal field of a BDDL file."""
    obj_name = OBJECT_NAMES.get(target_obj, f"{target_obj}_1")
    # Match and replace the (:goal ...) section
    pattern = r'\(:goal\s+\(And \(In \w+_1 basket_1_contain_region\)\)\s*\)'
    replacement = f'(:goal\n    (And (In {obj_name} basket_1_contain_region))\n  )'
    return re.sub(pattern, replacement, content)


def get_objects_in_scene(content: str) -> List[str]:
    """Extract list of objects in the scene from a BDDL file."""
    # Match the (:objects ...) section
    match = re.search(r'\(:objects\s+(.*?)\s*\)', content, re.DOTALL)
    if not match:
        return []

    objects_text = match.group(1)
    # Extract object names (format: xxx_1 - xxx)
    objects = []
    for line in objects_text.strip().split('\n'):
        line = line.strip()
        if line and ' - ' in line:
            obj_name = line.split(' - ')[0].strip()
            if obj_name != 'basket_1':
                # Convert to short name
                short_name = obj_name.replace('_1', '')
                objects.append(short_name)
    return objects


def generate_libero_object_dim1():
    """Generate libero_object dimension 1 extension (change manipulated object)."""
    print("\n=== Generating libero_object Dimension 1 (Change Object) ===")

    src_dir = BDDL_DIR / "libero_object" / "original"
    dst_dir = BDDL_DIR / "libero_object" / "extended" / "dim1_change_object"
    init_src_dir = INIT_DIR / "libero_object" / "original"
    init_dst_dir = INIT_DIR / "libero_object" / "extended" / "dim1_change_object"

    dst_dir.mkdir(parents=True, exist_ok=True)
    init_dst_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0

    for ext_id, scene_task_id, target_obj in LIBERO_OBJECT_DIM1:
        # Read source scene's BDDL file
        scene_task_name = LIBERO_OBJECT_TASKS[scene_task_id]
        src_bddl = src_dir / f"{scene_task_name}.bddl"
        src_init = init_src_dir / f"{scene_task_name}.pruned_init"

        if not src_bddl.exists():
            print(f"  [ERROR] Source BDDL not found: {src_bddl}")
            continue

        content = read_bddl(src_bddl)

        # Verify target object is in the scene
        scene_objects = get_objects_in_scene(content)
        if target_obj not in scene_objects:
            print(f"  [WARNING] ext_{ext_id}: Object '{target_obj}' not in scene {scene_task_id}")
            print(f"            Scene objects: {scene_objects}")
            continue

        # Modify BDDL content
        new_instruction = f"Pick the {target_obj.replace('_', ' ')} and place it in the basket"
        content = modify_bddl_language(content, new_instruction)
        content = modify_bddl_obj_of_interest(content, target_obj)
        content = modify_bddl_goal(content, target_obj)

        # Generate filename
        dst_name = f"ext_{ext_id:02d}_scene{scene_task_id}_{target_obj}"
        dst_bddl = dst_dir / f"{dst_name}.bddl"
        dst_init = init_dst_dir / f"{dst_name}.pruned_init"

        # Write BDDL file
        write_bddl(dst_bddl, content)

        # Copy init file (use the same scene's init)
        if src_init.exists():
            shutil.copy(src_init, dst_init)

        print(f"  [OK] ext_{ext_id:02d}: scene={scene_task_id}, object={target_obj}")
        success_count += 1

    print(f"\nDimension 1: Generated {success_count}/{len(LIBERO_OBJECT_DIM1)} files")
    return success_count


def generate_libero_object_dim2():
    """Generate libero_object dimension 2 extension (paraphrase)."""
    print("\n=== Generating libero_object Dimension 2 (Paraphrase) ===")

    src_dir = BDDL_DIR / "libero_object" / "original"
    dst_dir = BDDL_DIR / "libero_object" / "extended" / "dim2_paraphrase"
    init_src_dir = INIT_DIR / "libero_object" / "original"
    init_dst_dir = INIT_DIR / "libero_object" / "extended" / "dim2_paraphrase"

    dst_dir.mkdir(parents=True, exist_ok=True)
    init_dst_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0

    for ext_id, orig_task_id, new_instruction in LIBERO_OBJECT_DIM2:
        # Read original task's BDDL file
        orig_task_name = LIBERO_OBJECT_TASKS[orig_task_id]
        src_bddl = src_dir / f"{orig_task_name}.bddl"
        src_init = init_src_dir / f"{orig_task_name}.pruned_init"

        if not src_bddl.exists():
            print(f"  [ERROR] Source BDDL not found: {src_bddl}")
            continue

        content = read_bddl(src_bddl)

        # Only modify language
        content = modify_bddl_language(content, new_instruction)

        # Generate filename
        # Extract object name from instruction
        obj_match = re.search(r'(grab|take) the (\w+)', new_instruction)
        if obj_match:
            obj_name = obj_match.group(2)
        else:
            obj_name = "unknown"

        verb = "grab" if "grab" in new_instruction else "take"
        dst_name = f"ext_{ext_id:02d}_task{orig_task_id}_{verb}_{obj_name}"
        dst_bddl = dst_dir / f"{dst_name}.bddl"
        dst_init = init_dst_dir / f"{dst_name}.pruned_init"

        # Write files
        write_bddl(dst_bddl, content)
        if src_init.exists():
            shutil.copy(src_init, dst_init)

        print(f"  [OK] ext_{ext_id:02d}: task={orig_task_id}, instruction='{new_instruction[:50]}...'")
        success_count += 1

    print(f"\nDimension 2: Generated {success_count}/{len(LIBERO_OBJECT_DIM2)} files")
    return success_count


# ============================================================================
# libero_goal extension generation
# ============================================================================

def modify_goal_obj_of_interest(content: str, obj: str, target: str) -> str:
    """Modify libero_goal's obj_of_interest field."""
    obj_name = GOAL_OBJECTS.get(obj, f"{obj}_1")
    target_name = GOAL_TARGETS.get(target, target)

    # Match and replace the (:obj_of_interest ...) section
    pattern = r'\(:obj_of_interest\s+[\w_]+\s+[\w_]+\s*\)'
    replacement = f'(:obj_of_interest\n    {obj_name}\n    {target_name}\n  )'
    return re.sub(pattern, replacement, content)


def modify_goal_goal(content: str, obj: str, target: str) -> str:
    """Modify libero_goal's goal field."""
    obj_name = GOAL_OBJECTS.get(obj, f"{obj}_1")
    target_name = GOAL_TARGETS.get(target, target)

    # Match and replace the (:goal ...) section
    pattern = r'\(:goal\s+\(And \([^)]+\)\)\s*\)'
    replacement = f'(:goal\n    (And (On {obj_name} {target_name}))\n  )'
    return re.sub(pattern, replacement, content)


def modify_goal_drawer(content: str, drawer_type: str) -> str:
    """Modify the goal for drawer-related tasks."""
    # Modify obj_of_interest
    pattern = r'\(:obj_of_interest\s+[\w_]+\s*\)'
    replacement = f'(:obj_of_interest\n    wooden_cabinet_1_{drawer_type}_region\n  )'
    content = re.sub(pattern, replacement, content)

    # Modify goal
    pattern = r'\(:goal\s+\(And \([^)]+\)\)\s*\)'
    replacement = f'(:goal\n    (And (Open wooden_cabinet_1_{drawer_type}_region))\n  )'
    return re.sub(pattern, replacement, content)


def generate_libero_goal_dim1():
    """Generate libero_goal dimension 1 extension (change object/target)."""
    print("\n=== Generating libero_goal Dimension 1 (Change Object/Target) ===")

    src_dir = BDDL_DIR / "libero_goal" / "original"
    dst_dir = BDDL_DIR / "libero_goal" / "extended" / "dim1_change_object"
    init_src_dir = INIT_DIR / "libero_goal" / "original"
    init_dst_dir = INIT_DIR / "libero_goal" / "extended" / "dim1_change_object"

    dst_dir.mkdir(parents=True, exist_ok=True)
    init_dst_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0

    for ext_id, base_task_id, instruction, obj, target in LIBERO_GOAL_DIM1:
        # Read base task's BDDL file
        base_task_name = LIBERO_GOAL_TASKS[base_task_id]
        src_bddl = src_dir / f"{base_task_name}.bddl"
        src_init = init_src_dir / f"{base_task_name}.pruned_init"

        if not src_bddl.exists():
            print(f"  [ERROR] Source BDDL not found: {src_bddl}")
            continue

        content = read_bddl(src_bddl)

        # Modify BDDL content
        content = modify_bddl_language(content, instruction)

        if obj is None:  # Special case for drawer tasks
            content = modify_goal_drawer(content, "bottom")
        else:
            content = modify_goal_obj_of_interest(content, obj, target)
            content = modify_goal_goal(content, obj, target)

        # Generate filename
        if obj:
            dst_name = f"ext_{ext_id:02d}_{obj}_to_{target}"
        else:
            dst_name = f"ext_{ext_id:02d}_bottom_drawer"
        dst_bddl = dst_dir / f"{dst_name}.bddl"
        dst_init = init_dst_dir / f"{dst_name}.pruned_init"

        # Write files
        write_bddl(dst_bddl, content)
        if src_init.exists():
            shutil.copy(src_init, dst_init)

        print(f"  [OK] ext_{ext_id:02d}: {instruction[:50]}...")
        success_count += 1

    print(f"\nDimension 1: Generated {success_count}/{len(LIBERO_GOAL_DIM1)} files")
    return success_count


def generate_libero_goal_dim4():
    """Generate libero_goal dimension 4 extension (paraphrase)."""
    print("\n=== Generating libero_goal Dimension 4 (Paraphrase) ===")

    src_dir = BDDL_DIR / "libero_goal" / "original"
    dst_dir = BDDL_DIR / "libero_goal" / "extended" / "dim4_paraphrase"
    init_src_dir = INIT_DIR / "libero_goal" / "original"
    init_dst_dir = INIT_DIR / "libero_goal" / "extended" / "dim4_paraphrase"

    dst_dir.mkdir(parents=True, exist_ok=True)
    init_dst_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0

    for ext_id, base_task_id, new_instruction in LIBERO_GOAL_DIM4:
        # Read base task's BDDL file
        base_task_name = LIBERO_GOAL_TASKS[base_task_id]
        src_bddl = src_dir / f"{base_task_name}.bddl"
        src_init = init_src_dir / f"{base_task_name}.pruned_init"

        if not src_bddl.exists():
            print(f"  [ERROR] Source BDDL not found: {src_bddl}")
            continue

        content = read_bddl(src_bddl)

        # Only modify language
        content = modify_bddl_language(content, new_instruction)

        # Generate filename
        verb = new_instruction.split()[0]  # place, move, switch
        dst_name = f"ext_{ext_id:02d}_task{base_task_id}_{verb}"
        dst_bddl = dst_dir / f"{dst_name}.bddl"
        dst_init = init_dst_dir / f"{dst_name}.pruned_init"

        # Write files
        write_bddl(dst_bddl, content)
        if src_init.exists():
            shutil.copy(src_init, dst_init)

        print(f"  [OK] ext_{ext_id:02d}: '{new_instruction}'")
        success_count += 1

    print(f"\nDimension 4: Generated {success_count}/{len(LIBERO_GOAL_DIM4)} files")
    return success_count


# ============================================================================
# libero_spatial extension generation
# ============================================================================

def modify_spatial_goal(content: str, target: str) -> str:
    """Modify libero_spatial's goal field (change target location)."""
    target_name = SPATIAL_TARGETS.get(target, target)
    # Match and replace the (:goal ...) section
    pattern = r'\(:goal\s+\(And \(On \w+ \w+\)\)\s*\)'
    replacement = f'(:goal\n    (And (On akita_black_bowl_1 {target_name}))\n  )'
    return re.sub(pattern, replacement, content)


def modify_spatial_obj_of_interest(content: str, obj: str, target: str) -> str:
    """Modify libero_spatial's obj_of_interest field."""
    obj_name = SPATIAL_OBJECTS.get(obj, f"{obj}_1")
    target_name = SPATIAL_TARGETS.get(target, target)
    # Match and replace
    pattern = r'\(:obj_of_interest\s+[\w_]+\s+[\w_]+\s*\)'
    replacement = f'(:obj_of_interest\n    {obj_name}\n    {target_name}\n  )'
    return re.sub(pattern, replacement, content)


def modify_spatial_goal_other_obj(content: str, obj: str, target: str) -> str:
    """Modify libero_spatial's goal field (manipulate other object)."""
    obj_name = SPATIAL_OBJECTS.get(obj, f"{obj}_1")
    target_name = SPATIAL_TARGETS.get(target, target)
    pattern = r'\(:goal\s+\(And \(On \w+ \w+\)\)\s*\)'
    replacement = f'(:goal\n    (And (On {obj_name} {target_name}))\n  )'
    return re.sub(pattern, replacement, content)


def modify_spatial_drawer_goal(content: str, action_type: str) -> str:
    """Modify libero_spatial's drawer action goal."""
    # action_type: close_top, open_top, open_middle, open_bottom
    action, drawer = action_type.split('_')
    drawer_region = f"wooden_cabinet_1_{drawer}_region"

    if action == "close":
        goal_cond = f"(Not (Open {drawer_region}))"
    else:  # open
        goal_cond = f"(Open {drawer_region})"

    pattern = r'\(:goal\s+\(And \([^)]+\)\)\s*\)'
    replacement = f'(:goal\n    (And {goal_cond})\n  )'
    content = re.sub(pattern, replacement, content)

    # Modify obj_of_interest
    pattern = r'\(:obj_of_interest\s+[\w_]+\s+[\w_]+\s*\)'
    replacement = f'(:obj_of_interest\n    {drawer_region}\n  )'
    return re.sub(pattern, replacement, content)


def generate_libero_spatial_dim1():
    """Generate libero_spatial dimension 1 extension (change bowl)."""
    print("\n=== Generating libero_spatial Dimension 1 (Change Bowl) ===")

    src_dir = BDDL_DIR / "libero_spatial" / "original"
    dst_dir = BDDL_DIR / "libero_spatial" / "extended" / "dim1_change_bowl"
    init_src_dir = INIT_DIR / "libero_spatial" / "original"
    init_dst_dir = INIT_DIR / "libero_spatial" / "extended" / "dim1_change_bowl"

    dst_dir.mkdir(parents=True, exist_ok=True)
    init_dst_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0

    for ext_id, base_task_id, instruction in LIBERO_SPATIAL_DIM1:
        base_task_name = LIBERO_SPATIAL_TASKS[base_task_id]
        src_bddl = src_dir / f"{base_task_name}.bddl"
        src_init = init_src_dir / f"{base_task_name}.pruned_init"

        if not src_bddl.exists():
            print(f"  [ERROR] Source BDDL not found: {src_bddl}")
            continue

        content = read_bddl(src_bddl)
        content = modify_bddl_language(content, instruction)

        # Modify to manipulate bowl_2
        pattern = r'\(:obj_of_interest\s+[\w_]+\s+[\w_]+\s*\)'
        content = re.sub(pattern, '(:obj_of_interest\n    akita_black_bowl_2\n    plate_1\n  )', content)

        pattern = r'\(:goal\s+\(And \(On \w+ \w+\)\)\s*\)'
        content = re.sub(pattern, '(:goal\n    (And (On akita_black_bowl_2 plate_1))\n  )', content)

        dst_name = f"ext_{ext_id:02d}_task{base_task_id}_bowl2"
        dst_bddl = dst_dir / f"{dst_name}.bddl"
        dst_init = init_dst_dir / f"{dst_name}.pruned_init"

        write_bddl(dst_bddl, content)
        if src_init.exists():
            shutil.copy(src_init, dst_init)

        print(f"  [OK] ext_{ext_id:02d}: {instruction[:50]}...")
        success_count += 1

    print(f"\nDimension 1: Generated {success_count}/{len(LIBERO_SPATIAL_DIM1)} files")
    return success_count


def generate_libero_spatial_dim2():
    """Generate libero_spatial dimension 2 extension (change target)."""
    print("\n=== Generating libero_spatial Dimension 2 (Change Target) ===")

    src_dir = BDDL_DIR / "libero_spatial" / "original"
    dst_dir = BDDL_DIR / "libero_spatial" / "extended" / "dim2_change_target"
    init_src_dir = INIT_DIR / "libero_spatial" / "original"
    init_dst_dir = INIT_DIR / "libero_spatial" / "extended" / "dim2_change_target"

    dst_dir.mkdir(parents=True, exist_ok=True)
    init_dst_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0

    for ext_id, base_task_id, instruction, target in LIBERO_SPATIAL_DIM2:
        base_task_name = LIBERO_SPATIAL_TASKS[base_task_id]
        src_bddl = src_dir / f"{base_task_name}.bddl"
        src_init = init_src_dir / f"{base_task_name}.pruned_init"

        if not src_bddl.exists():
            print(f"  [ERROR] Source BDDL not found: {src_bddl}")
            continue

        content = read_bddl(src_bddl)
        content = modify_bddl_language(content, instruction)
        content = modify_spatial_obj_of_interest(content, "bowl", target)
        content = modify_spatial_goal(content, target)

        dst_name = f"ext_{ext_id:02d}_task{base_task_id}_to_{target}"
        dst_bddl = dst_dir / f"{dst_name}.bddl"
        dst_init = init_dst_dir / f"{dst_name}.pruned_init"

        write_bddl(dst_bddl, content)
        if src_init.exists():
            shutil.copy(src_init, dst_init)

        print(f"  [OK] ext_{ext_id:02d}: {instruction[:50]}...")
        success_count += 1

    print(f"\nDimension 2: Generated {success_count}/{len(LIBERO_SPATIAL_DIM2)} files")
    return success_count


def generate_libero_spatial_dim3():
    """Generate libero_spatial dimension 3 extension (change object)."""
    print("\n=== Generating libero_spatial Dimension 3 (Change Object) ===")

    src_dir = BDDL_DIR / "libero_spatial" / "original"
    dst_dir = BDDL_DIR / "libero_spatial" / "extended" / "dim3_change_object"
    init_src_dir = INIT_DIR / "libero_spatial" / "original"
    init_dst_dir = INIT_DIR / "libero_spatial" / "extended" / "dim3_change_object"

    dst_dir.mkdir(parents=True, exist_ok=True)
    init_dst_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0

    for ext_id, base_task_id, instruction, obj, target in LIBERO_SPATIAL_DIM3:
        base_task_name = LIBERO_SPATIAL_TASKS[base_task_id]
        src_bddl = src_dir / f"{base_task_name}.bddl"
        src_init = init_src_dir / f"{base_task_name}.pruned_init"

        if not src_bddl.exists():
            print(f"  [ERROR] Source BDDL not found: {src_bddl}")
            continue

        content = read_bddl(src_bddl)
        content = modify_bddl_language(content, instruction)
        content = modify_spatial_obj_of_interest(content, obj, target)
        content = modify_spatial_goal_other_obj(content, obj, target)

        dst_name = f"ext_{ext_id:02d}_{obj}_to_{target}"
        dst_bddl = dst_dir / f"{dst_name}.bddl"
        dst_init = init_dst_dir / f"{dst_name}.pruned_init"

        write_bddl(dst_bddl, content)
        if src_init.exists():
            shutil.copy(src_init, dst_init)

        print(f"  [OK] ext_{ext_id:02d}: {instruction[:50]}...")
        success_count += 1

    print(f"\nDimension 3: Generated {success_count}/{len(LIBERO_SPATIAL_DIM3)} files")
    return success_count


def generate_libero_spatial_dim4():
    """Generate libero_spatial dimension 4 extension (paraphrase)."""
    print("\n=== Generating libero_spatial Dimension 4 (Paraphrase) ===")

    src_dir = BDDL_DIR / "libero_spatial" / "original"
    dst_dir = BDDL_DIR / "libero_spatial" / "extended" / "dim4_paraphrase"
    init_src_dir = INIT_DIR / "libero_spatial" / "original"
    init_dst_dir = INIT_DIR / "libero_spatial" / "extended" / "dim4_paraphrase"

    dst_dir.mkdir(parents=True, exist_ok=True)
    init_dst_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0

    for ext_id, base_task_id, instruction in LIBERO_SPATIAL_DIM4:
        base_task_name = LIBERO_SPATIAL_TASKS[base_task_id]
        src_bddl = src_dir / f"{base_task_name}.bddl"
        src_init = init_src_dir / f"{base_task_name}.pruned_init"

        if not src_bddl.exists():
            print(f"  [ERROR] Source BDDL not found: {src_bddl}")
            continue

        content = read_bddl(src_bddl)
        content = modify_bddl_language(content, instruction)

        verb = instruction.split()[0]
        dst_name = f"ext_{ext_id:02d}_task{base_task_id}_{verb}"
        dst_bddl = dst_dir / f"{dst_name}.bddl"
        dst_init = init_dst_dir / f"{dst_name}.pruned_init"

        write_bddl(dst_bddl, content)
        if src_init.exists():
            shutil.copy(src_init, dst_init)

        print(f"  [OK] ext_{ext_id:02d}: '{instruction[:50]}...'")
        success_count += 1

    print(f"\nDimension 4: Generated {success_count}/{len(LIBERO_SPATIAL_DIM4)} files")
    return success_count


def generate_libero_spatial_dim5():
    """Generate libero_spatial dimension 5 extension (drawer action)."""
    print("\n=== Generating libero_spatial Dimension 5 (Drawer Action) ===")

    src_dir = BDDL_DIR / "libero_spatial" / "original"
    dst_dir = BDDL_DIR / "libero_spatial" / "extended" / "dim5_drawer_action"
    init_src_dir = INIT_DIR / "libero_spatial" / "original"
    init_dst_dir = INIT_DIR / "libero_spatial" / "extended" / "dim5_drawer_action"

    dst_dir.mkdir(parents=True, exist_ok=True)
    init_dst_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0

    for ext_id, base_task_id, instruction, action_type in LIBERO_SPATIAL_DIM5:
        base_task_name = LIBERO_SPATIAL_TASKS[base_task_id]
        src_bddl = src_dir / f"{base_task_name}.bddl"
        src_init = init_src_dir / f"{base_task_name}.pruned_init"

        if not src_bddl.exists():
            print(f"  [ERROR] Source BDDL not found: {src_bddl}")
            continue

        content = read_bddl(src_bddl)
        content = modify_bddl_language(content, instruction)
        content = modify_spatial_drawer_goal(content, action_type)

        dst_name = f"ext_{ext_id:02d}_{action_type}"
        dst_bddl = dst_dir / f"{dst_name}.bddl"
        dst_init = init_dst_dir / f"{dst_name}.pruned_init"

        write_bddl(dst_bddl, content)
        if src_init.exists():
            shutil.copy(src_init, dst_init)

        print(f"  [OK] ext_{ext_id:02d}: '{instruction}'")
        success_count += 1

    print(f"\nDimension 5: Generated {success_count}/{len(LIBERO_SPATIAL_DIM5)} files")
    return success_count


def print_suite_summary(suite_name: str, counts: Dict[str, int]):
    """Print suite summary."""
    print(f"\n{'=' * 70}")
    print(f"Summary - {suite_name}")
    print("=" * 70)
    total = 0
    for dim_name, count in counts.items():
        print(f"  {dim_name}: {count} files")
        total += count
    print(f"  {'─' * 30}")
    print(f"  Total: {total} files")

    # List generated files
    ext_dir = BDDL_DIR / suite_name / "extended"
    if ext_dir.exists():
        print(f"\n  Generated files in {ext_dir}:")
        for dim_dir in sorted(ext_dir.iterdir()):
            if dim_dir.is_dir():
                files = list(dim_dir.glob("*.bddl"))
                print(f"    {dim_dir.name}/ ({len(files)} files)")


def main():
    print("=" * 70)
    print("Extended BDDL File Generator")
    print("=" * 70)

    results = {}

    # Generate libero_object extensions
    print("\n" + "#" * 70)
    print("# libero_object")
    print("#" * 70)
    obj_dim1 = generate_libero_object_dim1()
    obj_dim2 = generate_libero_object_dim2()
    results["libero_object"] = {
        "Dim1 (Change Object)": obj_dim1,
        "Dim2 (Paraphrase)": obj_dim2,
    }

    # Generate libero_goal extensions
    print("\n" + "#" * 70)
    print("# libero_goal")
    print("#" * 70)
    goal_dim1 = generate_libero_goal_dim1()
    goal_dim4 = generate_libero_goal_dim4()
    results["libero_goal"] = {
        "Dim1 (Change Object/Target)": goal_dim1,
        "Dim4 (Paraphrase)": goal_dim4,
    }

    # Generate libero_spatial extensions
    print("\n" + "#" * 70)
    print("# libero_spatial")
    print("#" * 70)
    spatial_dim1 = generate_libero_spatial_dim1()
    spatial_dim2 = generate_libero_spatial_dim2()
    spatial_dim3 = generate_libero_spatial_dim3()
    spatial_dim4 = generate_libero_spatial_dim4()
    spatial_dim5 = generate_libero_spatial_dim5()
    results["libero_spatial"] = {
        "Dim1 (Change Bowl)": spatial_dim1,
        "Dim2 (Change Target)": spatial_dim2,
        "Dim3 (Change Object)": spatial_dim3,
        "Dim4 (Paraphrase)": spatial_dim4,
        "Dim5 (Drawer Action)": spatial_dim5,
    }

    # Print per-suite summaries
    for suite_name, counts in results.items():
        print_suite_summary(suite_name, counts)

    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    grand_total = 0
    for suite_name, counts in results.items():
        suite_total = sum(counts.values())
        print(f"  {suite_name}: {suite_total} extensions")
        grand_total += suite_total
    print(f"  {'─' * 30}")
    print(f"  Grand Total: {grand_total} extension files")


if __name__ == "__main__":
    main()
