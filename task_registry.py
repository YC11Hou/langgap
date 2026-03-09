"""Single source of truth for Task ID -> BDDL mapping

All extended tasks (Task 40-98) ID <-> BDDL name mapping.
Collection scripts, evaluation scripts, and documentation should all reference this file.

Note: 63<->84, 64<->60 have been swapped to match data already collected by collect_multi.sh.
- Task 63: ext_06_scene2_tomato_sauce (originally 84, object)
- Task 64: ext_05_scene2_milk (originally 60, object)
- Task 60: ext_05_task8_bowl2 (originally 64, spatial held-out)
- Task 84: ext_03_task3_bowl2 (originally 63, spatial held-out)
"""

EXTENDED_TASK_MAP = {
    # =====================================================================
    # libero_spatial training tasks
    # =====================================================================
    # dim1_change_bowl
    40: {"suite": "libero_spatial", "dim": "dim1_change_bowl", "bddl": "ext_01_task0_bowl2"},
    41: {"suite": "libero_spatial", "dim": "dim1_change_bowl", "bddl": "ext_02_task2_bowl2"},
    42: {"suite": "libero_spatial", "dim": "dim1_change_bowl", "bddl": "ext_04_task4_bowl2"},
    # dim2_change_target
    43: {"suite": "libero_spatial", "dim": "dim2_change_target", "bddl": "ext_01_task0_to_stove"},
    44: {"suite": "libero_spatial", "dim": "dim2_change_target", "bddl": "ext_02_task0_to_cabinet"},
    45: {"suite": "libero_spatial", "dim": "dim2_change_target", "bddl": "ext_04_task2_to_ramekin"},
    46: {"suite": "libero_spatial", "dim": "dim2_change_target", "bddl": "ext_07_task7_to_cabinet"},
    47: {"suite": "libero_spatial", "dim": "dim2_change_target", "bddl": "ext_09_task8_to_stove"},
    48: {"suite": "libero_spatial", "dim": "dim2_change_target", "bddl": "ext_11_task9_to_stove"},
    # dim3_change_object
    49: {"suite": "libero_spatial", "dim": "dim3_change_object", "bddl": "ext_01_ramekin_to_plate"},
    50: {"suite": "libero_spatial", "dim": "dim3_change_object", "bddl": "ext_03_ramekin_to_cabinet"},
    51: {"suite": "libero_spatial", "dim": "dim3_change_object", "bddl": "ext_05_cookie_box_to_plate"},
    52: {"suite": "libero_spatial", "dim": "dim3_change_object", "bddl": "ext_07_cookie_box_to_cabinet"},
    # dim5_drawer_action
    53: {"suite": "libero_spatial", "dim": "dim5_drawer_action", "bddl": "ext_03_open_middle"},

    # =====================================================================
    # libero_goal training tasks
    # =====================================================================
    # dim1_change_object
    54: {"suite": "libero_goal", "dim": "dim1_change_object", "bddl": "ext_03_cream_cheese_to_stove"},
    55: {"suite": "libero_goal", "dim": "dim1_change_object", "bddl": "ext_06_wine_bottle_to_plate"},
    56: {"suite": "libero_goal", "dim": "dim1_change_object", "bddl": "ext_09_cream_cheese_to_top_drawer"},
    57: {"suite": "libero_goal", "dim": "dim1_change_object", "bddl": "ext_11_bottom_drawer"},

    # =====================================================================
    # libero_object training tasks
    # =====================================================================
    # dim1_change_object
    58: {"suite": "libero_object", "dim": "dim1_change_object", "bddl": "ext_01_scene0_salad_dressing"},
    59: {"suite": "libero_object", "dim": "dim1_change_object", "bddl": "ext_03_scene1_alphabet_soup"},
    # --- SWAPPED: 60<->64 ---
    60: {"suite": "libero_spatial", "dim": "dim1_change_bowl", "bddl": "ext_05_task8_bowl2"},
    61: {"suite": "libero_object", "dim": "dim1_change_object", "bddl": "ext_07_scene3_ketchup"},
    62: {"suite": "libero_object", "dim": "dim1_change_object", "bddl": "ext_09_scene4_bbq_sauce"},
    # --- SWAPPED: 63<->84, 64<->60 (to match data already collected by collect_multi.sh) ---
    63: {"suite": "libero_object", "dim": "dim1_change_object", "bddl": "ext_06_scene2_tomato_sauce"},
    64: {"suite": "libero_object", "dim": "dim1_change_object", "bddl": "ext_05_scene2_milk"},

    # =====================================================================
    # libero_spatial held-out tasks
    # =====================================================================
    # dim2_change_target
    65: {"suite": "libero_spatial", "dim": "dim2_change_target", "bddl": "ext_03_task2_to_stove"},
    66: {"suite": "libero_spatial", "dim": "dim2_change_target", "bddl": "ext_05_task3_to_stove"},
    67: {"suite": "libero_spatial", "dim": "dim2_change_target", "bddl": "ext_06_task3_to_cabinet"},
    68: {"suite": "libero_spatial", "dim": "dim2_change_target", "bddl": "ext_08_task7_to_ramekin"},
    69: {"suite": "libero_spatial", "dim": "dim2_change_target", "bddl": "ext_10_task8_to_cabinet"},
    70: {"suite": "libero_spatial", "dim": "dim2_change_target", "bddl": "ext_12_task9_to_ramekin"},
    # dim3_change_object
    71: {"suite": "libero_spatial", "dim": "dim3_change_object", "bddl": "ext_02_ramekin_to_stove"},
    72: {"suite": "libero_spatial", "dim": "dim3_change_object", "bddl": "ext_04_ramekin_to_cookie_box"},
    73: {"suite": "libero_spatial", "dim": "dim3_change_object", "bddl": "ext_06_cookie_box_to_stove"},
    74: {"suite": "libero_spatial", "dim": "dim3_change_object", "bddl": "ext_08_cookie_box_to_ramekin"},
    # dim5_drawer_action
    75: {"suite": "libero_spatial", "dim": "dim5_drawer_action", "bddl": "ext_02_open_top"},
    76: {"suite": "libero_spatial", "dim": "dim5_drawer_action", "bddl": "ext_04_open_bottom"},

    # =====================================================================
    # libero_goal held-out tasks
    # =====================================================================
    # dim1_change_object
    77: {"suite": "libero_goal", "dim": "dim1_change_object", "bddl": "ext_02_wine_bottle_to_stove"},
    78: {"suite": "libero_goal", "dim": "dim1_change_object", "bddl": "ext_05_cream_cheese_to_cabinet"},
    79: {"suite": "libero_goal", "dim": "dim1_change_object", "bddl": "ext_07_cream_cheese_to_plate"},
    80: {"suite": "libero_goal", "dim": "dim1_change_object", "bddl": "ext_12_bowl_to_stove_front"},
    # dim2_change_target
    81: {"suite": "libero_goal", "dim": "dim2_change_target", "bddl": "put_the_wine_bottle_in_front_of_the_stove"},

    # =====================================================================
    # libero_object held-out tasks
    # =====================================================================
    # dim1_change_object
    82: {"suite": "libero_object", "dim": "dim1_change_object", "bddl": "ext_02_scene0_cream_cheese"},
    83: {"suite": "libero_object", "dim": "dim1_change_object", "bddl": "ext_04_scene1_butter"},
    # --- SWAPPED: 84<->63 ---
    84: {"suite": "libero_spatial", "dim": "dim1_change_bowl", "bddl": "ext_03_task3_bowl2"},
    85: {"suite": "libero_object", "dim": "dim1_change_object", "bddl": "ext_08_scene3_chocolate_pudding"},
    86: {"suite": "libero_object", "dim": "dim1_change_object", "bddl": "ext_10_scene4_milk"},
    87: {"suite": "libero_object", "dim": "dim1_change_object", "bddl": "ext_11_scene5_bbq_sauce"},
    88: {"suite": "libero_object", "dim": "dim1_change_object", "bddl": "ext_12_scene5_orange_juice"},
    89: {"suite": "libero_object", "dim": "dim1_change_object", "bddl": "ext_13_scene6_ketchup"},
    90: {"suite": "libero_object", "dim": "dim1_change_object", "bddl": "ext_14_scene6_tomato_sauce"},
    91: {"suite": "libero_object", "dim": "dim1_change_object", "bddl": "ext_15_scene7_butter"},
    92: {"suite": "libero_object", "dim": "dim1_change_object", "bddl": "ext_16_scene7_cream_cheese"},
    93: {"suite": "libero_object", "dim": "dim1_change_object", "bddl": "ext_17_scene8_orange_juice"},
    94: {"suite": "libero_object", "dim": "dim1_change_object", "bddl": "ext_18_scene8_bbq_sauce"},
    95: {"suite": "libero_object", "dim": "dim1_change_object", "bddl": "ext_19_scene8_ketchup"},
    96: {"suite": "libero_object", "dim": "dim1_change_object", "bddl": "ext_20_scene9_chocolate_pudding"},
    97: {"suite": "libero_object", "dim": "dim1_change_object", "bddl": "ext_21_scene9_butter"},
    98: {"suite": "libero_object", "dim": "dim1_change_object", "bddl": "ext_22_scene9_salad_dressing"},
}


# 16 training task IDs (tasks collected by collect_multi.sh)
TRAINING_TASK_IDS = [40, 41, 42, 43, 44, 45, 49, 50, 51, 54, 78, 79, 59, 62, 63, 64]


def get_bddl_name(task_id: int) -> str:
    """Return the BDDL filename for the given task_id."""
    return EXTENDED_TASK_MAP[task_id]["bddl"]


def get_id_to_name() -> dict:
    """Return {task_id: bddl_name} mapping, compatible with unified_eval.py's EXTENDED_TASK_ID_TO_NAME."""
    return {tid: info["bddl"] for tid, info in EXTENDED_TASK_MAP.items()}


def get_training_ids_str() -> str:
    """Return comma-separated training task ID string for the --task_id argument."""
    return ",".join(str(t) for t in TRAINING_TASK_IDS)
