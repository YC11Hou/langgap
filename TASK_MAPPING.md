# Task Mapping Table

## Overview

| Type | Task Count | Episodes | Dataset |
|------|--------|----------|--------|
| **Official Tasks (Task 0-39)** | 40 | 1693 | `<YOUR_HF_USERNAME>/langgap_full` |
| **Extended Training Tasks (Task 40-62)** | 16 | 2400 | `<YOUR_HF_USERNAME>/langgap_full` |
| **Extended Held-out Tasks (Task 63-98)** | 36 | — | — |
| **Total** | **99** | **4093** | |

---

## Official 40 Tasks (Task 0-39)

### libero_spatial (Task 0-9)

| Task ID | Episodes | Instruction |
|---------|----------|-------------|
| 0 | 38 | pick up the black bowl between the plate and the ramekin and place it on the plate |
| 1 | 36 | pick up the black bowl next to the ramekin and place it on the plate |
| 2 | 34 | pick up the black bowl from the table center and place it on the plate |
| 3 | 41 | pick up the black bowl on the cookie box and place it on the plate |
| 4 | 43 | pick up the black bowl in the top drawer of the cabinet and place it on the plate |
| 5 | 33 | pick up the black bowl on the ramekin and place it on the plate |
| 6 | 29 | pick up the black bowl next to the cookie box and place it on the plate |
| 7 | 49 | pick up the black bowl on the stove and place it on the plate |
| 8 | 35 | pick up the black bowl next to the plate and place it on the plate |
| 9 | 41 | pick up the black bowl on the wooden cabinet and place it on the plate |

### libero_goal (Task 10-19)

| Task ID | Episodes | Instruction |
|---------|----------|-------------|
| 10 | 49 | put the bowl on the plate |
| 11 | 36 | put the bowl on the stove |
| 12 | 36 | put the bowl on top of the cabinet |
| 13 | 40 | put the wine bottle on top of the cabinet |
| 14 | 47 | put the wine bottle on the rack |
| 15 | 33 | put the cream cheese in the bowl |
| 16 | 50 | turn on the stove |
| 17 | 48 | open the top drawer and put the bowl inside |
| 18 | 46 | open the middle drawer of the cabinet |
| 19 | 43 | push the plate to the front of the stove |

### libero_object (Task 20-29)

| Task ID | Episodes | Instruction |
|---------|----------|-------------|
| 20 | 45 | pick up the alphabet soup and place it in the basket |
| 21 | 45 | pick up the cream cheese and place it in the basket |
| 22 | 45 | pick up the salad dressing and place it in the basket |
| 23 | 46 | pick up the bbq sauce and place it in the basket |
| 24 | 44 | pick up the ketchup and place it in the basket |
| 25 | 45 | pick up the tomato sauce and place it in the basket |
| 26 | 47 | pick up the butter and place it in the basket |
| 27 | 45 | pick up the milk and place it in the basket |
| 28 | 42 | pick up the chocolate pudding and place it in the basket |
| 29 | 50 | pick up the orange juice and place it in the basket |

### libero_10 (Task 30-39)

| Task ID | Episodes | Instruction |
|---------|----------|-------------|
| 30 | 46 | put the frying pan on the stove |
| 31 | 42 | put the moka pot on the stove |
| 32 | 39 | turn on the stove and put the frying pan on it |
| 33 | 35 | put the black bowl in the bottom drawer of the cabinet and close it |
| 34 | 45 | put the white mug on the left plate and put the yellow and white mug on the right plate |
| 35 | 43 | put the white mug on the right plate and put the yellow and white mug on the left plate |
| 36 | 47 | pick up the book and place it in the back compartment of the caddy |
| 37 | 45 | pick up the book and place it in the front compartment of the caddy |
| 38 | 46 | pick up the book and place it in the right compartment of the caddy |
| 39 | 44 | push the plate to the front of the stove and put the bowl on the plate |

---

## Extended Training Tasks (Task 40-62, 23 tasks)

### libero_spatial Extended (Task 40-53, 14 tasks)

| Task ID | Episodes | Instruction | BDDL File |
|---------|----------|-------------|-----------|
| 40 | 50 | pick up the black bowl on the right of the ramekin... | dim1_change_bowl/ext_01_task0_bowl2 |
| 41 | 50 | pick up the black bowl on the right of the plate... | dim1_change_bowl/ext_02_task2_bowl2 |
| 42 | 50 | pick up the black bowl on top of the wooden cabinet... | dim1_change_bowl/ext_04_task4_bowl2 |
| 43 | 50 | ...place it on the stove | dim2_change_target/ext_01_task0_to_stove |
| 44 | 50 | ...place it on the cabinet | dim2_change_target/ext_02_task0_to_cabinet |
| 45 | 50 | ...place it on the ramekin | dim2_change_target/ext_04_task2_to_ramekin |
| 46 | 50 | ...place it on the cabinet | dim2_change_target/ext_07_task7_to_cabinet |
| 47 | 50 | ...place it on the stove | dim2_change_target/ext_09_task8_to_stove |
| 48 | 50 | ...place it on the stove | dim2_change_target/ext_11_task9_to_stove |
| 49 | 50 | pick up the ramekin and place it on the plate | dim3_change_object/ext_01_ramekin_to_plate |
| **50** | **50** | **pick up the ramekin and place it on the cabinet** | **dim3_change_object/ext_03_ramekin_to_cabinet** |
| 51 | 50 | pick up the cookie box and place it on the plate | dim3_change_object/ext_05_cookie_box_to_plate |
| 52 | 50 | pick up the cookie box and place it on the cabinet | dim3_change_object/ext_07_cookie_box_to_cabinet |
| 53 | 50 | open the middle drawer... | dim5_drawer_action/ext_03_open_middle |

### libero_goal Extended (Task 54-57, 4 tasks)

| Task ID | Episodes | Instruction | BDDL File |
|---------|----------|-------------|-----------|
| 54 | 50 | put the cream cheese on the stove | dim1_change_object/ext_03_cream_cheese_to_stove |
| 55 | 100* | put the wine bottle on the plate | dim1_change_object/ext_06_wine_bottle_to_plate |
| 56 | 50 | open the top drawer and put the cream cheese inside | dim1_change_object/ext_09_cream_cheese_to_top_drawer |
| 57 | 50 | open the bottom drawer... | dim1_change_object/ext_11_bottom_drawer |

### libero_object Extended (Task 58-62, 5 tasks)

| Task ID | Episodes | Instruction | Scene |
|---------|----------|-------------|-------|
| 58 | 100* | pick up the salad dressing... | Scene 0 |
| 59 | 100* | pick up the alphabet soup... | Scene 1 |
| 60 | — | *(SWAPPED -> spatial dim1_change_bowl/ext_05_task8_bowl2, held-out)* | — |
| 61 | 50 | pick up the ketchup... | Scene 3 |
| 62 | 50 | pick up the bbq sauce... | Scene 4 |
| 63 | 150 | pick up the tomato sauce... *(SWAPPED from 84)* | Scene 2 |
| 64 | 150 | pick up the milk... *(SWAPPED from 60)* | Scene 2 |

> *Note: Tasks with Episodes=100 include MimicGen augmented data.
> **SWAP Record (2026-02-17)**: 63<->84, 64<->60 swapped to match data already collected by collect_multi.sh. See `task_registry.py` for details.

---

## Extended Held-out Tasks (Task 63-98, 36 tasks)

These tasks are used for evaluation only (not included in training), to test model generalization on unseen task variants.

### libero_spatial held-out (Task 63-76, 14 tasks)

#### dim1_change_bowl

| Task ID | Instruction | BDDL File |
|---------|-------------|-----------|
| 60 | pick up the black bowl next to the ramekin and place it on the plate | dim1_change_bowl/ext_05_task8_bowl2 *(SWAPPED from 64)* |
| 63 | *(SWAPPED -> object dim1_change_object/ext_06_scene2_tomato_sauce, training)* | — |
| 64 | *(SWAPPED -> object dim1_change_object/ext_05_scene2_milk, training)* | — |
| 84 | pick up the black bowl on top of the wooden cabinet and place it on the plate | dim1_change_bowl/ext_03_task3_bowl2 *(SWAPPED from 63)* |

#### dim2_change_target

| Task ID | Instruction | BDDL File |
|---------|-------------|-----------|
| 65 | pick up the black bowl from the middle of the table and place it on the stove | dim2_change_target/ext_03_task2_to_stove |
| 66 | pick up the black bowl on the cookie box and place it on the stove | dim2_change_target/ext_05_task3_to_stove |
| 67 | pick up the black bowl on the cookie box and place it on the cabinet | dim2_change_target/ext_06_task3_to_cabinet |
| 68 | pick up the black bowl on the stove and place it on the ramekin | dim2_change_target/ext_08_task7_to_ramekin |
| 69 | pick up the black bowl next to the plate and place it on the cabinet | dim2_change_target/ext_10_task8_to_cabinet |
| 70 | pick up the black bowl on the wooden cabinet and place it on the ramekin | dim2_change_target/ext_12_task9_to_ramekin |

#### dim3_change_object

| Task ID | Instruction | BDDL File |
|---------|-------------|-----------|
| 71 | pick up the ramekin and place it on the stove | dim3_change_object/ext_02_ramekin_to_stove |
| 72 | pick up the ramekin and place it on the cookie box | dim3_change_object/ext_04_ramekin_to_cookie_box |
| 73 | pick up the cookie box and place it on the stove | dim3_change_object/ext_06_cookie_box_to_stove |
| 74 | pick up the cookie box and place it on the ramekin | dim3_change_object/ext_08_cookie_box_to_ramekin |

#### dim5_drawer_action

| Task ID | Instruction | BDDL File |
|---------|-------------|-----------|
| 75 | open the top drawer of the cabinet | dim5_drawer_action/ext_02_open_top |
| 76 | open the bottom drawer of the cabinet | dim5_drawer_action/ext_04_open_bottom |

### libero_goal held-out (Task 77-81, 5 tasks)

#### dim1_change_object

| Task ID | Instruction | BDDL File |
|---------|-------------|-----------|
| 77 | put the wine bottle on the stove | dim1_change_object/ext_02_wine_bottle_to_stove |
| 78 | put the cream cheese on top of the cabinet | dim1_change_object/ext_05_cream_cheese_to_cabinet |
| 79 | put the cream cheese on the plate | dim1_change_object/ext_07_cream_cheese_to_plate |
| 80 | put the bowl in front of the stove | dim1_change_object/ext_12_bowl_to_stove_front |

#### dim2_change_target

| Task ID | Instruction | BDDL File |
|---------|-------------|-----------|
| 81 | put the wine bottle in front of the stove | dim2_change_target/put_the_wine_bottle_in_front_of_the_stove |

### libero_object held-out (Task 82-98, 17 tasks)

#### dim1_change_object

| Task ID | Instruction | BDDL File |
|---------|-------------|-----------|
| 82 | Pick the cream cheese and place it in the basket | dim1_change_object/ext_02_scene0_cream_cheese |
| 83 | Pick the butter and place it in the basket | dim1_change_object/ext_04_scene1_butter |
| 84 | *(SWAPPED -> spatial dim1_change_bowl/ext_03_task3_bowl2, see spatial held-out)* | — |
| 85 | Pick the chocolate pudding and place it in the basket | dim1_change_object/ext_08_scene3_chocolate_pudding |
| 86 | Pick the milk and place it in the basket | dim1_change_object/ext_10_scene4_milk |
| 87 | Pick the bbq sauce and place it in the basket | dim1_change_object/ext_11_scene5_bbq_sauce |
| 88 | Pick the orange juice and place it in the basket | dim1_change_object/ext_12_scene5_orange_juice |
| 89 | Pick the ketchup and place it in the basket | dim1_change_object/ext_13_scene6_ketchup |
| 90 | Pick the tomato sauce and place it in the basket | dim1_change_object/ext_14_scene6_tomato_sauce |
| 91 | Pick the butter and place it in the basket | dim1_change_object/ext_15_scene7_butter |
| 92 | Pick the cream cheese and place it in the basket | dim1_change_object/ext_16_scene7_cream_cheese |
| 93 | Pick the orange juice and place it in the basket | dim1_change_object/ext_17_scene8_orange_juice |
| 94 | Pick the bbq sauce and place it in the basket | dim1_change_object/ext_18_scene8_bbq_sauce |
| 95 | Pick the ketchup and place it in the basket | dim1_change_object/ext_19_scene8_ketchup |
| 96 | Pick the chocolate pudding and place it in the basket | dim1_change_object/ext_20_scene9_chocolate_pudding |
| 97 | Pick the butter and place it in the basket | dim1_change_object/ext_21_scene9_butter |
| 98 | Pick the salad dressing and place it in the basket | dim1_change_object/ext_22_scene9_salad_dressing |

---

## Data Location

### HuggingFace Datasets

| Dataset | Contents | Status |
|--------|------|------|
| `<YOUR_HF_USERNAME>/task50_scripted` | Task 50 scripted, 50 eps | Exp 1 |
| `<YOUR_HF_USERNAME>/langgap_6` | 1 orig + 5 ext spatial, 300 eps | Exp 2 |
| `<YOUR_HF_USERNAME>/langgap_45` | 40 orig + 5 ext spatial | Exp 3 |
| `<YOUR_HF_USERNAME>/langgap_ext` | 16 ext, ~2400 eps | Exp 4 |
| `<YOUR_HF_USERNAME>/langgap_full` | 40 orig + 16 ext, 4093 eps | Exp 5 |

> `<YOUR_HF_USERNAME>/langgap` (63 tasks, 2996 eps) is deprecated -- extended task data came from early keyboard collection with poor quality.

---

## Training Task Configuration

Training script parameters:
```bash
--task=0-39    # Official 40 tasks
--task=10      # Single task
--task=50      # Task 50 extended task
--task=all     # All 63 tasks
```

See `README.md` for training commands.

---

**Last Updated**: 2026-02-12
