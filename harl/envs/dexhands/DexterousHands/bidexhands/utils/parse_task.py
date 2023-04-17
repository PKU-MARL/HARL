# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from harl.envs.dexhands.DexterousHands.bidexhands.tasks.shadow_hand_over import (
    ShadowHandOver,
)
from harl.envs.dexhands.DexterousHands.bidexhands.tasks.shadow_hand_catch_underarm import (
    ShadowHandCatchUnderarm,
)
from harl.envs.dexhands.DexterousHands.bidexhands.tasks.shadow_hand_two_catch_underarm import (
    ShadowHandTwoCatchUnderarm,
)
from harl.envs.dexhands.DexterousHands.bidexhands.tasks.shadow_hand_catch_abreast import (
    ShadowHandCatchAbreast,
)
from harl.envs.dexhands.DexterousHands.bidexhands.tasks.shadow_hand_lift_underarm import (
    ShadowHandLiftUnderarm,
)
from harl.envs.dexhands.DexterousHands.bidexhands.tasks.shadow_hand_catch_over2underarm import (
    ShadowHandCatchOver2Underarm,
)
from harl.envs.dexhands.DexterousHands.bidexhands.tasks.shadow_hand_door_close_inward import (
    ShadowHandDoorCloseInward,
)
from harl.envs.dexhands.DexterousHands.bidexhands.tasks.shadow_hand_door_close_outward import (
    ShadowHandDoorCloseOutward,
)
from harl.envs.dexhands.DexterousHands.bidexhands.tasks.shadow_hand_door_open_inward import (
    ShadowHandDoorOpenInward,
)
from harl.envs.dexhands.DexterousHands.bidexhands.tasks.shadow_hand_door_open_outward import (
    ShadowHandDoorOpenOutward,
)
from harl.envs.dexhands.DexterousHands.bidexhands.tasks.shadow_hand_bottle_cap import (
    ShadowHandBottleCap,
)
from harl.envs.dexhands.DexterousHands.bidexhands.tasks.shadow_hand_push_block import (
    ShadowHandPushBlock,
)
from harl.envs.dexhands.DexterousHands.bidexhands.tasks.shadow_hand_swing_cup import (
    ShadowHandSwingCup,
)
from harl.envs.dexhands.DexterousHands.bidexhands.tasks.shadow_hand_grasp_and_place import (
    ShadowHandGraspAndPlace,
)
from harl.envs.dexhands.DexterousHands.bidexhands.tasks.shadow_hand_scissors import (
    ShadowHandScissors,
)
from harl.envs.dexhands.DexterousHands.bidexhands.tasks.shadow_hand_switch import (
    ShadowHandSwitch,
)
from harl.envs.dexhands.DexterousHands.bidexhands.tasks.shadow_hand_pen import (
    ShadowHandPen,
)
from harl.envs.dexhands.DexterousHands.bidexhands.tasks.shadow_hand_re_orientation import (
    ShadowHandReOrientation,
)
from harl.envs.dexhands.DexterousHands.bidexhands.tasks.shadow_hand_kettle import (
    ShadowHandKettle,
)
from harl.envs.dexhands.DexterousHands.bidexhands.tasks.shadow_hand_block_stack import (
    ShadowHandBlockStack,
)

# Allegro hand
from harl.envs.dexhands.DexterousHands.bidexhands.tasks.allegro_hand_over import (
    AllegroHandOver,
)
from harl.envs.dexhands.DexterousHands.bidexhands.tasks.allegro_hand_catch_underarm import (
    AllegroHandCatchUnderarm,
)

# Meta
from harl.envs.dexhands.DexterousHands.bidexhands.tasks.shadow_hand_meta.shadow_hand_meta_mt1 import (
    ShadowHandMetaMT1,
)
from harl.envs.dexhands.DexterousHands.bidexhands.tasks.shadow_hand_meta.shadow_hand_meta_ml1 import (
    ShadowHandMetaML1,
)
from harl.envs.dexhands.DexterousHands.bidexhands.tasks.shadow_hand_meta.shadow_hand_meta_mt4 import (
    ShadowHandMetaMT4,
)

from harl.envs.dexhands.DexterousHands.bidexhands.tasks.hand_base.vec_task import (
    VecTaskCPU,
    VecTaskGPU,
    VecTaskPython,
    VecTaskPythonArm,
)
from harl.envs.dexhands.DexterousHands.bidexhands.tasks.hand_base.multi_vec_task import (
    MultiVecTaskPython,
    SingleVecTaskPythonArm,
)
from harl.envs.dexhands.DexterousHands.bidexhands.tasks.hand_base.multi_task_vec_task import (
    MultiTaskVecTaskPython,
)
from harl.envs.dexhands.DexterousHands.bidexhands.tasks.hand_base.meta_vec_task import (
    MetaVecTaskPython,
)
from harl.envs.dexhands.DexterousHands.bidexhands.tasks.hand_base.vec_task_rlgames import (
    RLgamesVecTaskPython,
)

from harl.envs.dexhands.DexterousHands.bidexhands.utils.config import (
    warn_task_name,
)

import json


def parse_task(args, cfg, sim_params, agent_index):
    # create native task and pass custom config
    device_id = args.device_id
    rl_device = args.rl_device

    try:
        task = eval(args.task)(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=args.physics_engine,
            device_type=args.device,
            device_id=device_id,
            headless=args.headless,
            agent_index=agent_index,
            is_multi_agent=True,
        )
    except NameError as e:
        print(e)
        warn_task_name()
    env = MultiVecTaskPython(task, rl_device)
    return env
