# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

# Old direct WandB import kept for reference:
# import wandb
try:
    import wandb
except ImportError:  # pragma: no cover - depends on local env
    wandb = None

def train(args):
    # Old direct WandB init kept for reference:
    # wandb.init(project='humanoid', name=args.run_name, entity="zfu")
    # task_path = task_registry.task_paths[args.task]
    # wandb.save(os.path.join(LEGGED_GYM_ENVS_DIR, task_path, f"{args.task}_config.py"), policy="now")
    # wandb.save(os.path.join(LEGGED_GYM_ENVS_DIR, task_path, f"{args.task}.py"), policy="now")
    # wandb.save(LEGGED_GYM_ROOT_DIR + "../rsl_rl/modules/actor_critic.py", policy="now")
    # wandb.save(LEGGED_GYM_ROOT_DIR + "../rsl_rl/algorithms/ppo.py", policy="now")
    # wandb.save(LEGGED_GYM_ROOT_DIR + "../rsl_rl/runners/on_policy_runner.py", policy="now")

    use_wandb = os.environ.get("USE_WANDB", "0") == "1"
    if use_wandb and wandb is not None:
        try:
            wandb.init(project='humanoid', name=args.run_name)
            task_path = task_registry.task_paths[args.task]
            wandb.save(os.path.join(LEGGED_GYM_ENVS_DIR, task_path, f"{args.task}_config.py"), policy="now")
            wandb.save(os.path.join(LEGGED_GYM_ENVS_DIR, task_path, f"{args.task}.py"), policy="now")
        except Exception as error:
            print(f"WandB disabled after init error: {error}")

    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
