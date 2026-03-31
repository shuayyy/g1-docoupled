#TOCO: @shuaiyy creeate base files for 1 trainig

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
import time
from warnings import WarningMessage
import numpy as np
import os, sys
from copy import deepcopy

from isaacgym.torch_utils import *
from isaacgym import gymapi, gymtorch

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float, euler_from_quat
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.human import load_target_jt
from .g1_config import G1RoughCfg
import IPython; e = IPython.embed

def sample_int_from_float(x):
    if int(x) == x:
        return int(x)
    return int(x) if np.random.rand() < (x - int(x)) else int(x) + 1
class G1():
    def __init__(self, cfg: G1RoughCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        self._super_init(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        
        # human retargeted poses
        self._init_target_jt()

        self.init_done = True

        def _super_init(self, cfg, sim_params, physics_engine, sim_device, headless):
            """ Calls the super class init and creates the simulation, terrain and environments by calling create_sim().

            Args:
                cfg (Dict): Environment config file
                sim_params (gymapi.SimParams): simulation parameters
                physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
                device_type (string): 'cuda' or 'cpu'
                device_id (int): 0, 1, ...
                headless (bool): Run without rendering if True  

            """
            self.gym = gymapi.acquire_gym()
            self.sim_params = sim_params
            self.physics_engine = physics_engine
            self.sim_device = sim_device

            self.headless = headless


            pass
