import argparse
import importlib.util
import time
from collections import deque
from pathlib import Path

import mujoco
import mujoco.viewer as mjv
import numpy as np
import torch
import yaml


def quat_to_euler_xyz(quat_wxyz):
    w, x, y, z = quat_wxyz
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return np.array([roll, pitch, yaw], dtype=np.float32)


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class TorchscriptPolicyWrapper:
    def __init__(self, policy_path, device):
        self.device = torch.device(device)
        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()

    def __call__(self, obs_tensor):
        with torch.no_grad():
            return self.policy(obs_tensor)


def resolve_path(base_dir, path_like):
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


class RLTransformerSimDeploy:
    def __init__(self, config_path):
        self.config_path = Path(config_path).resolve()
        with open(self.config_path, "r", encoding="utf-8") as file_obj:
            self.cfg = yaml.safe_load(file_obj)

        self.base_dir = self.config_path.parent
        self.device = self.cfg["device"]
        self.num_actions = self.cfg["num_actions"]
        self.obs_context_len = self.cfg["obs_context_len"]
        self.num_obs_single = self.cfg["num_obs_single"]
        self.policy_frequency = self.cfg["policy_frequency"]
        self.sim_dt = self.cfg["sim_dt"]
        self.policy_decimation = max(1, int(round(1.0 / (self.policy_frequency * self.sim_dt))))

        xml_path = resolve_path(self.base_dir, self.cfg["xml_path"])
        policy_path = resolve_path(self.base_dir, self.cfg["policy_path"])
        motion_file = resolve_path(self.base_dir, self.cfg["motion_file"])

        self.default_dof_pos = np.asarray(self.cfg["default_angles"], dtype=np.float32)
        self.kps = np.asarray(self.cfg["kps"], dtype=np.float32)
        self.kds = np.asarray(self.cfg["kds"], dtype=np.float32)
        self.torque_limits = np.asarray(self.cfg["torque_limits"], dtype=np.float32)
        self.action_scale = np.asarray(self.cfg["action_scale"], dtype=np.float32)

        self.obs_scales = self.cfg["obs_scales"]
        self.commands_scale = np.asarray(self.cfg["commands_scale"], dtype=np.float32)
        self.command = np.array(
            [
                self.cfg["command_lin_vel_x"],
                self.cfg["command_lin_vel_y"],
                0.0,
                self.cfg["command_heading"],
            ],
            dtype=np.float32,
        )

        # Import the motion loader directly from file so MuJoCo deploy does not depend on Isaac Gym package imports.
        human_utils_path = (self.base_dir.parent / "legged_gym" / "legged_gym" / "utils" / "human.py").resolve()
        human_spec = importlib.util.spec_from_file_location("rl_transformer_human_utils", human_utils_path)
        human_module = importlib.util.module_from_spec(human_spec)
        human_spec.loader.exec_module(human_module)
        TargetJTMotionLib = human_module.TargetJTMotionLib

        self.motion_lib = TargetJTMotionLib(
            str(motion_file),
            "cpu",
            self.num_actions,
            max_cached_motions=self.cfg.get("max_cached_motions", 64),
        )
        self.motion_frequency = float(self.cfg["motion_frequency"])
        self.target_delay = float(self.cfg["target_delay"])
        self.resample_motion_on_end = bool(self.cfg.get("resample_motion_on_end", True))

        self.policy = TorchscriptPolicyWrapper(str(policy_path), self.device)

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)
        self.viewer = mjv.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)

        self.obs_history = deque(maxlen=self.obs_context_len)
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)

        self.current_motion_id = None
        self.current_motion_len = None
        self.motion_step = 0

        self.mujoco_default_qpos = np.concatenate(
            [
                np.array([0.0, 0.0, 0.793], dtype=np.float32),
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                self.default_dof_pos,
            ]
        )

        self.reset()

    def sample_motion(self):
        motion_id = int(self.motion_lib.sample_motions(1)[0].item())
        motion_len = int(self.motion_lib.get_motion_lengths(torch.tensor([motion_id]))[0].item())
        self.current_motion_id = motion_id
        self.current_motion_len = motion_len
        self.motion_step = 0

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.mujoco_default_qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.last_action[:] = 0.0
        self.obs_history.clear()
        zero_obs = np.zeros(self.num_obs_single, dtype=np.float32)
        for _ in range(self.obs_context_len):
            self.obs_history.append(zero_obs.copy())

        self.sample_motion()

    def extract_state(self):
        dof_pos = self.data.qpos[7:7 + self.num_actions].copy()
        dof_vel = self.data.qvel[6:6 + self.num_actions].copy()
        quat_wxyz = self.data.qpos[3:7].copy()
        ang_vel = self.data.qvel[3:6].copy()
        return dof_pos, dof_vel, quat_wxyz, ang_vel

    def get_motion_targets(self):
        current_frame = min(int(round(self.motion_step * self.policy_frequency / self.motion_frequency)), self.current_motion_len - 1)
        delayed_frame = max(0, int(round((self.motion_step / self.policy_frequency - self.target_delay) * self.motion_frequency)))
        delayed_frame = min(delayed_frame, self.current_motion_len - 1)

        motion_ids = torch.tensor([self.current_motion_id], dtype=torch.long)
        current = self.motion_lib.get_frames(motion_ids, torch.tensor([current_frame], dtype=torch.long))[0].cpu().numpy()
        delayed = self.motion_lib.get_frames(motion_ids, torch.tensor([delayed_frame], dtype=torch.long))[0].cpu().numpy()
        return current, delayed, current_frame

    def build_obs(self):
        dof_pos, dof_vel, quat_wxyz, ang_vel = self.extract_state()
        rpy = quat_to_euler_xyz(quat_wxyz)
        heading = rpy[2]
        if self.cfg["heading_command"]:
            self.command[2] = np.clip(0.5 * wrap_to_pi(self.command[3] - heading), -1.0, 1.0)

        _, delayed_target_jt, current_frame = self.get_motion_targets()
        obs = np.concatenate(
            [
                rpy[:2] * self.obs_scales["orn"],
                ang_vel * self.obs_scales["ang_vel"],
                self.command[:3] * self.commands_scale[:3],
                (dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
                dof_vel * self.obs_scales["dof_vel"],
                self.last_action,
                delayed_target_jt * self.obs_scales["dof_pos"],
            ]
        ).astype(np.float32)
        if obs.shape[0] != self.num_obs_single:
            raise ValueError(f"Expected observation dim {self.num_obs_single}, got {obs.shape[0]}")
        self.obs_history.append(obs)
        obs_hist = np.stack(self.obs_history, axis=0)
        return obs_hist, current_frame

    def run(self, sim_duration=120.0):
        steps = int(sim_duration / self.sim_dt)
        for sim_step in range(steps):
            if sim_step % self.policy_decimation == 0:
                obs_hist, current_frame = self.build_obs()
                obs_tensor = torch.from_numpy(obs_hist).float().unsqueeze(0).to(self.device)
                raw_action = self.policy(obs_tensor).detach().cpu().numpy().squeeze(0)
                self.last_action = raw_action.astype(np.float32)

                scaled_actions = np.clip(raw_action, -10.0, 10.0) * self.action_scale
                target_dof_pos = scaled_actions + self.default_dof_pos
                dof_pos, dof_vel, _, _ = self.extract_state()
                pd_torque = self.kps * (target_dof_pos - dof_pos) - self.kds * dof_vel
                self.data.ctrl[:] = np.clip(pd_torque, -self.torque_limits, self.torque_limits)

                self.motion_step += 1
                if current_frame >= self.current_motion_len - 1 and self.resample_motion_on_end:
                    self.sample_motion()

            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "configs" / "g1_rl_transformer.yaml"),
    )
    parser.add_argument("--duration", type=float, default=120.0)
    args = parser.parse_args()

    controller = RLTransformerSimDeploy(args.config)
    controller.run(sim_duration=args.duration)


if __name__ == "__main__":
    main()
