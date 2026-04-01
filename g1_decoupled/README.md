# g1_decoupled

This directory contains the G1 training stack used in `RL_transformer`.

## Fresh Setup

Clone the repo on the lab PC:

```bash
git clone https://github.com/shuayyy/g1-docoupled.git
cd g1-docoupled/g1_decoupled
```

Create the Python environment:

```bash
conda create -n HIT python=3.8
conda activate HIT
```

Install the Python dependencies used across the current G1 stack:

```bash
python -m pip install \
  torch torchvision \
  numpy==1.23.5 scipy matplotlib pillow \
  wandb ipython pyyaml opencv-python einops packaging \
  h5py chardet mujoco==2.3.7 dm_control==1.0.14
```

Optional packages from the broader HumanPlus/HIT setup:

```bash
python -m pip install \
  pyquaternion rospkg pexpect getkey h5py_cache
```

Notes:

- `numpy==1.23.5` is used here instead of the old `numpy==1.20` pin because the older pin causes compatibility issues with newer Pillow builds.
- `mujoco` and `dm_control` are needed for the MuJoCo sim2sim path.
- `wandb` is optional at runtime, but installed here for convenience.

## Isaac Gym

Install Isaac Gym separately, for example under:

```text
$HOME/isaacgym
```

Make sure the shell can find it:

```bash
export PYTHONPATH="$PYTHONPATH:$HOME/isaacgym/python"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

If you want this permanent, add those lines to `~/.bashrc`.

## Local Package Install

Install the local RL packages:

```bash
cd RL_transformer/rsl_rl
python -m pip install -e .

cd ../legged_gym
python -m pip install -e .
```

If you later use the legacy HIT imitation code under `humanplus/HIT`, install its local DETR package too:

```bash
cd /path/to/humanplus/HIT/detr
python -m pip install -e .
```

## Dataset

The G1 motion dataset config is:

```text
RL_transformer/legged_gym/motion_data_configs/twist2_dataset.yaml
```

This should point to the TWIST2 G1-retargeted motion data on that machine.

If the dataset root changes on the lab PC, update that YAML or the path logic in:

```text
RL_transformer/legged_gym/legged_gym/utils/human.py
```

## Test The Environment

From:

```bash
cd RL_transformer/legged_gym
```

Run:

```bash
python legged_gym/tests/test_env.py --task g1 --headless
```

If this works, the environment, asset loading, and motion loading are set up correctly.

## Train

From:

```bash
cd RL_transformer/legged_gym
```

Run:

```bash
python legged_gym/scripts/train.py --task g1 --headless
```

Optional WandB:

```bash
USE_WANDB=1 python legged_gym/scripts/train.py --task g1 --headless
```

Checkpoints are saved under:

```text
RL_transformer/legged_gym/logs/rough_g1/<run_name>/
```

## Sim2Sim Deploy

The MuJoCo sim deploy files are:

```text
RL_transformer/deploy_real/server_low_level_g1_rl_transformer_sim.py
RL_transformer/deploy_real/configs/g1_rl_transformer.yaml
RL_transformer/sim2sim.sh
```

Run:

```bash
cd RL_transformer
bash sim2sim.sh
```

## Main Files

- `RL_transformer/legged_gym/legged_gym/envs/g1/g1.py`
- `RL_transformer/legged_gym/legged_gym/envs/g1/g1_config.py`
- `RL_transformer/legged_gym/legged_gym/utils/human.py`
