#!/bin/bash

SCRIPT_DIR=$(dirname "$(realpath "$0")")

cd "${SCRIPT_DIR}/deploy_real" || exit 1

python server_low_level_g1_rl_transformer_sim.py \
    --config ./configs/g1_rl_transformer.yaml
