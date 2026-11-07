#!/usr/bin/env bash
# ./train.sh monday        -> smolvla, run on the laptop or colab
# ./train.sh monday act    -> act, lighter, runs on the pi 5
# reads ./datasets/pill_<cell>, writes ./outputs/pill_<cell>_<policy>
# If we run out of memory on the GPU, then just drop BATCH to 8 or 4 and bump steps up
set -e
source "$(dirname "$0")/env.sh"

CELL=$1
POLICY=${2:-smolvla}

if [ "$POLICY" = "act" ]; then
  POLICY_ARGS="--policy.type=act"
  BATCH=8
else
  POLICY_ARGS="--policy.path=lerobot/smolvla_base"
  BATCH=16
fi

lerobot-train \
  $POLICY_ARGS \
  --dataset.repo_id="$HF_USER/pill_$CELL" \
  --dataset.root="$DATA_ROOT/pill_$CELL" \
  --batch_size=$BATCH \
  --steps=20000 \
  --output_dir="outputs/pill_${CELL}_${POLICY}" \
  --job_name="pill_${CELL}_${POLICY}" \
  --wandb.enable=false \
  --policy.push_to_hub=false