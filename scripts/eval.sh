#!/usr/bin/env bash
# usage: ./eval.sh monday "Pick the pill from the tray and place it in the Monday cell" [smolvla|act]
# runs the policy 10 times, then we count how many actually land in the right cell
set -e
source "$(dirname "$0")/env.sh"

CELL=$1
TASK=$2
POLICY=${3:-smolvla}

lerobot-record \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID" \
  --robot.cameras="$CAMERAS" \
  --policy.path="outputs/pill_${CELL}_${POLICY}/checkpoints/last/pretrained_model" \
  --dataset.repo_id="$HF_USER/eval_pill_$CELL" \
  --dataset.root="$DATA_ROOT/eval_pill_$CELL" \
  --dataset.single_task="$TASK" \
  --dataset.num_episodes=10 \
  --dataset.episode_time_s=20 \
  --dataset.push_to_hub=false