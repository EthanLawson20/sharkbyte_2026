#!/usr/bin/env bash
# usage: ./record.sh monday "Pick the pill from the tray and place it in the Monday cell" 60
# saves to ./datasets/pill_monday
set -e
source "$(dirname "$0")/env.sh"

CELL=$1
TASK=$2
EPISODES=${3:-60}

lerobot-record \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID" \
  --robot.cameras="$CAMERAS" \
  --teleop.type=so101_leader \
  --teleop.port="$LEADER_PORT" \
  --teleop.id="$LEADER_ID" \
  --dataset.repo_id="$HF_USER/pill_$CELL" \
  --dataset.root="$DATA_ROOT/pill_$CELL" \
  --dataset.single_task="$TASK" \
  --dataset.num_episodes="$EPISODES" \
  --dataset.episode_time_s=20 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=false