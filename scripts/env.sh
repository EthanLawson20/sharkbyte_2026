#!/usr/bin/env bash
# my ports and camera numbers. running lerobot-find-port and lerobot-find-cameras will give the correct output

export HF_USER="local"                 # lerobot wants a user/name string, this is just a label
export DATA_ROOT="./datasets"          # recordings go here, gitignored

export FOLLOWER_PORT="/dev/ttyACM0"    # will have to specify COM port
export LEADER_PORT="/dev/ttyACM1"
export FOLLOWER_ID="pill_follower"     # has to match the calibration json name
export LEADER_ID="pill_leader"

# which webcam is which
export CAM_TOP=0
export CAM_WRIST=1

export CAMERAS="{ top: {type: opencv, index_or_path: $CAM_TOP, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: $CAM_WRIST, width: 640, height: 480, fps: 30} }"