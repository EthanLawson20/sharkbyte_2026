"""This is the real arm. same four methods as StubArm so scheduler.py doesn't care which one it gets.
each skill is either a trained checkpoint or a recorded episode to replay (my fallback if
training doesn't pan out)."""

import time

import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.factory import get_policy_class
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device

# same values as scripts/env.sh
PORT = "/dev/ttyACM0"
ROBOT_ID = "pill_follower"
CAM_TOP = 0
CAM_WRIST = 1
FPS = 30

# skill name -> how to run it
# "policy": checkpoint dir + task string, "replay": dataset root + episode number
SKILLS = {
    "pick_bottle": {"replay": "datasets/pick_bottle", "episode": 0},
    "dispense":    {"replay": "datasets/dispense", "episode": 0},
    "home":        {"replay": "datasets/home", "episode": 0},
    "place": {
        "policy": "outputs/pill_all_act/checkpoints/last/pretrained_model",
        "task": "Pick the pill from the tray and place it in the {day} cell",
        "seconds": 15,
    },
}


class RealArm:
    def __init__(self):
        cams = {
            "top": OpenCVCameraConfig(index_or_path=CAM_TOP, width=640, height=480, fps=FPS),
            "wrist": OpenCVCameraConfig(index_or_path=CAM_WRIST, width=640, height=480, fps=FPS),
        }
        self.robot = SO101Follower(SO101FollowerConfig(port=PORT, id=ROBOT_ID, cameras=cams))
        self.robot.connect()
        self.obs_features = hw_to_dataset_features(self.robot.observation_features, "observation")
        self.policies = {}  # cache so I don't reload the checkpoint every cell
        self.log = []

    # the four methods scheduler.py calls

    def pick_bottle(self, pill):
        self._run("pick_bottle", pill=pill)

    def dispense_to_tray(self, count):
        self._run("dispense", count=count)

    def place_in_cell(self, day, slot):
        self._run("place", day=day.capitalize(), slot=slot)

    def home(self):
        self._run("home")

    def close(self):
        self.robot.disconnect()

    # internals

    def _run(self, name, **fmt):
        skill = SKILLS[name]
        self.log.append(f"{name} {fmt}")
        print(f"[arm] {name} {fmt}")
        if "policy" in skill:
            self._run_policy(skill["policy"], skill["task"].format(**fmt), skill["seconds"])
        else:
            self._replay(skill["replay"], skill["episode"])

    def _load_policy(self, path):
        if path not in self.policies:
            # reads the type out of the checkpoint so this works for act or smolvla
            import json, os
            with open(os.path.join(path, "config.json")) as f:
                ptype = json.load(f)["type"]
            cls = get_policy_class(ptype)
            policy = cls.from_pretrained(path)
            policy.eval()
            self.policies[path] = policy
        return self.policies[path]

    def _run_policy(self, path, task, seconds):
        policy = self._load_policy(path)
        policy.reset()
        device = get_safe_torch_device(policy.config.device)
        use_amp = getattr(policy.config, "use_amp", False)
        n = int(seconds * FPS)
        for _ in range(n):
            t0 = time.perf_counter()
            obs = self.robot.get_observation()
            frame = build_dataset_frame(self.obs_features, obs, prefix="observation")
            with torch.no_grad():
                a = predict_action(frame, policy, device, use_amp, task=task,
                                   robot_type=self.robot.robot_type)
            action = {k: a[i].item() for i, k in enumerate(self.robot.action_features)}
            self.robot.send_action(action)
            # hold the loop at FPS
            time.sleep(max(0, 1 / FPS - (time.perf_counter() - t0)))

    def _replay(self, root, episode):
        # plays back the joint positions that were recorded
        ds = LeRobotDataset(repo_id=f"local/{root.split('/')[-1]}", root=root, episodes=[episode])
        keys = self.robot.action_features
        for i in range(len(ds)):
            t0 = time.perf_counter()
            a = ds[i]["action"]
            self.robot.send_action({k: a[j].item() for j, k in enumerate(keys)})
            time.sleep(max(0, 1 / ds.fps - (time.perf_counter() - t0)))


if __name__ == "__main__":
    # just a quick test: python real_arm.py place monday
    import sys
    arm = RealArm()
    try:
        if sys.argv[1] == "place":
            arm.place_in_cell(sys.argv[2], "am")
        else:
            arm._run(sys.argv[1])
    finally:
        arm.close()