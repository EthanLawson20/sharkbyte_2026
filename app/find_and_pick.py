"""camera finds the pill, arm goes and gets it. no training.

teach mode: I push the arm by hand to a hover pose and a grasp pose over a pill at
a few spots, press enter each time, it saves pixel -> joints. run mode: detect the
pill in the overhead cam, blend the nearest taught poses, hover, drop, grip, lift,
home, then replay the place_slotN recording.

python find_and_pick.py teach           # ~15 min, do 6 spots
python find_and_pick.py test            # just shows what the camera sees
python find_and_pick.py pick 1          # pick and place in slot 1
python find_and_pick.py pick auto       # slot chosen by the pill's color
python find_and_pick.py check           # which taught spots look wrong
python find_and_pick.py calib           # white tape on the jaw: measure tape->jaw offset once
python find_and_pick.py loop auto       # demo mode, stays connected, enter = one pick
python find_and_pick.py wrist           # wrist cam view, set WRIST_TARGET where the pill sits between the jaws
"""

import json
import sys
import time

import cv2
import numpy as np

try:  # newer lerobot
    from lerobot.robots.so101_follower import SO101Follower as Follower, SO101FollowerConfig as FollowerConfig
except ModuleNotFoundError:  # 0.4.x layout
    from lerobot.robots import so_follower as _m
    Follower = getattr(_m, "SO101Follower", None) or getattr(_m, "SOFollower")
    FollowerConfig = getattr(_m, "SO101FollowerConfig", None) or getattr(_m, "SOFollowerConfig")
from lerobot.datasets.lerobot_dataset import LeRobotDataset

PORT = "COM4"
ROBOT_ID = "my_follower"
CAM_TOP = 2
CAM_WRIST = 1
FPS = 30
GRID_FILE = "grid.json"

# any saturated colored blob counts as a pill, mat is black and case is white so this is easy
SAT_MIN = 110
VAL_MIN = 90
MIN_AREA = 150
# only look here for pills (x0, y0, x1, y1), keeps the ones already in the case out of it
ROI = (0, 0, 340, 480)

# squeeze this much past the taught closed value so it actually holds on
GRIP_EXTRA = 14
# wrist cam servo loop at hover. where the pill should sit in the wrist image when the jaws
# are centred on it (x, y), how many px off is close enough, max tries, and gains in
# degrees per pixel. flip a gain's sign if a nudge makes the error bigger.
WRIST_TARGET = (408, 56)
WRIST_TOL = 40
WRIST_TRIES = 0  # wrist cam can't see the pill at hover height, leaving this off
K_PAN = 0.02       # x error -> shoulder_pan
K_REACH = 0.02     # y error -> elbow_flex (forward/back)
WRIST_ROI = (0, 0, 1280, 560)  # wrist cam runs at 1280x720, jaws live below y=560

# nudge added to every grasp pose, same order as JOINTS. the arm settles a hair off
# from where I read it with torque off. jaws stop above the pill -> shoulder_lift +3,
# toward/away from base -> elbow_flex +-3, left/right -> shoulder_pan +-2
GRASP_OFFSET = [0.4, 0, 0, 0, 0, 0]
# open the jaws this much wider than taught on the way down so a few mm of error doesn't matter
OPEN_EXTRA = 20
# overhead servo: white tape on top of the gripper housing (not the jaw). the top cam lines the tape up with the pill
# before descending. OFFSET is pixel (dx, dy) from the tape to the point between the jaw tips,
# measured once with `calib`. set SERVO = False to go back to grid only.
SERVO = True
MARKER_SAT_MAX = 70
MARKER_VAL_MIN = 190
MARKER_MIN_AREA = 60
SERVO_TRIES = 5
SERVO_TOL = 6      # px
JOG = 2.5          # degrees used to measure how pan and elbow move the tape in the image

# after descending, drag sideways this many pan degrees so the fixed jaw sweeps the pill into the
# moving jaw before closing. flip the sign if it pushes the pill away instead
RAKE = 2.5
# after lifting, look with the top cam. pill still on the mat = missed, try again shifted a bit
RETRIES = 3
RETRY_SHIFTS = [0.0, -1.2, 1.2]  # pan nudge per attempt
MISS_RADIUS = 60  # px, pill within this of where it started = still there
# which slot each color goes to. hue ranges are opencv 0-180
SLOT_BY_COLOR = {"red": 1, "blue": 2, "green": 3}

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# these don't decide where the gripper is, so don't interpolate them, use the median taught value
LOCKED = ["wrist_roll"]


# ---------- camera ----------

def open_cam(index, w=640, h=480):
    cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    return cap


def grab(cap):
    for _ in range(3):  # flush stale frames
        cap.read()
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("no frame from top cam")
    return frame


def find_pill(frame, roi=None):
    x0, y0, x1, y1 = roi or ROI
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, SAT_MIN, VAL_MIN), (180, 255, 255))
    mask[:, :x0] = 0; mask[:, x1:] = 0; mask[:y0, :] = 0; mask[y1:, :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > MIN_AREA]
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    m = cv2.moments(c)
    blob = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(blob, [c], -1, 255, -1)
    hue = circular_hue(hsv[..., 0][blob > 0])
    return (m["m10"] / m["m00"], m["m01"] / m["m00"], hue)


def find_marker(frame):
    # white tape on the jaw: bright and unsaturated, inside the pickup zone
    x0, y0, x1, y1 = ROI
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, MARKER_VAL_MIN), (180, MARKER_SAT_MAX, 255))
    mask[:, :x0] = 0; mask[:, x1:] = 0; mask[:y0, :] = 0; mask[y1:, :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > MARKER_MIN_AREA]
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    m = cv2.moments(c)
    return (m["m10"] / m["m00"], m["m01"] / m["m00"])


def circular_hue(hues):
    # red straddles 0/180 so a plain average lands in the middle and reads as blue. average on a circle
    a = hues.astype(float) * (2 * np.pi / 180)
    return (np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) / 2) % 180


def color_name(hue):
    if hue < 12 or hue > 160:
        return "red"
    if 35 <= hue <= 85:
        return "green"
    if 90 <= hue <= 135:
        return "blue"
    return "other"


# ---------- arm ----------

def joints(robot):
    obs = robot.get_observation()
    return [obs[f"{j}.pos"] for j in JOINTS]


def goto(robot, target, seconds=1.5):
    # ease from where I am to target so it doesn't snap
    start = joints(robot)
    n = int(seconds * FPS)
    for i in range(1, n + 1):
        a = i / n
        a = a * a * (3 - 2 * a)  # smoothstep
        pose = [s + (t - s) * a for s, t in zip(start, target)]
        robot.send_action({f"{j}.pos": v for j, v in zip(JOINTS, pose)})
        time.sleep(1 / FPS)


def set_gripper(robot, val, seconds=0.6):
    pose = joints(robot)
    pose[-1] = val
    goto(robot, pose, seconds)


def replay(robot, root, episode=0, speed=1.3):
    ds = LeRobotDataset(repo_id=f"local/{root.split('/')[-1]}", root=root, episodes=[episode])
    # read the joint table straight from the parquet, going through ds[i] decodes video and takes a minute
    acts = np.array(ds.hf_dataset.select_columns("action")["action"], dtype=float)
    # chop the part at the end where I was just holding still at home
    moving = np.abs(np.diff(acts[:, :5], axis=0)).max(1) > 0.3
    last = int(np.where(moving)[0].max()) + 15 if moving.any() else len(acts) - 1
    acts = acts[: last + 1]
    # ease to the first frame first so the replay doesn't jump
    goto(robot, acts[0].tolist(), 1.5)
    for a in acts:
        t0 = time.perf_counter()
        robot.send_action({f"{j}.pos": v for j, v in zip(JOINTS, a.tolist())})
        time.sleep(max(0, 1 / (ds.fps * speed) - (time.perf_counter() - t0)))


# ---------- grid ----------

def _fit(grid, key):
    pts = grid["points"]
    scalar = isinstance(pts[0][key], (int, float))
    Y = np.array([[p[key]] if scalar else p[key] for p in pts], dtype=float)
    A = np.array([[1.0, p["px"], p["py"]] for p in pts])
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    resid = Y - A @ coef
    return coef, resid, scalar


def blend(grid, px, py, key, k=3):
    """pixel -> pose. linear fit over all taught points, then nudge by the leftover
    error of the nearest few so it still lands exactly on spots I taught."""
    pts = grid["points"]
    coef, resid, scalar = _fit(grid, key)
    out = np.array([1.0, px, py]) @ coef
    d = np.array([((p["px"] - px) ** 2 + (p["py"] - py) ** 2) ** 0.5 for p in pts])
    idx = np.argsort(d)[:k]
    w = 1 / (d[idx] + 1e-3)
    out = out + (w[:, None] * resid[idx]).sum(0) / w.sum()
    if scalar:
        return float(out[0])
    out = out.tolist()
    for name in LOCKED:
        j = JOINTS.index(name)
        out[j] = float(np.median([p[key][j] for p in pts]))
    return out


def check(grid):
    # per joint, how far each taught spot is from the straight line fit
    _, resid, _ = _fit(grid, "grasp")
    names = [j for j in JOINTS[:5] if j not in LOCKED]
    cols = [JOINTS.index(j) for j in names]
    print("errors per joint (" + ", ".join(names) + ")")
    for i, (p, r) in enumerate(zip(grid["points"], resid)):
        errs = [abs(float(r[c])) for c in cols]
        flag = "  <-- suspicious" if max(errs) > 10 else ""
        print(f"point {i}: pixel {p['px']:.0f},{p['py']:.0f}  " + "  ".join(f"{e:5.1f}" for e in errs) + flag)
    print("wrist_roll is locked so it's not listed. one column always big = that joint was posed inconsistently")


# ---------- modes ----------

def teach(robot, cap):
    grid = {"home": None, "points": []}
    robot.bus.disable_torque()
    print("torque off, move the arm by hand")
    input("put the arm at HOME and press enter")
    grid["home"] = joints(robot)
    while True:
        ans = input(f"\n{len(grid['points'])} spots taught. put a pill somewhere, press enter (or q to finish) ")
        if ans.strip().lower() == "q":
            break
        frame = grab(cap)
        pos = find_pill(frame)
        if pos is None:
            print("camera can't see the pill, move it inside the ROI box")
            continue
        print(f"pill at pixel {pos[0]:.0f},{pos[1]:.0f} ({color_name(pos[2])})")
        input("hold the gripper OPEN about 5cm above the pill, press enter")
        hover = joints(robot)
        input("lower it so the open jaws are around the pill, press enter")
        grasp = joints(robot)
        input("now squeeze the gripper closed on the pill, press enter")
        closed = joints(robot)[-1]
        grid["points"].append({"px": pos[0], "py": pos[1], "hover": hover, "grasp": grasp, "closed": closed})
        print("saved")
    robot.bus.enable_torque()
    with open(GRID_FILE, "w") as f:
        json.dump(grid, f, indent=2)
    print(f"wrote {GRID_FILE} with {len(grid['points'])} points")


def test(cap):
    print("q to quit")
    while True:
        frame = grab(cap)
        pos = find_pill(frame)
        cv2.rectangle(frame, ROI[:2], ROI[2:], (255, 255, 0), 1)
        mk = find_marker(frame)
        if mk:
            cv2.drawMarker(frame, (int(mk[0]), int(mk[1])), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
        if pos:
            cv2.circle(frame, (int(pos[0]), int(pos[1])), 12, (0, 255, 0), 2)
            cv2.putText(frame, f"{color_name(pos[2])} {pos[0]:.0f},{pos[1]:.0f} h{pos[2]:.0f}", (int(pos[0]) + 15, int(pos[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("top", frame)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def wrist_servo(robot, wcap, pose):
    """at hover, look with the wrist cam and nudge pan / reach until the pill is centred.
    returns the corrected pose and the joint deltas so I can apply them to the grasp too."""
    delta = [0.0] * len(JOINTS)
    for i in range(WRIST_TRIES):
        pos = find_pill(grab(wcap), WRIST_ROI)
        if pos is None:
            print("  wrist cam doesn't see the pill, skipping correction")
            break
        ex = pos[0] - WRIST_TARGET[0]
        ey = pos[1] - WRIST_TARGET[1]
        print(f"  wrist view: pill at {pos[0]:.0f},{pos[1]:.0f}  err {ex:+.0f},{ey:+.0f}")
        if abs(ex) < WRIST_TOL and abs(ey) < WRIST_TOL:
            break
        d_pan = -K_PAN * ex
        d_reach = K_REACH * ey
        delta[0] += d_pan
        delta[2] += d_reach
        pose = list(pose)
        pose[0] += d_pan
        pose[2] += d_reach
        goto(robot, pose, 0.6)
        time.sleep(0.2)
    return pose, delta


def overhead_servo(robot, cap, pose, target, offset):
    """at hover, move pan and elbow until the jaw tape sits at target - offset in the top cam.
    measures the pixel-per-degree response first so I don't have to guess signs or gains."""
    def tape():
        for _ in range(3):
            m = find_marker(grab(cap))
            if m is not None:
                return np.array(m)
        return None

    pose = list(pose)
    goal = np.array(target) - np.array(offset)
    m0 = tape()
    if m0 is None:
        print("  can't see the jaw tape, skipping servo")
        return pose, [0.0] * len(JOINTS)
    # jog pan, see what moved, jog elbow, see what moved
    J = np.zeros((2, 2))
    for col, j in enumerate((0, 2)):
        p = list(pose); p[j] += JOG
        goto(robot, p, 0.5); time.sleep(0.15)
        m1 = tape()
        goto(robot, pose, 0.5); time.sleep(0.15)
        if m1 is None:
            print("  lost the tape while jogging, skipping servo")
            return pose, [0.0] * len(JOINTS)
        J[:, col] = (m1 - m0) / JOG
    if abs(np.linalg.det(J)) < 1e-3:
        print("  jog didn't move the tape enough, skipping servo")
        return pose, [0.0] * len(JOINTS)
    Jinv = np.linalg.inv(J)
    delta = [0.0] * len(JOINTS)
    for i in range(SERVO_TRIES):
        m = tape()
        if m is None:
            break
        err = goal - m
        print(f"  servo: tape at {m[0]:.0f},{m[1]:.0f}  err {err[0]:+.0f},{err[1]:+.0f}")
        if np.abs(err).max() < SERVO_TOL:
            break
        d = Jinv @ err
        d = np.clip(d, -4, 4) * 0.8  # don't lurch
        pose[0] += d[0]; pose[2] += d[1]
        delta[0] += d[0]; delta[2] += d[1]
        goto(robot, pose, 0.5); time.sleep(0.15)
    return pose, delta


def pick(robot, cap, slot, wcap=None):
    with open(GRID_FILE) as f:
        grid = json.load(f)
    frame = grab(cap)
    pos = find_pill(frame)
    if pos is None:
        print("no pill seen")
        return False
    px, py, hue = pos
    color = color_name(hue)
    if slot == "auto":
        slot = SLOT_BY_COLOR.get(color)
        if slot is None:
            print(f"don't know where {color} goes")
            return False
    print(f"{color} pill at {px:.0f},{py:.0f} -> slot {slot}")
    hover = blend(grid, px, py, "hover")
    grasp = blend(grid, px, py, "grasp")
    grasp = [g + o for g, o in zip(grasp, GRASP_OFFSET)]
    hover = [h + o for h, o in zip(hover, GRASP_OFFSET)]
    closed = blend(grid, px, py, "closed")
    # squeeze a bit further in whichever direction closes, open wider on the way down
    direction = 1 if closed > grasp[-1] else -1
    closed = closed + direction * GRIP_EXTRA
    hover[-1] = hover[-1] - direction * OPEN_EXTRA
    grasp[-1] = grasp[-1] - direction * OPEN_EXTRA
    print("grasp pose:", [round(v, 1) for v in grasp])

    goto(robot, grid["home"], 1.5)
    # always swing in from the same side so the pan gear slack gets taken up the same way every time
    pre = list(hover)
    pre[0] -= 5
    goto(robot, pre, 1.2)
    goto(robot, hover, 0.8)
    if wcap is not None and WRIST_TRIES > 0:
        hover, delta = wrist_servo(robot, wcap, hover)
        grasp = [g + d for g, d in zip(grasp, delta)]
    if SERVO and grid.get("marker_offset"):
        hover, delta = overhead_servo(robot, cap, hover, (px, py), grid["marker_offset"])
        grasp = [g + d for g, d in zip(grasp, delta)]
    for attempt in range(RETRIES):
        shift = RETRY_SHIFTS[attempt % len(RETRY_SHIFTS)]
        g = list(grasp)
        g[0] += shift
        h = list(hover)
        h[0] += shift
        if attempt:
            print(f"  missed, retry {attempt} with pan shift {shift:+.1f}")
            set_gripper(robot, hover[-1], 0.6)  # open back up
            goto(robot, h, 1.0)
        goto(robot, g, 1.2)
        if RAKE:
            g[0] += RAKE
            goto(robot, g, 0.6)
        set_gripper(robot, closed, 1.5)
        time.sleep(0.3)
        set_gripper(robot, closed - direction * 8, 0.6)  # ease off
        set_gripper(robot, closed, 1.0)  # bite again, usually recenters it
        time.sleep(0.3)
        hover_closed = list(h)
        hover_closed[-1] = closed
        goto(robot, hover_closed, 2.0)  # slow lift so it doesn't fling it
        # did I actually get it? pill still on the mat near where it was means no
        still = find_pill(grab(cap))
        if still is None or ((still[0] - px) ** 2 + (still[1] - py) ** 2) ** 0.5 > MISS_RADIUS:
            break
    else:
        print("  gave up after retries")
    home_closed = list(grid["home"])
    home_closed[-1] = closed
    goto(robot, home_closed, 1.5)
    replay(robot, f"datasets/place_slot{slot}")
    return True


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    cap = open_cam(CAM_TOP)
    if mode == "test":
        test(cap)
        sys.exit()
    if mode == "wrist":
        # shows the wrist cam with the target crosshair so I can set WRIST_TARGET and WRIST_ROI
        wcap = open_cam(CAM_WRIST, 1280, 720)
        while True:
            frame = grab(wcap)
            cv2.rectangle(frame, WRIST_ROI[:2], WRIST_ROI[2:], (255, 255, 0), 1)
            cv2.drawMarker(frame, WRIST_TARGET, (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
            pos = find_pill(frame, WRIST_ROI)
            if pos:
                cv2.circle(frame, (int(pos[0]), int(pos[1])), 12, (0, 255, 0), 2)
                cv2.putText(frame, f"{pos[0]:.0f},{pos[1]:.0f}", (int(pos[0]) + 15, int(pos[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("wrist", frame)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()
        wcap.release()
        sys.exit()
    if mode == "calib":
        # two steps because the gripper hides the pill from the top cam once it's over it
        with open(GRID_FILE) as f:
            grid = json.load(f)
        robot = Follower(FollowerConfig(port=PORT, id=ROBOT_ID))
        robot.connect()
        robot.bus.disable_torque()
        input("torque off. pill on the mat, arm OUT of the camera view, enter ")
        pl = find_pill(grab(cap))
        if pl is None:
            print("pill not seen, move it inside the box and rerun"); robot.bus.enable_torque(); robot.disconnect(); sys.exit()
        print(f"pill at {pl[0]:.0f},{pl[1]:.0f}")
        input("now hold the arm at hover height with the open jaw tips right above that pill, gripper vertical, enter ")
        m = find_marker(grab(cap))
        if m is None:
            print("tape not seen, check `test` shows the magenta cross"); robot.bus.enable_torque(); robot.disconnect(); sys.exit()
        grid["marker_offset"] = [pl[0] - m[0], pl[1] - m[1]]
        with open(GRID_FILE, "w") as f:
            json.dump(grid, f, indent=2)
        print(f"tape at {m[0]:.0f},{m[1]:.0f}, saved offset {[round(v) for v in grid['marker_offset']]}")
        robot.bus.enable_torque()
        robot.disconnect()
        sys.exit()
    if mode == "check":
        with open(GRID_FILE) as f:
            check(json.load(f))
        sys.exit()
    robot = Follower(FollowerConfig(port=PORT, id=ROBOT_ID))
    robot.connect()
    try:
        if mode == "teach":
            teach(robot, cap)
        elif mode in ("pick", "loop"):
            arg = sys.argv[2] if len(sys.argv) > 2 else "auto"
            wcap = open_cam(CAM_WRIST, 1280, 720)
            try:
                if mode == "pick":
                    pick(robot, cap, arg if arg == "auto" else int(arg), wcap)
                else:
                    # stays connected, one pick per enter. this is the demo mode
                    while True:
                        if input("\nenter to pick (q to quit) ").strip().lower() == "q":
                            break
                        pick(robot, cap, arg if arg == "auto" else int(arg), wcap)
            finally:
                wcap.release()
    finally:
        robot.disconnect()
        cap.release()