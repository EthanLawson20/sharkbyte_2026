"""camera check on the tray before each pill goes in. STUB=True makes it always pass until the rig exists."""

import cv2

STUB = True
CAM_INDEX = 0

# hsv ranges per pill, placeholder for noe, will tune on the real tray under the lamp
PILL_COLORS = {
    "white_round": ((0, 0, 180), (180, 40, 255)),
    "orange_oval": ((5, 120, 120), (20, 255, 255)),
}


def grab():
    cap = cv2.VideoCapture(CAM_INDEX)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("no frame from camera")
    return frame


def count_pills(frame, pill):
    lo, hi = PILL_COLORS[pill]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lo, hi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # ignore specks. a claritin should be a few hundred px from the top cam
    return sum(1 for c in contours if 150 < cv2.contourArea(c) < 2000)


def tray_has(pill, count):
    if STUB:
        return True
    frame = grab()
    n = count_pills(frame, pill)
    cv2.imwrite(f"last_check_{pill}.jpg", frame)  # for the dashboard
    print(f"[verify] saw {n} {pill}, wanted {count}")
    return n == count