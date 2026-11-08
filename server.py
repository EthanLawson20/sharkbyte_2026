"""Minimal feed for pill_dashboard.html.
Run: pip install flask opencv-python  &&  python server.py
Endpoints: /video (MJPEG) and /state (JSON, detections in 0..1 coords).
"""
import time, threading, cv2
from flask import Flask, Response, jsonify

CAMERA_INDEX = 0
app = Flask(__name__)
lock = threading.Lock()
state = {
    "detections": [],   # [{"label": "round", "conf": 0.93, "x": .1, "y": .2, "w": .05, "h": .05}]
    "counts": {"round": 0, "capsule": 0, "oval": 0, "reject": 0},
    "mis_sorts": 0,
    "robot": {"state": "idle", "target": None, "picks_per_min": 0, "latency_ms": None},
    "events": [],       # strings; cleared after each /state read
}
latest = {"jpg": None}

def capture_loop():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05); continue
        t0 = time.time()
        # TODO: run your pill detector on `frame` and write results into `state`:
        #   h, w = frame.shape[:2]
        #   state["detections"] = [{"label": cls, "conf": c, "x": x1/w, "y": y1/h, "w": (x2-x1)/w, "h": (y2-y1)/h} ...]
        #   state["robot"]["latency_ms"] = round((time.time() - t0) * 1000)
        # TODO: from your LeRobot control loop, update state["robot"], state["counts"], state["events"].
        latest["jpg"] = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()

def mjpeg():
    while True:
        if latest["jpg"]:
            yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + latest["jpg"] + b"\r\n"
        time.sleep(1 / 30)

@app.after_request
def cors(r):
    r.headers["Access-Control-Allow-Origin"] = "*"
    return r

@app.route("/video")
def video():
    return Response(mjpeg(), mimetype="multipart/x-mixed-replace; boundary=f")

@app.route("/state")
def get_state():
    with lock:
        out = dict(state); state["events"] = []
    return jsonify(out)

if __name__ == "__main__":
    threading.Thread(target=capture_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, threaded=True)
