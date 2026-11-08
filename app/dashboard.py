"""live dashboard for the demo. streams the overhead cam with the pill detection drawn on,
the wrist cam raw, and a status panel. python app\\dashboard.py then open http://localhost:8000
in a browser. uses the cameras, so stop it before running pick/loop (replay doesn't need cams)."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

# same numbers as find_and_pick.py, kept here so this file has no lerobot import
CAM_TOP = 2
CAM_WRIST = 1
ROI = (0, 0, 340, 480)
SAT_MIN = 110
VAL_MIN = 90
MIN_AREA = 150
SLOT_BY_COLOR = {"red": 1, "blue": 2, "green": 3}
PORT = 8000


def circular_hue(hues):
    # red straddles 0/180 so a plain average lands in the middle and reads as blue. average on a circle
    a = hues.astype(float) * (2 * np.pi / 180)
    return (np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) / 2) % 180


def color_name(hue):
    refs = {"red": 0, "green": 60, "blue": 115}
    d = lambda h, r: min(abs(h - r), 180 - abs(h - r))
    return min(refs, key=lambda c: d(hue, refs[c]))


def find_pill(frame):
    x0, y0, x1, y1 = ROI
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


class Cam(threading.Thread):
    def __init__(self, index, w, h, annotate=False):
        super().__init__(daemon=True)
        self.index, self.w, self.h = index, w, h
        self.annotate = annotate
        self.status = {"seen": False}
        # start with a "no signal" frame so the page never hangs waiting on a dead camera
        blank = np.full((h, w, 3), 30, np.uint8)
        cv2.putText(blank, "connecting...", (w // 2 - 110, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120, 120, 120), 2)
        self.jpg = cv2.imencode(".jpg", blank)[1].tobytes()
        self.start()

    def run(self):
        try:
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)  # quiet the msmf spam
        except Exception:
            pass
        # open the camera here, in the thread, so the web server is up before windows finishes waking the cams
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_MSMF)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        while True:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.2)
                continue
            if self.annotate:
                pos = find_pill(frame)
                cv2.rectangle(frame, ROI[:2], ROI[2:], (255, 220, 0), 1)
                if pos:
                    x, y, hue = pos
                    name = color_name(hue)
                    cv2.circle(frame, (int(x), int(y)), 14, (0, 255, 0), 2)
                    cv2.putText(frame, f"{name} -> slot {SLOT_BY_COLOR[name]}", (int(x) + 18, int(y) + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    self.status = {"seen": True, "color": name, "slot": SLOT_BY_COLOR[name],
                                   "x": int(x), "y": int(y), "hue": int(hue)}
                else:
                    self.status = {"seen": False}
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                self.jpg = buf.tobytes()
            time.sleep(0.03)


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pillar</title>
<style>
:root{--bg:#000;--surface:#1C1C1E;--surface2:#2C2C2E;--text:#F5F5F7;--secondary:#86868B;--tertiary:#5E5E63;--blue:#0A84FF;--green:#30D158}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--text);font-family:-apple-system,"SF Pro Display","SF Pro Text","Segoe UI Variable Display","Segoe UI",system-ui,sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:1280px;margin:0 auto;padding:40px 48px 56px}
nav{display:flex;align-items:center;justify-content:space-between;height:28px}
nav .mark{font-size:17px;font-weight:600;letter-spacing:-.01em}
nav .live{font-size:13px;color:var(--secondary)}
nav .live::before{content:"";display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--green);margin-right:7px;vertical-align:1px}
h1{margin:64px 0 0;text-align:center;font-size:clamp(34px,4.2vw,56px);line-height:1.08;font-weight:600;letter-spacing:-.025em;min-height:1.1em}
h1 .dim{color:var(--secondary);font-weight:500}
.sub{margin:14px 0 0;text-align:center;font-size:19px;line-height:1.4;color:var(--secondary);font-weight:400;min-height:1.4em}
.grid{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(300px,1fr);gap:20px;margin-top:52px}
.panel{background:var(--surface);border-radius:22px;overflow:hidden}
.panel img{display:block;width:100%;height:100%;object-fit:cover}
.feed{aspect-ratio:4/3}
.col{display:flex;flex-direction:column;gap:20px}
.case{padding:26px 26px 22px}
.case p{margin:0 0 18px;font-size:15px;color:var(--secondary);font-weight:500}
.slots{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.slot{aspect-ratio:1/1.25;border-radius:16px;background:var(--surface2);position:relative;display:flex;align-items:flex-end;justify-content:center;padding-bottom:12px;font-size:15px;font-weight:600;color:var(--tertiary)}
.slot .pill{position:absolute;top:38%;left:50%;width:38%;aspect-ratio:1;border-radius:50%;transform:translate(-50%,-50%) scale(0);transition:transform .28s cubic-bezier(.2,.8,.2,1)}
.slot.on .pill{transform:translate(-50%,-50%) scale(1)}
.slot.on{color:var(--text)}
.wrist{aspect-ratio:16/10}
.foot{margin-top:28px;display:flex;justify-content:center;gap:28px;font-size:13px;color:var(--tertiary)}
@media (max-width:880px){.grid{grid-template-columns:1fr}.wrap{padding:28px 20px 40px}h1{margin-top:40px}}
@media (prefers-reduced-motion:reduce){.slot .pill{transition:none}}
</style></head><body>
<div class="wrap">
<nav><span class="mark">Pillar</span><span class="live">Live</span></nav>
<h1 id="say"><span class="dim">Waiting for a tablet.</span></h1>
<p class="sub" id="sub">Place one anywhere in the pickup zone.</p>
<div class="grid">
  <div class="panel feed"><img src="/top.mjpg" alt="Overhead camera"></div>
  <div class="col">
    <div class="panel case"><p>Destination</p>
      <div class="slots">
        <div class="slot" data-slot="1"><span class="pill"></span>1</div>
        <div class="slot" data-slot="2"><span class="pill"></span>2</div>
        <div class="slot" data-slot="3"><span class="pill"></span>3</div>
      </div>
    </div>
    <div class="panel wrist"><img src="/wrist.mjpg" alt="Wrist camera"></div>
  </div>
</div>
<div class="foot"><span>Overhead camera</span><span>Wrist camera</span><span id="px"></span></div>
</div>
<script>
const fill={red:"#FF453A",blue:"#0A84FF",green:"#30D158"};
const words={red:"Red",blue:"Blue",green:"Green"};
async function tick(){
  try{
    const s=await (await fetch("/status")).json();
    const say=document.getElementById("say"),sub=document.getElementById("sub"),px=document.getElementById("px");
    document.querySelectorAll(".slot").forEach(el=>el.classList.remove("on"));
    if(!s.seen){
      say.innerHTML='<span class="dim">Waiting for a tablet.</span>';
      sub.textContent="Place one anywhere in the pickup zone.";
      px.textContent="";
    }else{
      say.innerHTML=`${words[s.color]} tablet found. <span class="dim">Slot ${s.slot}.</span>`;
      sub.textContent="The arm will pick it up and place it in the matching slot.";
      const el=document.querySelector(`.slot[data-slot="${s.slot}"]`);
      el.querySelector(".pill").style.background=fill[s.color];el.classList.add("on");
      px.textContent=`Position ${s.x}, ${s.y}`;
    }
  }catch(e){}
  setTimeout(tick,300);
}
tick();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/":
            body = PAGE.encode()
            self.send_response(200); self.send_header("Content-Type", "text/html"); self.send_header("Content-Length", len(body)); self.end_headers()
            self.wfile.write(body)
        elif self.path == "/status":
            body = json.dumps(top.status).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", len(body)); self.end_headers()
            self.wfile.write(body)
        elif self.path in ("/top.mjpg", "/wrist.mjpg"):
            cam = top if self.path == "/top.mjpg" else wrist
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    jpg = cam.jpg
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    self.wfile.flush()
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
        else:
            self.send_response(404); self.end_headers()


if __name__ == "__main__":
    top = Cam(CAM_TOP, 640, 480, annotate=True)
    wrist = Cam(CAM_WRIST, 1280, 720)
    print(f"dashboard at http://localhost:{PORT}  (ctrl+c to stop)")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()