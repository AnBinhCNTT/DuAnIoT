
import json
import threading
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path
 
import cv2
import numpy as np
import mediapipe as mp
import paho.mqtt.client as mqtt
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
 
BASE = Path(__file__).parent
MQTT_HOST = "broker.emqx.io"
PREFIX = "thang_smartlight_8421"          # PHẢI giống hệt bên ESP32
T_CMD = PREFIX + "/cmd"
T_DATA = PREFIX + "/data"
 
PINCH_LO = 0.15
PINCH_HI = 1.00
ONLINE_TIMEOUT = 15
 
HAND_LED = {"L": "Y", "R": "W"}
COLOR = {"L": (0, 215, 255), "R": (255, 255, 255)}
 
 
class State:
    def __init__(self):
        self.running = False
        self.thread = None
        self.frame = None
        self.mode = "AUTO"
        self.hand_seen = {"L": False, "R": False}
        self.hand_pct = {"L": 0.0, "R": 0.0}
        self.sent = {"Y": -1, "W": -1}
        self.sent_t = {"Y": 0.0, "W": 0.0}
        self.dev = {}
        self.dev_t = 0.0
        self.synced = False
        self.hist = deque(maxlen=240)
        self.events = []
        self.room_hand = None
        self.room_hand_t = 0.0
        self.sent_a = -1
        self.sent_a_t = 0.0
        self.released_a = True
 
    def reset_sent(self):
        self.sent = {"Y": -1, "W": -1}
 
    def log(self, level, text):
        self.events.insert(0, {"time": datetime.now().strftime("%H:%M:%S"),
                               "level": level, "text": text})
        del self.events[50:]
 
 
st = State()
 
 
def on_connect(c, userdata, flags, reason_code, properties=None):
    print("[MQTT] ket noi:", reason_code, flush=True)
    c.subscribe(T_DATA, qos=0)
    st.synced = False          # nối lại broker -> gửi lại chế độ khi có dữ liệu
 
 
def on_message(c, userdata, msg):
    try:
        d = json.loads(msg.payload.decode())
        now = time.time()
        if st.dev_t == 0 or now - st.dev_t > ONLINE_TIMEOUT:
            st.log("ok", "ESP32 đã kết nối")
            st.synced = False          # ESP32 vừa online lại -> gửi lại chế độ
        st.dev, st.dev_t = d, now
        st.hist.append({"light": d.get("light", 0), "y": d.get("y", 0), "w": d.get("w", 0)})
        if not st.synced:
            st.synced = True
            c.publish(T_CMD, "MODE:" + st.mode)
    except Exception as e:
        print("[MQTT] loi:", repr(e), msg.payload, flush=True)
 
 
def on_disconnect(c, userdata, *args):
    print("[MQTT] mat ket noi broker:", args, flush=True)
 
 
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                     client_id=f"smartlight-py-{int(time.time())}")
client.on_connect = on_connect
client.on_message = on_message
client.on_disconnect = on_disconnect
client.reconnect_delay_set(min_delay=1, max_delay=8)
client.connect(MQTT_HOST, 1883, 60)
client.loop_start()
 
 
def send_both():
    """Thủ công: gộp 2 đèn vào 1 lệnh, tối đa 20 lệnh/giây."""
    now = time.time()
    if now - st.sent_t["Y"] < 0.05:
        return
    if not (st.hand_seen["L"] or st.hand_seen["R"]):
        return
    y = int(round(st.hand_pct["L"])) if st.hand_seen["L"] else max(0, st.sent["Y"])
    w = int(round(st.hand_pct["R"])) if st.hand_seen["R"] else max(0, st.sent["W"])
    if (y, w) != (st.sent["Y"], st.sent["W"]):
        client.publish(T_CMD, f"S:{y},{w}", qos=0)
        st.sent["Y"], st.sent["W"] = y, w
        st.sent_t["Y"] = now
 
 
def send_ambient(pct):
    now = time.time()
    pct = int(max(0, min(100, pct)))
    if st.sent_a < 0 or abs(pct - st.sent_a) >= 1:
        if now - st.sent_a_t >= 0.05:
            client.publish(T_CMD, f"A:{pct}")
            st.sent_a, st.sent_a_t = pct, now
            st.released_a = False
 
 
def release_ambient():
    if not st.released_a:
        client.publish(T_CMD, "A:LDR")
        st.released_a = True
        st.sent_a = -1
        st.room_hand = None
 
 
def placeholder():
    img = np.zeros((480, 640, 3), np.uint8)
    cv2.putText(img, "Bam START de bat camera", (110, 245),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
    return cv2.imencode(".jpg", img)[1].tobytes()
 
 
PLACEHOLDER = placeholder()
 
 
def open_camera():
    """Thử nhiều camera / backend, trả về (cap, index) hoặc (None, None)."""
    for idx in (0, 1, 2):
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    return cap, idx
            cap.release()
    return None, None
 
 
def run_camera():
    mp_hands = mp.solutions.hands
    drawer = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6,
                           min_tracking_confidence=0.6)
 
    cap, idx = open_camera()
    if cap is None:
        st.log("bad", "Không mở được camera (đang bị app khác dùng?)")
        print("[CAM] khong mo duoc camera", flush=True)
        hands.close()
        return
    st.log("ok", f"Camera số {idx} đã mở")
    print(f"[CAM] da mo camera so {idx}", flush=True)
 
    fails = 0
    while st.running:
        ok, img = cap.read()
        if not ok:
            fails += 1
            if fails > 100:                    # ~5 giây không có hình
                st.log("bad", "Camera không trả hình, đã dừng")
                break
            time.sleep(0.05)
            continue
        fails = 0
 
        img = cv2.flip(img, 1)
        h, w = img.shape[:2]
        res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
 
        seen = {"L": False, "R": False}
        wrist = {"L": None, "R": None}
        if res.multi_hand_landmarks:
            for hl, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                side = "L" if hd.classification[0].label == "Left" else "R"
                if seen[side]:
                    continue
                seen[side] = True
                col = COLOR[side]
                drawer.draw_landmarks(img, hl, mp_hands.HAND_CONNECTIONS,
                                      drawer.DrawingSpec(color=col, thickness=2, circle_radius=3),
                                      drawer.DrawingSpec(color=col, thickness=2))
                P = lambda i, _hl=hl: np.array([_hl.landmark[i].x * w, _hl.landmark[i].y * h])
                ratio = np.linalg.norm(P(4) - P(8)) / max(1e-6, np.linalg.norm(P(0) - P(9)))
                raw_pct = float(np.clip((ratio - PINCH_LO) / (PINCH_HI - PINCH_LO), 0, 1) * 100)
                if st.hand_seen[side]:
                    st.hand_pct[side] = 0.2 * st.hand_pct[side] + 0.8 * raw_pct
                else:
                    st.hand_pct[side] = raw_pct
                a, b = P(4), P(8)
                cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), col, 3)
                wr = P(0)
                wrist[side] = wr
                if st.mode == "MANUAL":
                    cv2.putText(img, f"{HAND_LED[side]} {int(st.hand_pct[side])}%",
                                (int(wr[0]) - 20, int(wr[1]) + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        st.hand_seen = seen
 
        if st.mode == "MANUAL":
            send_both()
        else:
            side = "L" if seen["L"] else ("R" if seen["R"] else None)
            if side is not None:
                room = int(round(st.hand_pct[side]))
                st.room_hand = room
                st.room_hand_t = time.time()
                send_ambient(room)
                if wrist[side] is not None:
                    wr = wrist[side]
                    cv2.putText(img, f"PHONG {room}%  den {100 - room}%",
                                (int(wr[0]) - 40, int(wr[1]) + 36),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2)
            else:
                release_ambient()
 
        hud = "AUTO: 1 tay | mo ngon = phong sang | chum = phong toi" if st.mode == "AUTO" \
            else "THU CONG: tay trai = den vang, tay phai = den trang"
        cv2.putText(img, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        st.frame = cv2.imencode(".jpg", img)[1].tobytes()
    cap.release()
    hands.close()
 
 
def loop():
    """Chạy camera; nếu lỗi thì ghi log và cho phép bấm 'Bật camera' lại."""
    try:
        run_camera()
    except Exception as e:
        traceback.print_exc()
        st.log("bad", f"Lỗi camera: {e!r}")
    finally:
        st.running = False
        st.frame = None
 
 
def view_dev():
    dev = dict(st.dev)
    fresh = st.room_hand is not None and (time.time() - st.room_hand_t) < 1.0
    if st.mode == "AUTO" and fresh:
        light = int(st.room_hand)
        dev["light"] = light
        dev["y"] = 100 - light
        dev["w"] = 100 - light
        dev["mode"] = "AUTO"
    return dev
 
 
app = FastAPI()
 
 
@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")
 
 
@app.post("/api/start")
def api_start():
    if not st.running:
        st.running = True
        st.reset_sent()
        st.log("ok", "Bật camera")
        st.thread = threading.Thread(target=loop, daemon=True)
        st.thread.start()
    return {"ok": True}
 
 
@app.post("/api/stop")
def api_stop():
    if st.running:
        st.running = False
        if st.thread:
            st.thread.join(timeout=3)
        st.frame = None
        st.hand_seen = {"L": False, "R": False}
        release_ambient()
        st.log("warn", "Tắt camera")
    return {"ok": True}
 
 
class ModeBody(BaseModel):
    mode: str
 
 
@app.post("/api/mode")
def api_mode(b: ModeBody):
    m = "MANUAL" if b.mode.upper() == "MANUAL" else "AUTO"
    st.mode = m
    st.reset_sent()
    st.synced = True
    if m == "MANUAL":
        release_ambient()
    client.publish(T_CMD, "MODE:" + m)
    st.log("ok", "Tự động: một tay chỉnh ánh sáng phòng" if m == "AUTO"
           else "Thủ công: tay trái vàng, tay phải trắng")
    return {"ok": True}
 
 
@app.get("/api/status")
def api_status():
    return {
        "running": st.running, "mode": st.mode,
        "hand_seen": st.hand_seen,
        "hand_pct": {k: int(v) for k, v in st.hand_pct.items()},
        "hand_led": HAND_LED,
        "mqtt": client.is_connected(),
        "sent": {k: max(0, v) for k, v in st.sent.items()},
        "online": (time.time() - st.dev_t < ONLINE_TIMEOUT) if st.dev_t else False,
        "dev": view_dev(),
        "room_hand": st.room_hand,
        "hist": list(st.hist), "events": st.events,
    }
 
 
@app.get("/video")
def video():
    def gen():
        while True:
            f = st.frame if st.running and st.frame else PLACEHOLDER
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n"
            time.sleep(0.04)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")
 
 
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)