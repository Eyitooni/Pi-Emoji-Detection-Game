"""
expression_engine.py
Camera capture, face detection and facial-expression analysis for
Raspberry Pi 5 + Camera Module 3.

How it works
  * Picamera2 grabs frames from the Camera Module 3 (continuous autofocus).
    On a normal PC it falls back to a USB webcam through OpenCV.
  * MediaPipe Face Landmarker finds faces and returns 52 "blendshape" scores
    (how much the smile, brow raise, jaw drop, etc. are active).
  * A small rule-based classifier turns those scores into six expressions:
    happy, surprised, sad, angry, disgust, neutral.
  * Results are smoothed over time so labels do not flicker.

Both web_app.py and gui_app.py use this one engine.
"""

import os
import threading
import time
import urllib.request
from datetime import datetime

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

EXPRESSIONS = ["happy", "surprised", "sad", "angry", "disgust", "neutral"]

COLORS_HEX = {
    "happy": "#E3A008",
    "surprised": "#7A5AF8",
    "sad": "#2F7FC1",
    "angry": "#D64545",
    "disgust": "#3E9B64",
    "neutral": "#8693A1",
}


def hex_to_bgr(h):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


COLORS_BGR = {k: hex_to_bgr(v) for k, v in COLORS_HEX.items()}


def ensure_model():
    """Download the MediaPipe face model once (about 4 MB)."""
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Downloading face model (one time only)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


# --------------------------------------------------------------------------
# Expression scoring
# --------------------------------------------------------------------------
def _avg(bs, *names):
    return sum(bs.get(n, 0.0) for n in names) / len(names)


def _clip(v):
    return max(0.0, min(1.0, v))


def score_expressions(bs):
    """Map MediaPipe blendshape scores (0..1) to six expression scores (0..1)."""
    smile = _avg(bs, "mouthSmileLeft", "mouthSmileRight")
    cheek = _avg(bs, "cheekSquintLeft", "cheekSquintRight")
    frown = _avg(bs, "mouthFrownLeft", "mouthFrownRight")
    brow_down = _avg(bs, "browDownLeft", "browDownRight")
    brow_inner = bs.get("browInnerUp", 0.0)
    brow_outer = _avg(bs, "browOuterUpLeft", "browOuterUpRight")
    jaw = bs.get("jawOpen", 0.0)
    eye_wide = _avg(bs, "eyeWideLeft", "eyeWideRight")
    squint = _avg(bs, "eyeSquintLeft", "eyeSquintRight")
    sneer = _avg(bs, "noseSneerLeft", "noseSneerRight")
    up_lip = _avg(bs, "mouthUpperUpLeft", "mouthUpperUpRight")
    press = _avg(bs, "mouthPressLeft", "mouthPressRight")
    shrug = bs.get("mouthShrugLower", 0.0)

    happy = _clip(smile * 1.7 + cheek * 0.3)
    surprised = _clip(0.45 * jaw + 0.45 * eye_wide + 0.5 * max(brow_outer, brow_inner) - 0.5 * smile)
    sad = _clip(frown * 1.3 + brow_inner * 0.45 + shrug * 0.3 - smile)
    angry = _clip(brow_down * 1.2 + squint * 0.35 + press * 0.4 - smile * 0.5)
    disgust = _clip(sneer * 1.4 + up_lip * 0.6 - smile * 0.3)

    scores = {
        "happy": happy,
        "surprised": surprised,
        "sad": sad,
        "angry": angry,
        "disgust": disgust,
    }
    # Neutral wins when no expression is clearly active.
    scores["neutral"] = _clip(1.0 - max(scores.values()) * 2.2)
    return scores


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------
class ExpressionEngine:
    def __init__(self, size=(960, 540), mirror=True, max_faces=4,
                 snapshot_dir=None, smoothing=0.35):
        self.size = size
        self.mirror = mirror
        self.max_faces = max_faces
        self.smoothing = smoothing
        self.snapshot_dir = snapshot_dir or os.path.join(BASE_DIR, "snapshots")

        self._cond = threading.Condition()
        self._seq = 0
        self._jpeg = None
        self._frame = None
        self._state = self._empty_state()
        self._smooth = {}
        self._session = {e: 0.0 for e in EXPRESSIONS}

        self.running = False
        self._thread = None
        self._landmarker = None
        self._read = None
        self._close_cam = None

    # ---- lifecycle -------------------------------------------------------
    def start(self):
        if self.running:
            return
        self._open_camera()
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=ensure_model()),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=self.max_faces,
            output_face_blendshapes=True,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self._close_cam:
            try:
                self._close_cam()
            except Exception:
                pass
        if self._landmarker:
            self._landmarker.close()

    # ---- camera ----------------------------------------------------------
    def _open_camera(self):
        try:
            from picamera2 import Picamera2
            from libcamera import controls

            cam = Picamera2()
            cfg = cam.create_video_configuration(
                main={"size": self.size, "format": "RGB888"}
            )
            cam.configure(cfg)
            cam.start()
            try:
                # Camera Module 3 has autofocus: keep the face sharp.
                cam.set_controls({"AfMode": controls.AfModeEnum.Continuous})
            except Exception:
                pass

            # Note: Picamera2's "RGB888" is stored as B,G,R, which is what OpenCV wants.
            self._read = cam.capture_array

            def close():
                cam.stop()
                cam.close()

            self._close_cam = close
            print("Camera: Raspberry Pi Camera (Picamera2)")
            return
        except ImportError:
            print("Picamera2 not found, falling back to a USB webcam.")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError(
                "No camera found. Check the ribbon cable and run: rpicam-hello"
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])

        def read():
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Camera read failed")
            return cv2.resize(frame, self.size)

        self._read = read
        self._close_cam = cap.release
        print("Camera: USB webcam (OpenCV)")

    # ---- main loop -------------------------------------------------------
    def _loop(self):
        t_start = time.monotonic()
        last_ts = -1
        last_t = t_start
        fps = 0.0

        while self.running:
            try:
                frame = self._read()
            except Exception as exc:  # camera unplugged etc.
                self._publish(None, None, dict(self._empty_state(), error=str(exc)))
                time.sleep(0.5)
                continue

            if self.mirror:
                frame = cv2.flip(frame, 1)
            frame = frame[:, :, :3].copy()

            # Detect on a smaller copy: faster on the Pi, landmarks are normalised.
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (640, int(h * 640 / w)))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            now = time.monotonic()
            ts = max(int((now - t_start) * 1000), last_ts + 1)
            last_ts = ts
            result = self._landmarker.detect_for_video(mp_img, ts)

            dt = now - last_t
            last_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            faces = self._analyse(result, w, h)
            primary = self._primary_index(faces)
            if primary is not None:
                self._session[faces[primary]["expression"]] += dt

            self._annotate(frame, faces, primary)
            state = {
                "error": None,
                "fps": round(fps, 1),
                "face_count": len(faces),
                "primary": primary,
                "faces": faces,
                "session": self._session_percent(),
                "time": time.time(),
            }
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            self._publish(frame, buf.tobytes() if ok else None, state)

    # ---- analysis --------------------------------------------------------
    def _analyse(self, result, W, H):
        raw_faces = []
        for i, lms in enumerate(result.face_landmarks or []):
            xs = [p.x for p in lms]
            ys = [p.y for p in lms]
            bs = {c.category_name: c.score for c in result.face_blendshapes[i]}
            raw_faces.append({
                "box": (min(xs), min(ys), max(xs), max(ys)),
                "nose_x": lms[1].x,
                "scores": score_expressions(bs),
                "blink": _avg(bs, "eyeBlinkLeft", "eyeBlinkRight"),
                "jaw": bs.get("jawOpen", 0.0),
            })

        # Left-to-right order keeps each person's smoothing slot fairly stable.
        raw_faces.sort(key=lambda f: f["box"][0])
        self._smooth = {k: v for k, v in self._smooth.items() if k < len(raw_faces)}

        faces = []
        a = self.smoothing
        for i, f in enumerate(raw_faces):
            prev = self._smooth.get(i, f["scores"])
            sm = {e: prev[e] + a * (f["scores"][e] - prev[e]) for e in EXPRESSIONS}
            self._smooth[i] = sm

            total = sum(sm.values()) or 1.0
            pct = {e: round(sm[e] / total, 3) for e in EXPRESSIONS}
            dominant = max(sm, key=sm.get)

            x0, y0, x1, y1 = f["box"]
            ratio = (f["nose_x"] - x0) / max(x1 - x0, 1e-6)
            if ratio < 0.40:
                gaze = "Looking left"
            elif ratio > 0.60:
                gaze = "Looking right"
            else:
                gaze = "Facing camera"

            pad_x, pad_y = (x1 - x0) * 0.06, (y1 - y0) * 0.10
            bx0, by0 = max(0.0, x0 - pad_x), max(0.0, y0 - pad_y)
            bx1, by1 = min(1.0, x1 + pad_x), min(1.0, y1 + pad_y)

            faces.append({
                "id": i,
                "expression": dominant,
                "confidence": pct[dominant],
                "scores": pct,
                "eyes": "Closed" if f["blink"] > 0.5 else "Open",
                "mouth": "Open" if f["jaw"] > 0.25 else "Closed",
                "gaze": gaze,
                "box": [int(bx0 * W), int(by0 * H),
                        int((bx1 - bx0) * W), int((by1 - by0) * H)],
            })
        return faces

    @staticmethod
    def _primary_index(faces):
        if not faces:
            return None
        return max(range(len(faces)), key=lambda i: faces[i]["box"][2] * faces[i]["box"][3])

    def _session_percent(self):
        total = sum(self._session.values())
        if total <= 0:
            return {e: 0.0 for e in EXPRESSIONS}
        return {e: round(v / total, 3) for e, v in self._session.items()}

    # ---- drawing ---------------------------------------------------------
    @staticmethod
    def _annotate(frame, faces, primary):
        for i, f in enumerate(faces):
            x, y, w, h = f["box"]
            color = COLORS_BGR[f["expression"]]
            thick = 3 if i == primary else 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thick)

            label = "%s %d%%" % (f["expression"].capitalize(), round(f["confidence"] * 100))
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), base = cv2.getTextSize(label, font, 0.7, 2)
            ly = y - 8 if y - th - base - 8 > 0 else y + h + th + 8
            cv2.rectangle(frame, (x, ly - th - 6), (x + tw + 12, ly + base), color, -1)
            cv2.putText(frame, label, (x + 6, ly - 2), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # ---- shared output ---------------------------------------------------
    @staticmethod
    def _empty_state():
        return {
            "error": None, "fps": 0.0, "face_count": 0, "primary": None,
            "faces": [], "session": {e: 0.0 for e in EXPRESSIONS}, "time": time.time(),
        }

    def _publish(self, frame, jpeg, state):
        with self._cond:
            if frame is not None:
                self._frame = frame
                self._jpeg = jpeg
            self._state = state
            self._seq += 1
            self._cond.notify_all()

    def get_state(self):
        with self._cond:
            return self._state

    def get_frame(self):
        """Latest annotated BGR frame (numpy array) or None."""
        with self._cond:
            return None if self._frame is None else self._frame.copy()

    def wait_jpeg(self, last_seq, timeout=2.0):
        """Block until a newer frame exists. Returns (seq, jpeg_bytes)."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq != last_seq, timeout)
            return self._seq, self._jpeg

    def reset_session(self):
        self._session = {e: 0.0 for e in EXPRESSIONS}

    def save_snapshot(self):
        frame = self.get_frame()
        if frame is None:
            return None
        os.makedirs(self.snapshot_dir, exist_ok=True)
        path = os.path.join(
            self.snapshot_dir, datetime.now().strftime("face_%Y%m%d_%H%M%S.jpg")
        )
        cv2.imwrite(path, frame)
        return path
