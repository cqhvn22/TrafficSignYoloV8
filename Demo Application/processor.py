"""
processor.py
Lõi xử lý: detect (YOLO) + track (ByteTrack) + OCR biển báo tốc độ (EasyOCR)
+ vẽ box/label. Tách riêng để dùng chung cho cả 3 mode: Camera / Ảnh / Video.
"""

from ultralytics import YOLO
import cv2
import easyocr
import colorsys
from collections import deque, Counter

# ==========================
# CONFIG (giữ nguyên từ bản gốc)
# ==========================
MODEL_PATH = "best.pt"

CONF_THRESH = 0.4
IOU_THRESH = 0.4
IMG_SIZE = 640

TRACKER_CFG = "bytetrack.yaml"

GHOST_FRAMES = 25
MIN_HITS = 2

BBOX_SMOOTH_ALPHA = 0.55
SIZE_SMOOTH_ALPHA = 0.3
CONF_SMOOTH_ALPHA = 0.4

VEL_SMOOTH_ALPHA = 0.6
VEL_DECAY = 0.85

OCR_VOTES_NEEDED = 3
OCR_VOTE_POOL = 6
OCR_EVERY_N_FRAMES = 3
SPEED_SIGN_CLASS = "P.127"
SPEED_MIN, SPEED_MAX = 5, 130


# ==========================
# TIỆN ÍCH BOX <-> (TÂM, KÍCH THƯỚC)
# ==========================
def box_to_center_size(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1


def center_size_to_box(cx, cy, w, h):
    return int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)


def clamp_box(box, width, height):
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, width - 2))
    y1 = max(0, min(y1, height - 2))
    x2 = max(x1 + 1, min(x2, width - 1))
    y2 = max(y1 + 1, min(y2, height - 1))
    return x1, y1, x2, y2


# ==========================
# VẼ
# ==========================
def get_class_color(cls_id):
    hue = (cls_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)  # BGR


def draw_dashed_rect(frame, pt1, pt2, color, thickness=1, dash=8, gap=6):
    x1, y1 = pt1
    x2, y2 = pt2
    for x in range(x1, x2, dash + gap):
        xe = min(x + dash, x2)
        cv2.line(frame, (x, y1), (xe, y1), color, thickness)
        cv2.line(frame, (x, y2), (xe, y2), color, thickness)
    for y in range(y1, y2, dash + gap):
        ye = min(y + dash, y2)
        cv2.line(frame, (x1, y), (x1, ye), color, thickness)
        cv2.line(frame, (x2, y), (x2, ye), color, thickness)


def draw_corner_box(frame, box, color, thickness=2, corner_ratio=0.18, dashed=False):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cl = max(int(min(w, h) * corner_ratio), 12)

    if dashed:
        draw_dashed_rect(frame, (x1, y1), (x2, y2), color, 1)
    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    t = thickness + 1
    cv2.line(frame, (x1, y1), (x1 + cl, y1), color, t, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x1, y1 + cl), color, t, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2 - cl, y1), color, t, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2, y1 + cl), color, t, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1 + cl, y2), color, t, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1, y2 - cl), color, t, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2 - cl, y2), color, t, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2, y2 - cl), color, t, cv2.LINE_AA)


def draw_label(frame, x1, y1, text, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, fth = 0.55, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, fth + 1)
    pad = 6
    ly1 = max(y1 - th - 2 * pad, 0)
    cv2.rectangle(frame, (x1, ly1), (x1 + tw + 2 * pad, ly1 + th + 2 * pad), color, -1, cv2.LINE_AA)
    cv2.putText(frame, text, (x1 + pad, ly1 + th + pad - 2), font, scale, (255, 255, 255), fth + 1, cv2.LINE_AA)


# ==========================
# OCR
# ==========================
def preprocess_for_ocr(crop):
    if crop is None or crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.equalizeHist(gray)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


class TrafficSignProcessor:
    """
    Bọc model YOLO + EasyOCR + state tracking.
    Dùng được cho 3 chế độ:
      - process_frame_tracked(frame): dùng cho camera/video (có track_id xuyên suốt)
      - process_image_single(frame): dùng cho 1 ảnh tĩnh (không cần track id, chỉ detect + OCR 1 lần)
    """

    def __init__(self, model_path=MODEL_PATH, use_gpu=True, progress_cb=None):
        if progress_cb:
            progress_cb("Đang tải model YOLO...")
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        self.class_colors = {i: get_class_color(i) for i in self.class_names}

        if progress_cb:
            progress_cb("Đang tải EasyOCR (lần đầu có thể chậm)...")
        try:
            self.reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
        except Exception:
            # fallback nếu không có GPU / lỗi CUDA
            self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)

        self.tracked = {}
        self.frame_idx = 0

    def reset_tracking(self):
        """Gọi khi bắt đầu 1 phiên camera/video mới để xóa track cũ."""
        self.tracked = {}
        self.frame_idx = 0

    # ---------- OCR ----------
    def read_speed(self, crop):
        proc = preprocess_for_ocr(crop)
        if proc is None:
            return ""
        try:
            result = self.reader.readtext(proc, allowlist='0123456789', detail=0)
            for r in result:
                digits = r.strip()
                if digits.isdigit() and 1 <= len(digits) <= 3 and SPEED_MIN <= int(digits) <= SPEED_MAX:
                    return digits
        except Exception:
            pass
        return ""

    # ---------- MODE: CAMERA / VIDEO (có tracking xuyên frame) ----------
    def process_frame_tracked(self, frame):
        """
        Xử lý 1 frame trong luồng video/camera, có track_id ổn định,
        ghim box khi mất detect, OCR khóa tốc độ dần theo thời gian.
        Trả về frame đã vẽ (frame gốc bị sửa trực tiếp).
        """
        self.frame_idx += 1
        frame_h, frame_w = frame.shape[:2]

        results = self.model.track(
            frame,
            persist=True,
            tracker=TRACKER_CFG,
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            imgsz=IMG_SIZE,
            verbose=False
        )

        result = results[0]
        seen_ids = set()

        if result.boxes is not None and result.boxes.id is not None:
            for box in result.boxes:
                track_id = int(box.id[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_name = self.class_names[cls_id]

                seen_ids.add(track_id)
                cx, cy, w, h = box_to_center_size((x1, y1, x2, y2))

                if track_id not in self.tracked:
                    self.tracked[track_id] = {
                        "center": (cx, cy),
                        "size": (w, h),
                        "velocity": (0.0, 0.0),
                        "cls_id": cls_id,
                        "cls_name": cls_name,
                        "conf": conf,
                        "hits": 1,
                        "missed": 0,
                        "speed_votes": deque(maxlen=OCR_VOTE_POOL),
                        "speed_locked": "",
                        "last_ocr_frame": -999,
                    }
                else:
                    t = self.tracked[track_id]
                    old_cx, old_cy = t["center"]

                    raw_vx, raw_vy = cx - old_cx, cy - old_cy
                    vx, vy = t["velocity"]
                    t["velocity"] = (
                        vx * (1 - VEL_SMOOTH_ALPHA) + raw_vx * VEL_SMOOTH_ALPHA,
                        vy * (1 - VEL_SMOOTH_ALPHA) + raw_vy * VEL_SMOOTH_ALPHA,
                    )

                    t["center"] = (
                        old_cx * (1 - BBOX_SMOOTH_ALPHA) + cx * BBOX_SMOOTH_ALPHA,
                        old_cy * (1 - BBOX_SMOOTH_ALPHA) + cy * BBOX_SMOOTH_ALPHA,
                    )
                    old_w, old_h = t["size"]
                    t["size"] = (
                        old_w * (1 - SIZE_SMOOTH_ALPHA) + w * SIZE_SMOOTH_ALPHA,
                        old_h * (1 - SIZE_SMOOTH_ALPHA) + h * SIZE_SMOOTH_ALPHA,
                    )
                    t["cls_id"] = cls_id
                    t["cls_name"] = cls_name
                    t["conf"] = t["conf"] * (1 - CONF_SMOOTH_ALPHA) + conf * CONF_SMOOTH_ALPHA
                    t["hits"] += 1
                    t["missed"] = 0

                t = self.tracked[track_id]

                if cls_name == SPEED_SIGN_CLASS and not t["speed_locked"]:
                    if self.frame_idx - t["last_ocr_frame"] >= OCR_EVERY_N_FRAMES:
                        t["last_ocr_frame"] = self.frame_idx
                        crop = frame[y1:y2, x1:x2]
                        speed = self.read_speed(crop)
                        if speed:
                            t["speed_votes"].append(speed)
                            most_common, count = Counter(t["speed_votes"]).most_common(1)[0]
                            if count >= OCR_VOTES_NEEDED:
                                t["speed_locked"] = most_common

        # Ngoại suy track bị mất
        for tid in list(self.tracked.keys()):
            if tid not in seen_ids:
                t = self.tracked[tid]
                t["missed"] += 1
                if t["missed"] > GHOST_FRAMES:
                    del self.tracked[tid]
                    continue
                cx, cy = t["center"]
                vx, vy = t["velocity"]
                t["center"] = (cx + vx, cy + vy)
                t["velocity"] = (vx * VEL_DECAY, vy * VEL_DECAY)

        # Vẽ
        for tid, t in self.tracked.items():
            if t["hits"] < MIN_HITS:
                continue

            cx, cy = t["center"]
            w, h = t["size"]
            box = center_size_to_box(cx, cy, w, h)
            box = clamp_box(box, frame_w, frame_h)
            x1, y1, x2, y2 = box

            color = self.class_colors.get(t["cls_id"], (0, 255, 0))
            is_ghost = t["missed"] > 0

            draw_corner_box(frame, (x1, y1, x2, y2), color, dashed=is_ghost)

            label = f"{t['cls_name']} {t['conf']:.2f} #{tid}"
            if t["cls_name"] == SPEED_SIGN_CLASS and t["speed_locked"]:
                label += f" | {t['speed_locked']} km/h"

            draw_label(frame, x1, y1, label, color)

        return frame

    # ---------- MODE: ẢNH TĨNH (không cần track_id, OCR ngay lập tức) ----------
    def process_image_single(self, frame):
        """
        Xử lý 1 ảnh độc lập (không có khái niệm 'frame trước'):
        detect trực tiếp (không track), với biển tốc độ thì OCR ngay luôn
        (không cần chờ vote nhiều lần vì ảnh tĩnh chỉ có 1 lượt detect).
        Trả về frame đã vẽ.
        """
        frame_h, frame_w = frame.shape[:2]

        results = self.model.predict(
            frame,
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            imgsz=IMG_SIZE,
            verbose=False
        )
        result = results[0]

        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_name = self.class_names[cls_id]

                box_clamped = clamp_box((x1, y1, x2, y2), frame_w, frame_h)
                x1, y1, x2, y2 = box_clamped

                color = self.class_colors.get(cls_id, (0, 255, 0))
                draw_corner_box(frame, (x1, y1, x2, y2), color, dashed=False)

                label = f"{cls_name} {conf:.2f}"

                if cls_name == SPEED_SIGN_CLASS:
                    # ảnh tĩnh: thử đọc OCR vài lần trên cùng 1 crop (vì không có gì
                    # để "vote" qua thời gian), lấy kết quả phổ biến nhất nếu đọc được
                    crop = frame[y1:y2, x1:x2]
                    speed = self.read_speed(crop)
                    if speed:
                        label += f" | {speed} km/h"

                draw_label(frame, x1, y1, label, color)

        return frame