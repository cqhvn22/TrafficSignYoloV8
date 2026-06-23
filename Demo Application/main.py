import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import os
import time

import cv2
from PIL import Image, ImageTk

from processor import TrafficSignProcessor, MODEL_PATH


APP_TITLE = "Nhận diện biển báo giao thông Việt Nam - YOLOv8 + EasyOCR"
DISPLAY_MAX_W = 960
DISPLAY_MAX_H = 600


class TrafficSignApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1080x760")
        self.root.minsize(820, 600)

        self.processor = None  # load lazy (chạy nền) để không treo UI lúc mở app
        self.processor_ready = False

        # Trạng thái phiên đang chạy (camera / video)
        self.cap = None
        self.running = False           # đang chạy camera hoặc video
        self.stop_requested = False
        self.worker_thread = None
        self.frame_queue = queue.Queue(maxsize=2)  # frame đã xử lý, để UI thread hiển thị

        # Video export (tùy chọn, dùng cho mode Video)
        self.video_writer = None
        self.export_path = None
        self.last_result_frame = None

        self._build_ui()
        self._load_model_async()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # UI LAYOUT
    # ------------------------------------------------------------------
    def _build_ui(self):
        # ---- Top: thanh chọn mode ----
        top = ttk.Frame(self.root, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Chế độ:", font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT, padx=(0, 10))

        self.btn_camera = ttk.Button(top, text="📷 Camera", command=self.start_camera_mode)
        self.btn_camera.pack(side=tk.LEFT, padx=4)

        self.btn_image = ttk.Button(top, text="🖼 Ảnh", command=self.start_image_mode)
        self.btn_image.pack(side=tk.LEFT, padx=4)

        self.btn_video = ttk.Button(top, text="🎬 Video", command=self.start_video_mode)
        self.btn_video.pack(side=tk.LEFT, padx=4)

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        self.btn_stop = ttk.Button(top, text="⏹ Dừng", command=self.stop_current, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=4)

        self.btn_save = ttk.Button(top, text="💾 Lưu ảnh kết quả", command=self.save_current_image, state=tk.DISABLED)
        self.btn_save.pack(side=tk.LEFT, padx=4)

        # ---- Center: khung hiển thị ảnh/video ----
        center = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        center.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.display_label = ttk.Label(center, background="#1e1e1e", anchor=tk.CENTER)
        self.display_label.pack(fill=tk.BOTH, expand=True)
        self._set_placeholder_text("Chọn một chế độ ở trên để bắt đầu.")

        # ---- Bottom: status bar + progress ----
        bottom = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        bottom.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_var = tk.StringVar(value="Đang khởi động...")
        self.status_label = ttk.Label(bottom, textvariable=self.status_var, anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.progress = ttk.Progressbar(bottom, mode="determinate", length=200)
        self.progress.pack(side=tk.RIGHT)

        self._set_controls_enabled(False)  # khóa nút cho tới khi model load xong

    def _set_placeholder_text(self, text):
        self.display_label.configure(text=text, image="", font=("Segoe UI", 13), foreground="#aaaaaa")

    def _set_controls_enabled(self, enabled):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.btn_camera.configure(state=state)
        self.btn_image.configure(state=state)
        self.btn_video.configure(state=state)

    # ------------------------------------------------------------------
    # LOAD MODEL (chạy nền để UI không bị treo)
    # ------------------------------------------------------------------
    def _load_model_async(self):
        def task():
            try:
                def progress_cb(msg):
                    self.root.after(0, lambda: self.status_var.set(msg))

                self.processor = TrafficSignProcessor(model_path=MODEL_PATH, progress_cb=progress_cb)
                self.processor_ready = True
                self.root.after(0, self._on_model_loaded)
            except Exception as e:
                self.root.after(0, lambda: self._on_model_load_failed(str(e)))

        threading.Thread(target=task, daemon=True).start()

    def _on_model_loaded(self):
        self.status_var.set("Sẵn sàng. Chọn chế độ Camera / Ảnh / Video.")
        self._set_controls_enabled(True)

    def _on_model_load_failed(self, err_msg):
        self.status_var.set("Lỗi tải model.")
        messagebox.showerror(
            "Lỗi tải model",
            f"Không thể tải model hoặc EasyOCR:\n{err_msg}\n\n"
            f"Kiểm tra file '{MODEL_PATH}' có nằm cùng thư mục với app_gui.py không."
        )

    # ------------------------------------------------------------------
    # HIỂN THỊ FRAME (OpenCV BGR ndarray -> Tkinter)
    # ------------------------------------------------------------------
    def _show_frame(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        scale = min(DISPLAY_MAX_W / w, DISPLAY_MAX_H / h, 1.0)
        if scale < 1.0:
            frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        self.display_label.imgtk = imgtk  # giữ reference, tránh bị garbage-collected
        self.display_label.configure(image=imgtk, text="")

    # ------------------------------------------------------------------
    # CHẾ ĐỘ CAMERA
    # ------------------------------------------------------------------
    def start_camera_mode(self):
        if not self._guard_ready():
            return
        if self.running:
            messagebox.showinfo("Đang chạy", "Vui lòng dừng chế độ hiện tại trước.")
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Lỗi camera", "Không thể mở camera. Kiểm tra camera có đang được dùng bởi app khác không.")
            self.cap = None
            return

        self.processor.reset_tracking()
        self.running = True
        self.stop_requested = False
        self.btn_stop.configure(state=tk.NORMAL)
        self.btn_save.configure(state=tk.NORMAL)
        self.status_var.set("Camera đang chạy. Nhấn Dừng để kết thúc.")

        self.worker_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.worker_thread.start()
        self._poll_frame_queue()

    def _camera_loop(self):
        prev_time = time.time()
        while self.running and not self.stop_requested:
            ret, frame = self.cap.read()
            if not ret:
                break
            try:
                frame = self.processor.process_frame_tracked(frame)
            except Exception as e:
                self.root.after(0, lambda err=e: self.status_var.set(f"Lỗi xử lý frame: {err}"))
                continue

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            self._push_frame(frame)

        self._cleanup_capture()
        self.root.after(0, self._on_stream_ended)

    # ------------------------------------------------------------------
    # CHẾ ĐỘ ẢNH
    # ------------------------------------------------------------------
    def start_image_mode(self):
        if not self._guard_ready():
            return
        if self.running:
            messagebox.showinfo("Đang chạy", "Vui lòng dừng chế độ hiện tại trước.")
            return

        path = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Tất cả file", "*.*")]
        )
        if not path:
            return

        self.status_var.set(f"Đang xử lý ảnh: {os.path.basename(path)} ...")
        self.btn_save.configure(state=tk.DISABLED)

        def task():
            frame = cv2.imread(path)
            if frame is None:
                self.root.after(0, lambda: messagebox.showerror("Lỗi", "Không đọc được file ảnh."))
                self.root.after(0, lambda: self.status_var.set("Sẵn sàng."))
                return
            try:
                result_frame = self.processor.process_image_single(frame)
            except Exception as e:
                self.root.after(0, lambda err=e: messagebox.showerror("Lỗi xử lý", str(err)))
                self.root.after(0, lambda: self.status_var.set("Sẵn sàng."))
                return

            self.last_result_frame = result_frame
            self.root.after(0, lambda: self._show_frame(result_frame))
            self.root.after(0, lambda: self.status_var.set(f"Hoàn tất: {os.path.basename(path)}"))
            self.root.after(0, lambda: self.btn_save.configure(state=tk.NORMAL))

        threading.Thread(target=task, daemon=True).start()

    # ------------------------------------------------------------------
    # CHẾ ĐỘ VIDEO
    # ------------------------------------------------------------------
    def start_video_mode(self):
        if not self._guard_ready():
            return
        if self.running:
            messagebox.showinfo("Đang chạy", "Vui lòng dừng chế độ hiện tại trước.")
            return

        path = filedialog.askopenfilename(
            title="Chọn video",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.webm"), ("Tất cả file", "*.*")]
        )
        if not path:
            return

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("Lỗi video", "Không thể mở file video này.")
            self.cap = None
            return

        # Hỏi người dùng có muốn xuất video kết quả không
        export = messagebox.askyesno("Xuất video", "Bạn có muốn lưu video kết quả ra file không?")
        self.export_path = None
        self.video_writer = None
        if export:
            save_path = filedialog.asksaveasfilename(
                title="Lưu video kết quả",
                defaultextension=".mp4",
                filetypes=[("MP4", "*.mp4")]
            )
            if save_path:
                self.export_path = save_path

        self.processor.reset_tracking()
        self.running = True
        self.stop_requested = False
        self.btn_stop.configure(state=tk.NORMAL)
        self.btn_save.configure(state=tk.NORMAL)
        self.status_var.set(f"Đang xử lý video: {os.path.basename(path)} ...")

        self.worker_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.worker_thread.start()
        self._poll_frame_queue()

    def _video_loop(self):
        fps_in = self.cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        frame_count = 0

        while self.running and not self.stop_requested:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_count += 1

            try:
                frame = self.processor.process_frame_tracked(frame)
            except Exception as e:
                self.root.after(0, lambda err=e: self.status_var.set(f"Lỗi xử lý frame: {err}"))
                continue

            if self.export_path:
                if self.video_writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    self.video_writer = cv2.VideoWriter(self.export_path, fourcc, fps_in, (w, h))
                self.video_writer.write(frame)

            if total_frames > 0:
                pct = frame_count / total_frames
                self.root.after(0, lambda p=pct: self._update_progress(p))

            self._push_frame(frame)

        self._cleanup_capture()
        self.root.after(0, self._on_stream_ended)

    def _update_progress(self, fraction):
        self.progress["value"] = max(0, min(100, fraction * 100))

    # ------------------------------------------------------------------
    # CƠ CHẾ CHUYỂN FRAME TỪ WORKER THREAD -> UI THREAD
    # ------------------------------------------------------------------
    def _push_frame(self, frame):
        # Giữ hàng đợi nhỏ: nếu UI chưa kịp vẽ frame cũ, bỏ frame cũ đi (tránh trễ/lag dồn)
        try:
            if self.frame_queue.full():
                self.frame_queue.get_nowait()
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass

    def _poll_frame_queue(self):
        try:
            while True:
                frame = self.frame_queue.get_nowait()
                self.last_result_frame = frame
                self._show_frame(frame)
        except queue.Empty:
            pass

        if self.running:
            self.root.after(20, self._poll_frame_queue)

    # ------------------------------------------------------------------
    # DỪNG / DỌN DẸP
    # ------------------------------------------------------------------
    def stop_current(self):
        if not self.running:
            return
        self.stop_requested = True
        self.status_var.set("Đang dừng...")
        self.btn_stop.configure(state=tk.DISABLED)

    def _cleanup_capture(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def _on_stream_ended(self):
        self.btn_stop.configure(state=tk.DISABLED)
        self.progress["value"] = 0
        msg = "Đã dừng."
        if self.export_path:
            msg += f" Video kết quả đã lưu tại: {self.export_path}"
        self.status_var.set(msg)
        self.export_path = None

    # ------------------------------------------------------------------
    # LƯU ẢNH KẾT QUẢ HIỆN TẠI
    # ------------------------------------------------------------------
    def save_current_image(self):
        if self.last_result_frame is None:
            messagebox.showinfo("Chưa có ảnh", "Chưa có kết quả nào để lưu.")
            return
        path = filedialog.asksaveasfilename(
            title="Lưu ảnh kết quả",
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")]
        )
        if not path:
            return
        cv2.imwrite(path, self.last_result_frame)
        messagebox.showinfo("Đã lưu", f"Đã lưu ảnh tại:\n{path}")

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------
    def _guard_ready(self):
        if not self.processor_ready:
            messagebox.showinfo("Đang tải", "Model đang được tải, vui lòng chờ trong giây lát.")
            return False
        return True

    def _on_close(self):
        self.stop_requested = True
        self.running = False
        if self.cap is not None:
            self.cap.release()
        if self.video_writer is not None:
            self.video_writer.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
        elif "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass

    app = TrafficSignApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()