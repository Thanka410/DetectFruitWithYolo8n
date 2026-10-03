import argparse
import base64
import os
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import serial
import tkinter as tk
from tkinter import ttk
from ultralytics import YOLO


VALID_CLASSES = {"apple", "banana", "orange"}


@dataclass
class Detection:
    label: str
    confidence: float
    xyxy: tuple[int, int, int, int]


@dataclass
class PendingFruit:
    label: str
    confidence: float
    image_path: str
    detected_at: str


def best_detection(result, names: dict[int, str], conf_threshold: float) -> Detection | None:
    best: Detection | None = None
    if result.boxes is None:
        return None

    for box in result.boxes:
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        label = names.get(class_id, str(class_id)).strip().lower()

        if label not in VALID_CLASSES or confidence < conf_threshold:
            continue

        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        detection = Detection(label=label, confidence=confidence, xyxy=(x1, y1, x2, y2))
        if best is None or detection.confidence > best.confidence:
            best = detection

    return best


def draw_detection(frame, detection: Detection) -> None:
    x1, y1, x2, y2 = detection.xyxy
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 0), 2)
    cv2.putText(
        frame,
        f"{detection.label} {detection.confidence:.2f}",
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 180, 0),
        2,
        cv2.LINE_AA,
    )


def frame_to_photo(frame, max_width: int, max_height: int) -> tk.PhotoImage:
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    resized = cv2.resize(frame, new_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    ok, encoded = cv2.imencode(".png", rgb)
    if not ok:
        raise RuntimeError("Cannot encode camera frame")
    data = base64.b64encode(encoded.tobytes())
    return tk.PhotoImage(data=data)


class FruitSorterApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = tk.Tk()
        self.root.title("Fruit Sorter YOLOv8 + ESP32")
        self.root.geometry("1180x720")

        self.capture_dir = Path(args.capture_dir)
        self.capture_dir.mkdir(parents=True, exist_ok=True)

        self.model = YOLO(args.model)
        self.serial_lock = threading.Lock()
        self.esp32 = serial.Serial(args.port, args.baud, timeout=0.05)
        time.sleep(2.0)
        self.esp32.reset_input_buffer()

        self.camera = cv2.VideoCapture(args.camera)
        if not self.camera.isOpened():
            raise RuntimeError(f"Khong mo duoc camera index {args.camera}")

        self.running = True
        self.detection_enabled = True
        self.relay_on = False
        self.last_sent_speed = args.motor_speed
        self.last_detection_time = 0.0
        self.last_detection_label: str | None = None

        self.counts = {"apple": 0, "banana": 0, "orange": 0}
        self.pending: list[PendingFruit] = []

        self.frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self.event_queue: queue.Queue = queue.Queue()
        self.serial_queue: queue.Queue[str] = queue.Queue()

        self._build_ui()

        self.serial_thread = threading.Thread(target=self._serial_reader_loop, daemon=True)
        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.serial_thread.start()
        self.camera_thread.start()

        self.send_command(f"MOTOR:SPEED:{self.last_sent_speed}")
        self.root.after(50, self._poll_events)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        camera_panel = ttk.Frame(self.root, padding=10)
        camera_panel.grid(row=0, column=0, sticky="nsew")
        camera_panel.rowconfigure(0, weight=1)
        camera_panel.columnconfigure(0, weight=1)

        self.camera_label = ttk.Label(camera_panel, text="Camera loading...", anchor="center")
        self.camera_label.grid(row=0, column=0, sticky="nsew")

        side = ttk.Frame(self.root, padding=10)
        side.grid(row=0, column=1, sticky="nsew")
        side.columnconfigure(0, weight=1)

        self.status_var = tk.StringVar(value="Connected")
        self.current_fruit_var = tk.StringVar(value="Chua co")
        self.relay_var = tk.StringVar(value="Relay: OFF")
        self.speed_var = tk.IntVar(value=self.args.motor_speed)
        self.queue_var = tk.StringVar(value="Hang cho: 0")

        ttk.Label(side, text="Trang thai").grid(row=0, column=0, sticky="w")
        ttk.Label(side, textvariable=self.status_var).grid(row=1, column=0, sticky="w", pady=(0, 10))

        ttk.Label(side, text="Qua vua nhan dien").grid(row=2, column=0, sticky="w")
        ttk.Label(side, textvariable=self.current_fruit_var, font=("Segoe UI", 16, "bold")).grid(
            row=3, column=0, sticky="w", pady=(0, 10)
        )

        self.snapshot_label = ttk.Label(side, text="Chua co anh", anchor="center")
        self.snapshot_label.grid(row=4, column=0, sticky="ew", pady=(0, 10))

        counts_box = ttk.LabelFrame(side, text="So luong da nhan dien", padding=8)
        counts_box.grid(row=5, column=0, sticky="ew", pady=(0, 10))
        self.count_vars = {
            "apple": tk.StringVar(value="apple: 0"),
            "banana": tk.StringVar(value="banana: 0"),
            "orange": tk.StringVar(value="orange: 0"),
        }
        for idx, label in enumerate(("apple", "banana", "orange")):
            ttk.Label(counts_box, textvariable=self.count_vars[label]).grid(row=idx, column=0, sticky="w")

        controls = ttk.LabelFrame(side, text="Dieu khien", padding=8)
        controls.grid(row=6, column=0, sticky="ew", pady=(0, 10))
        controls.columnconfigure(0, weight=1)

        self.relay_button = ttk.Button(controls, textvariable=self.relay_var, command=self.toggle_relay)
        self.relay_button.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        ttk.Label(controls, text="Toc do dong co").grid(row=1, column=0, sticky="w")
        self.speed_scale = ttk.Scale(
            controls,
            from_=0,
            to=255,
            orient="horizontal",
            command=self.on_speed_changed,
        )
        self.speed_scale.set(self.args.motor_speed)
        self.speed_scale.grid(row=2, column=0, sticky="ew")
        self.speed_label = ttk.Label(controls, text=f"PWM: {self.args.motor_speed}")
        self.speed_label.grid(row=3, column=0, sticky="w", pady=(0, 8))

        self.detect_button = ttk.Button(controls, text="Tam dung nhan dien", command=self.toggle_detection)
        self.detect_button.grid(row=4, column=0, sticky="ew")

        self.fake_ir_button = ttk.Button(controls, text="Gia lap IR", command=self.handle_ir_trigger)
        self.fake_ir_button.grid(row=5, column=0, sticky="ew", pady=(8, 0))

        queue_box = ttk.LabelFrame(side, text="Hang cho IR", padding=8)
        queue_box.grid(row=7, column=0, sticky="nsew")
        side.rowconfigure(7, weight=1)
        queue_box.rowconfigure(1, weight=1)
        queue_box.columnconfigure(0, weight=1)
        ttk.Label(queue_box, textvariable=self.queue_var).grid(row=0, column=0, sticky="w")
        self.queue_list = tk.Listbox(queue_box, height=8)
        self.queue_list.grid(row=1, column=0, sticky="nsew")

        log_box = ttk.LabelFrame(side, text="Log", padding=8)
        log_box.grid(row=8, column=0, sticky="nsew", pady=(10, 0))
        log_box.rowconfigure(0, weight=1)
        log_box.columnconfigure(0, weight=1)
        self.log_text = tk.Text(log_box, height=8, wrap="word")
        self.log_text.grid(row=0, column=0, sticky="nsew")

    def log(self, text: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{timestamp}] {text}\n")
        self.log_text.see("end")

    def send_command(self, command: str) -> None:
        with self.serial_lock:
            self.esp32.write((command + "\n").encode("utf-8"))
            self.esp32.flush()
        self.log(f"Laptop -> ESP32: {command}")

    def toggle_relay(self) -> None:
        self.relay_on = not self.relay_on
        if self.relay_on:
            self.relay_var.set("Relay: ON")
            self.send_command("RELAY:ON")
            self.send_command(f"MOTOR:SPEED:{self.speed_var.get()}")
        else:
            self.relay_var.set("Relay: OFF")
            self.send_command("RELAY:OFF")
            self.send_command("MOTOR:STOP")

    def on_speed_changed(self, value: str) -> None:
        speed = int(float(value))
        self.speed_var.set(speed)
        self.speed_label.config(text=f"PWM: {speed}")
        now = time.monotonic()
        if now - getattr(self, "_last_speed_send_at", 0.0) < 0.15:
            return
        self._last_speed_send_at = now
        self.last_sent_speed = speed
        self.send_command(f"MOTOR:SPEED:{speed}")

    def toggle_detection(self) -> None:
        self.detection_enabled = not self.detection_enabled
        self.detect_button.config(text="Tiep tuc nhan dien" if not self.detection_enabled else "Tam dung nhan dien")

    def _serial_reader_loop(self) -> None:
        while self.running:
            try:
                line = self.esp32.readline().decode("utf-8", errors="replace").strip()
                if line:
                    self.serial_queue.put(line)
            except serial.SerialException as exc:
                self.serial_queue.put(f"SERIAL_ERROR:{exc}")
                break

    def _camera_loop(self) -> None:
        while self.running:
            ok, frame = self.camera.read()
            if not ok:
                self.event_queue.put(("log", "Khong doc duoc frame tu camera"))
                time.sleep(0.2)
                continue

            display = frame.copy()
            detection = None

            if self.detection_enabled:
                results = self.model.predict(frame, imgsz=self.args.imgsz, conf=self.args.conf, verbose=False)
                detection = best_detection(results[0], self.model.names, self.args.conf)

            if detection is not None:
                draw_detection(display, detection)
                now = time.monotonic()
                enough_delay = now - self.last_detection_time >= self.args.cooldown
                changed_label = detection.label != self.last_detection_label
                if enough_delay or changed_label:
                    pending = self._save_detection(frame, detection)
                    self.event_queue.put(("detected", pending))
                    self.last_detection_time = now
                    self.last_detection_label = detection.label

            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(display)

    def _save_detection(self, frame, detection: Detection) -> PendingFruit:
        x1, y1, x2, y2 = detection.xyxy
        h, w = frame.shape[:2]
        pad = 20
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            crop = frame

        detected_at = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{detected_at}_{detection.label}_{detection.confidence:.2f}.jpg"
        image_path = str(self.capture_dir / filename)
        cv2.imwrite(image_path, crop)

        return PendingFruit(
            label=detection.label,
            confidence=detection.confidence,
            image_path=image_path,
            detected_at=detected_at,
        )

    def _poll_events(self) -> None:
        self._update_frame()
        self._update_serial_lines()
        self._update_app_events()
        if self.running:
            self.root.after(50, self._poll_events)

    def _update_frame(self) -> None:
        try:
            frame = self.frame_queue.get_nowait()
        except queue.Empty:
            return

        photo = frame_to_photo(frame, 820, 620)
        self.camera_label.config(image=photo, text="")
        self.camera_label.image = photo

    def _update_serial_lines(self) -> None:
        while True:
            try:
                line = self.serial_queue.get_nowait()
            except queue.Empty:
                break

            self.log(f"ESP32 -> {line}")
            if line == "IR:TRIGGER":
                self.handle_ir_trigger()
            elif line.startswith("SERIAL_ERROR:"):
                self.status_var.set(line)

    def _update_app_events(self) -> None:
        while True:
            try:
                event, payload = self.event_queue.get_nowait()
            except queue.Empty:
                break

            if event == "detected":
                self.add_pending_fruit(payload)
            elif event == "log":
                self.log(payload)

    def add_pending_fruit(self, fruit: PendingFruit) -> None:
        self.pending.append(fruit)
        self.counts[fruit.label] += 1
        self.current_fruit_var.set(f"{fruit.label} ({fruit.confidence:.2f})")
        self.log(f"Nhan dien: {fruit.label} {fruit.confidence:.2f}")
        self._refresh_counts()
        self._refresh_queue()
        self._show_snapshot(fruit.image_path)

    def handle_ir_trigger(self) -> None:
        if not self.pending:
            self.log("IR kich hoat nhung hang cho rong")
            return

        fruit = self.pending.pop(0)
        self.send_command(f"FRUIT:{fruit.label}")
        self.log(f"IR kich hoat, gui qua dau hang cho: {fruit.label}")
        self._refresh_queue()

    def _refresh_counts(self) -> None:
        for label in ("apple", "banana", "orange"):
            self.count_vars[label].set(f"{label}: {self.counts[label]}")

    def _refresh_queue(self) -> None:
        self.queue_var.set(f"Hang cho: {len(self.pending)}")
        self.queue_list.delete(0, "end")
        for idx, fruit in enumerate(self.pending, start=1):
            self.queue_list.insert("end", f"{idx}. {fruit.label} ({fruit.confidence:.2f})")

    def _show_snapshot(self, image_path: str) -> None:
        image = cv2.imread(image_path)
        if image is None:
            return
        photo = frame_to_photo(image, 320, 170)
        self.snapshot_label.config(image=photo, text="")
        self.snapshot_label.image = photo

    def close(self) -> None:
        self.running = False
        try:
            self.send_command("MOTOR:STOP")
            self.send_command("RELAY:OFF")
        except Exception:
            pass
        self.camera.release()
        self.esp32.close()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GUI phan loai trai cay YOLOv8 + ESP32")
    parser.add_argument("--model", default="best.pt", help="Duong dan model YOLOv8 .pt")
    parser.add_argument("--port", required=True, help="Cong COM cua ESP32, vi du COM5")
    parser.add_argument("--camera", default=0, type=int, help="Camera index")
    parser.add_argument("--baud", default=115200, type=int, help="Baudrate ESP32")
    parser.add_argument("--conf", default=0.65, type=float, help="Nguong confidence")
    parser.add_argument("--cooldown", default=2.0, type=float, help="Thoi gian toi thieu giua 2 lan dem")
    parser.add_argument("--imgsz", default=640, type=int, help="Kich thuoc anh dua vao YOLO")
    parser.add_argument("--motor-speed", default=180, type=int, help="Toc do motor mac dinh 0-255")
    parser.add_argument("--capture-dir", default="captures", help="Thu muc luu anh qua vua nhan dien")
    return parser.parse_args()


if __name__ == "__main__":
    app = FruitSorterApp(parse_args())
    app.run()
