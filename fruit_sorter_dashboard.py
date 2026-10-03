import argparse
import base64
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import serial
import tkinter as tk
from ultralytics import YOLO


VALID_CLASSES = {"apple", "banana", "orange"}
CLASS_VI = {
    "apple": "TAO",
    "banana": "CHUOI",
    "orange": "CAM",
}


@dataclass
class Detection:
    label: str
    confidence: float
    xyxy: tuple[int, int, int, int]


@dataclass
class PendingFruit:
    label: str
    confidence: float
    detected_at: str


def parse_roi(roi_text: str | None) -> tuple[float, float, float, float] | None:
    if not roi_text:
        return None

    values = [float(part.strip()) for part in roi_text.split(",")]
    if len(values) != 4:
        raise ValueError("ROI phai co dang x1,y1,x2,y2")

    x1, y1, x2, y2 = values
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("ROI dung ti le 0..1, vi du 0.25,0.15,0.85,0.85")

    return x1, y1, x2, y2


def roi_to_pixels(frame, roi: tuple[float, float, float, float] | None) -> tuple[int, int, int, int]:
    h, w = frame.shape[:2]
    if roi is None:
        return 0, 0, w, h

    x1, y1, x2, y2 = roi
    return int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)


def best_detection(
    result,
    names: dict[int, str],
    conf_threshold: float,
    offset_x: int = 0,
    offset_y: int = 0,
) -> Detection | None:
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
        x1 += offset_x
        x2 += offset_x
        y1 += offset_y
        y2 += offset_y
        detection = Detection(label, confidence, (x1, y1, x2, y2))
        if best is None or detection.confidence > best.confidence:
            best = detection
    return best


def draw_detection(frame, detection: Detection) -> None:
    x1, y1, x2, y2 = detection.xyxy
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(
        frame,
        f"{CLASS_VI[detection.label]} {detection.confidence * 100:.1f}%",
        (x1, max(35, y1 - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        3,
        cv2.LINE_AA,
    )


def frame_to_photo(frame, max_width: int, max_height: int) -> tk.PhotoImage:
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    resized = cv2.resize(frame, new_size)
    ok, encoded = cv2.imencode(".png", resized)
    if not ok:
        raise RuntimeError("Cannot encode frame")
    return tk.PhotoImage(data=base64.b64encode(encoded.tobytes()))


class DashboardApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = tk.Tk()
        self.root.title("He thong phan loai trai cay thong minh")
        self.root.geometry("1360x820")
        self.root.minsize(1200, 760)
        self.root.configure(bg="#eef2f6")

        self.roi = parse_roi(args.roi)

        self.model = YOLO(args.model)
        self.esp32 = serial.Serial(args.port, args.baud, timeout=0.05)
        time.sleep(2.0)
        self.esp32.reset_input_buffer()
        self.serial_lock = threading.Lock()

        self.camera = cv2.VideoCapture(args.camera)
        if not self.camera.isOpened():
            raise RuntimeError(f"Khong mo duoc camera index {args.camera}")

        self.running = True
        self.system_running = False
        self.last_detection_time = 0.0
        self.last_detection_label: str | None = None

        self.counts = {"apple": 0, "banana": 0, "orange": 0}
        self.pending: list[PendingFruit] = []
        self.last_result = "--"

        self.frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self.event_queue: queue.Queue = queue.Queue()
        self.serial_queue: queue.Queue[str] = queue.Queue()

        self._build_ui()

        threading.Thread(target=self._serial_loop, daemon=True).start()
        threading.Thread(target=self._camera_loop, daemon=True).start()

        self.root.after(50, self._poll)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build_ui(self) -> None:
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        header = tk.Frame(self.root, bg="#081525", height=78)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)
        header.grid_columnconfigure(0, weight=1)

        tk.Label(
            header,
            text="HE THONG PHAN LOAI TRAI CAY THONG MINH",
            bg="#081525",
            fg="white",
            font=("Segoe UI", 25, "bold"),
            anchor="w",
            padx=24,
        ).grid(row=0, column=0, sticky="nsew")

        body = tk.Frame(self.root, bg="#eef2f6", padx=18, pady=14)
        body.grid(row=1, column=0, sticky="nsew")
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=1, minsize=780)
        body.grid_columnconfigure(1, weight=0, minsize=430)

        left = tk.Frame(body, bg="#eef2f6")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        left.grid_rowconfigure(0, weight=0)
        left.grid_rowconfigure(1, weight=0)
        left.grid_columnconfigure(0, weight=1)

        self.camera_panel = tk.Frame(left, bg="white", highlightbackground="#d7dde5", highlightthickness=1)
        self.camera_panel.grid(row=0, column=0, sticky="ew")
        self.camera_panel.config(height=500, width=780)
        self.camera_panel.grid_propagate(False)
        self.camera_panel.grid_rowconfigure(0, weight=1)
        self.camera_panel.grid_columnconfigure(0, weight=1)

        self.camera_label = tk.Label(self.camera_panel, bg="#111827", fg="white", text="CAMERA")
        self.camera_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.result_banner = tk.Label(
            self.camera_panel,
            text="DANG CHO NHAN DIEN",
            bg="#0b0f14",
            fg="#21e329",
            font=("Segoe UI", 24, "bold"),
            anchor="w",
            padx=22,
        )
        self.result_banner.place(x=28, y=24, width=520, height=68)

        self.camera_footer = tk.Label(
            self.camera_panel,
            text="AI: --",
            bg="white",
            fg="#2ca44f",
            font=("Segoe UI", 13, "bold"),
            anchor="e",
            padx=16,
        )
        self.camera_footer.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 8))

        bottom = tk.Frame(left, bg="#eef2f6")
        bottom.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        bottom.grid_columnconfigure((0, 1, 2), weight=1)

        self.cards = {}
        self._create_card(bottom, 0, "KHO 1", "TAO", "apple", "#d93025")
        self._create_card(bottom, 1, "KHO 2", "CHUOI", "banana", "#1a73e8")
        self._create_card(bottom, 2, "KHO 3", "CAM", "orange", "#f9ab00")

        right = tk.Frame(body, bg="#eef2f6")
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_rowconfigure(2, weight=1)
        right.grid_columnconfigure(0, weight=1)

        status = self._panel(right, "TRANG THAI HE THONG")
        status.grid(row=0, column=0, sticky="ew")
        self.status_label = tk.Label(
            status,
            text="Cho trai cay moi",
            bg="white",
            fg="#2ca44f",
            font=("Segoe UI", 15, "bold"),
            anchor="w",
        )
        self.status_label.pack(fill="x", padx=18, pady=(6, 8))
        self.result_label = tk.Label(status, text="Ket qua: --", bg="white", fg="#2ca44f", font=("Segoe UI", 12), anchor="w")
        self.result_label.pack(fill="x", padx=18, pady=4)
        self.queue_label = tk.Label(status, text="Hang doi: --", bg="white", fg="#2457b8", font=("Segoe UI", 12, "bold"), anchor="w")
        self.queue_label.pack(fill="x", padx=18, pady=4)
        self.ir_label = tk.Label(status, text="IR: Khong", bg="white", fg="#333333", font=("Segoe UI", 11), anchor="w")
        self.ir_label.pack(fill="x", padx=18, pady=(4, 14))

        control = self._panel(right, "DIEU KHIEN")
        control.grid(row=1, column=0, sticky="ew", pady=(12, 0))

        row = tk.Frame(control, bg="white")
        row.pack(fill="x", padx=18, pady=(6, 10))
        self._button(row, "START", "#2fac49", self.start_system).pack(side="left", expand=True, fill="x", ipady=9, padx=(0, 8))
        self._button(row, "STOP", "#d93025", self.stop_system).pack(side="left", expand=True, fill="x", ipady=9, padx=(8, 0))

        clear_row = tk.Frame(control, bg="white")
        clear_row.pack(fill="x", padx=18, pady=(0, 10))
        self._button(clear_row, "XOA TAT CA HANG CHO", "#185abc", self.clear_queue).pack(fill="x", ipady=9)

        speed_box = tk.Frame(control, bg="white")
        speed_box.pack(fill="x", padx=18, pady=(0, 16))
        self.speed_value = tk.IntVar(value=self.args.motor_speed)
        self.speed_text = tk.Label(speed_box, text=f"Toc do dong co: {self.args.motor_speed}", bg="white", fg="#333", font=("Segoe UI", 11, "bold"))
        self.speed_text.pack(anchor="w")
        tk.Scale(
            speed_box,
            from_=0,
            to=255,
            orient="horizontal",
            variable=self.speed_value,
            command=self.change_speed,
            bg="white",
            highlightthickness=0,
        ).pack(fill="x")

        queue_panel = self._panel(right, "HANG CHO")
        queue_panel.grid(row=2, column=0, sticky="nsew", pady=(12, 0))
        self.queue_list = tk.Listbox(queue_panel, bg="#f7f9fc", borderwidth=0, font=("Segoe UI", 11))
        self.queue_list.pack(fill="both", expand=True, padx=18, pady=(6, 16))

    def _panel(self, parent, title: str) -> tk.Frame:
        outer = tk.Frame(parent, bg="white", highlightbackground="#d7dde5", highlightthickness=1)
        tk.Label(outer, text=title, bg="white", fg="#1f2937", font=("Segoe UI", 13, "bold"), anchor="w").pack(
            fill="x", padx=18, pady=(16, 4)
        )
        return outer

    def _button(self, parent, text: str, color: str, command) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=color,
            fg="white",
            activebackground=color,
            activeforeground="white",
            relief="flat",
            font=("Segoe UI", 11, "bold"),
            cursor="hand2",
        )

    def _create_card(self, parent, column: int, title: str, label: str, key: str, color: str) -> None:
        card = tk.Frame(parent, bg="white", highlightbackground="#d7dde5", highlightthickness=1)
        card.grid(row=0, column=column, sticky="ew", padx=6)
        card.grid_columnconfigure(0, weight=1)

        tk.Label(card, text=title, bg="white", fg="#1f2937", font=("Segoe UI", 17, "bold"), anchor="w").grid(
            row=0, column=0, sticky="ew", padx=18, pady=(14, 0)
        )
        tk.Label(card, text=label, bg="white", fg=color, font=("Segoe UI", 13, "bold"), anchor="w").grid(
            row=1, column=0, sticky="ew", padx=18
        )
        count = tk.Label(card, text="0", bg="white", fg="#111827", font=("Segoe UI", 22, "bold"), anchor="e")
        count.grid(row=0, column=1, rowspan=2, sticky="e", padx=18)

        canvas = tk.Canvas(card, height=14, bg="white", highlightthickness=0)
        canvas.grid(row=2, column=0, columnspan=2, sticky="ew", padx=18, pady=(8, 4))
        remain = tk.Label(card, text="Da nhan dien 0 qua", bg="white", fg="#2ca44f", font=("Segoe UI", 10), anchor="w")
        remain.grid(row=3, column=0, columnspan=2, sticky="ew", padx=18, pady=(0, 14))

        self.cards[key] = {"count": count, "bar": canvas, "text": remain, "color": color}

    def log(self, text: str) -> None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {text}")

    def send_command(self, cmd: str) -> None:
        with self.serial_lock:
            self.esp32.write((cmd + "\n").encode("utf-8"))
            self.esp32.flush()
        self.log(f"Laptop -> ESP32: {cmd}")

    def start_system(self) -> None:
        self.system_running = True
        self.send_command("RELAY:ON")
        self.send_command(f"MOTOR:SPEED:{self.speed_value.get()}")
        self.status_label.config(text="Bang chuyen dang chay")

    def stop_system(self) -> None:
        self.system_running = False
        self.send_command("MOTOR:STOP")
        self.send_command("RELAY:OFF")
        self.status_label.config(text="He thong dung")

    def clear_queue(self) -> None:
        self.pending.clear()
        self._update_queue()
        self.send_command("QUEUE:CLEAR")
        self.queue_label.config(text="Hang doi: da xoa")

    def change_speed(self, value: str) -> None:
        speed = int(float(value))
        self.speed_text.config(text=f"Toc do dong co: {speed}")
        now = time.monotonic()
        if now - getattr(self, "_last_speed_send", 0.0) < 0.15:
            return
        self._last_speed_send = now
        self.send_command(f"MOTOR:SPEED:{speed}")

    def _serial_loop(self) -> None:
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
                time.sleep(0.1)
                continue

            display = frame.copy()
            detection = None
            rx1, ry1, rx2, ry2 = roi_to_pixels(frame, self.roi)
            roi_frame = frame[ry1:ry2, rx1:rx2]

            if self.roi is not None:
                cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (255, 180, 0), 2)
                cv2.putText(
                    display,
                    "VUNG NHAN DIEN",
                    (rx1 + 8, max(24, ry1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 180, 0),
                    2,
                    cv2.LINE_AA,
                )

            results = self.model.predict(roi_frame, imgsz=self.args.imgsz, conf=self.args.conf, verbose=False)
            detection = best_detection(results[0], self.model.names, self.args.conf, rx1, ry1)

            if detection is not None:
                draw_detection(display, detection)
                now = time.monotonic()
                enough_delay = now - self.last_detection_time >= self.args.cooldown
                if enough_delay:
                    pending = PendingFruit(
                        label=detection.label,
                        confidence=detection.confidence,
                        detected_at=datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3],
                    )
                    self.event_queue.put(("detected", pending))
                    self.last_detection_time = now
                    self.last_detection_label = detection.label

            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(display)

    def _poll(self) -> None:
        self._poll_frame()
        self._poll_serial()
        self._poll_events()
        if self.running:
            self.root.after(50, self._poll)

    def _poll_frame(self) -> None:
        try:
            frame = self.frame_queue.get_nowait()
        except queue.Empty:
            return
        photo = frame_to_photo(frame, 740, 420)
        self.camera_label.config(image=photo, text="")
        self.camera_label.image = photo

    def _poll_serial(self) -> None:
        while True:
            try:
                line = self.serial_queue.get_nowait()
            except queue.Empty:
                break
            self.log(f"ESP32 -> {line}")
            if line.startswith("IR:TRIGGER"):
                self.ir_label.config(text="IR: Co vat", fg="#d93025")
                self.root.after(600, lambda: self.ir_label.config(text="IR: Khong", fg="#333333"))
            elif line.startswith("SORT:DONE:"):
                fruit = line.split(":", 2)[2].strip().lower()
                self.handle_sort_done(fruit)
            elif line.startswith("SERIAL_ERROR:"):
                self.status_label.config(text=line, fg="#d93025")

    def _poll_events(self) -> None:
        while True:
            try:
                event, payload = self.event_queue.get_nowait()
            except queue.Empty:
                break
            if event == "detected":
                self.add_pending(payload)

    def add_pending(self, fruit: PendingFruit) -> None:
        self.pending.append(fruit)
        self.counts[fruit.label] += 1
        percent = fruit.confidence * 100
        name = CLASS_VI[fruit.label]
        self.last_result = f"{name} - {percent:.1f}%"
        self.result_banner.config(text=f"{name} {percent:.1f}%")
        self.camera_footer.config(text=f"AI: {name} {percent:.1f}%")
        self.result_label.config(text=f"Ket qua: {name} - {percent:.1f}%")
        self.status_label.config(text=f"Da nhan dien {name}")
        self._update_cards()
        self._update_queue()
        self.send_command(f"FRUIT:{fruit.label}")

    def handle_sort_done(self, fruit_label: str) -> None:
        if not self.pending:
            self.queue_label.config(text="Hang doi: rong")
            self.log(f"ESP32 da gat {fruit_label}, nhung hang cho GUI rong")
            return

        fruit = self.pending.pop(0)
        expected = fruit.label
        if expected != fruit_label:
            self.log(f"Canh bao queue lech: GUI={expected}, ESP32={fruit_label}")

        name = CLASS_VI.get(fruit_label, fruit_label.upper())
        self.queue_label.config(text=f"Da gat xong: {name}")
        self._update_queue()

    def _update_cards(self) -> None:
        max_count = max(1, max(self.counts.values()))
        for key, card in self.cards.items():
            count = self.counts[key]
            card["count"].config(text=str(count))
            card["text"].config(text=f"Da nhan dien {count} qua")
            bar = card["bar"]
            bar.delete("all")
            width = max(1, bar.winfo_width())
            fill_width = int(width * count / max_count)
            bar.create_rectangle(0, 2, width, 12, fill="#d9dee7", outline="")
            bar.create_rectangle(0, 2, fill_width, 12, fill=card["color"], outline="")

    def _update_queue(self) -> None:
        self.queue_label.config(text=f"Hang doi: {len(self.pending)} qua")
        self.queue_list.delete(0, "end")
        for index, fruit in enumerate(self.pending, start=1):
            self.queue_list.insert("end", f"{index}. {CLASS_VI[fruit.label]} - {fruit.confidence * 100:.1f}%")

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
    parser = argparse.ArgumentParser(description="Dashboard phan loai trai cay YOLOv8 + ESP32")
    parser.add_argument("--model", default="best.pt")
    parser.add_argument("--port", required=True)
    parser.add_argument("--camera", default=0, type=int)
    parser.add_argument("--baud", default=115200, type=int)
    parser.add_argument("--conf", default=0.65, type=float)
    parser.add_argument("--cooldown", default=5.0, type=float)
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--motor-speed", default=180, type=int)
    parser.add_argument(
        "--roi",
        default=None,
        help="Gioi han vung nhan dien theo ti le x1,y1,x2,y2. Vi du: 0.25,0.15,0.85,0.85",
    )
    return parser.parse_args()


if __name__ == "__main__":
    DashboardApp(parse_args()).run()
