import argparse
import time
from dataclasses import dataclass

import cv2
import serial
from ultralytics import YOLO


VALID_CLASSES = {"apple", "banana", "orange"}


@dataclass
class Detection:
    label: str
    confidence: float
    xyxy: tuple[int, int, int, int]


def open_esp32_serial(port: str, baud: int) -> serial.Serial:
    esp32 = serial.Serial(port=port, baudrate=baud, timeout=1)
    time.sleep(2.0)
    esp32.reset_input_buffer()
    return esp32


def send_to_esp32(esp32: serial.Serial, label: str) -> None:
    message = f"FRUIT:{label}\n"
    esp32.write(message.encode("utf-8"))
    esp32.flush()

    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        response = esp32.readline().decode("utf-8", errors="replace").strip()
        if response:
            print(f"ESP32 -> {response}")
        else:
            break


def get_best_detection(result, names: dict[int, str], conf_threshold: float) -> Detection | None:
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
        detection = Detection(
            label=label,
            confidence=confidence,
            xyxy=(x1, y1, x2, y2),
        )

        if best is None or detection.confidence > best.confidence:
            best = detection

    return best


def draw_detection(frame, detection: Detection) -> None:
    x1, y1, x2, y2 = detection.xyxy
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 0), 2)

    text = f"{detection.label} {detection.confidence:.2f}"
    cv2.putText(
        frame,
        text,
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 180, 0),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nhan dien trai cay bang YOLOv8 va gui class sang ESP32"
    )
    parser.add_argument("--model", default="best.pt", help="Duong dan file model .pt")
    parser.add_argument("--port", required=True, help="Cong COM cua ESP32, vi du COM5")
    parser.add_argument("--camera", default=0, type=int, help="Camera index")
    parser.add_argument("--baud", default=115200, type=int, help="Baudrate UART")
    parser.add_argument("--conf", default=0.65, type=float, help="Nguong tin cay")
    parser.add_argument(
        "--cooldown",
        default=2.5,
        type=float,
        help="So giay toi thieu giua 2 lan gui lenh",
    )
    parser.add_argument("--imgsz", default=640, type=int, help="Kich thuoc anh inference")
    parser.add_argument("--debug", action="store_true", help="In them thong tin debug")
    args = parser.parse_args()

    model = YOLO(args.model)
    esp32 = open_esp32_serial(args.port, args.baud)
    camera = cv2.VideoCapture(args.camera)

    if not camera.isOpened():
        raise RuntimeError(f"Khong mo duoc camera index {args.camera}")

    last_sent_label: str | None = None
    last_sent_time = 0.0

    print("Dang chay nhan dien. Nhan phim q de thoat.")
    print(f"Model: {args.model}")
    print(f"Model classes: {model.names}")
    print(f"ESP32 port: {args.port}, baud: {args.baud}")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Khong doc duoc frame tu camera")
                break

            results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
            detection = get_best_detection(results[0], model.names, args.conf)

            now = time.monotonic()

            if args.debug and int(now * 2) % 2 == 0:
                box_count = 0 if results[0].boxes is None else len(results[0].boxes)
                print(f"YOLO boxes: {box_count}")

            if detection is not None:
                draw_detection(frame, detection)

                cooldown_done = now - last_sent_time >= args.cooldown
                label_changed = detection.label != last_sent_label

                if cooldown_done or label_changed:
                    print(f"Laptop -> ESP32: FRUIT:{detection.label}")
                    send_to_esp32(esp32, detection.label)
                    last_sent_label = detection.label
                    last_sent_time = now

            cv2.imshow("Fruit Detection YOLOv8", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        esp32.close()


if __name__ == "__main__":
    main()
