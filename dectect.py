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


def open_serial(port: str, baud: int) -> serial.Serial:
    ser = serial.Serial(port=port, baudrate=baud, timeout=1)
    time.sleep(2.0)
    ser.reset_input_buffer()
    return ser


def send_fruit_command(ser: serial.Serial, label: str) -> None:
    command = f"FRUIT:{label}\n"
    ser.write(command.encode("utf-8"))
    ser.flush()
    response = ser.readline().decode("utf-8", errors="replace").strip()
    if response:
        print(f"ESP32: {response}")


def best_detection(result, names: dict[int, str], conf_threshold: float) -> Detection | None:
    best: Detection | None = None

    if result.boxes is None:
        return None

    for box in result.boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = names.get(cls_id, str(cls_id))

        if label not in VALID_CLASSES or conf < conf_threshold:
            continue

        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        current = Detection(label=label, confidence=conf, xyxy=(x1, y1, x2, y2))
        if best is None or current.confidence > best.confidence:
            best = current

    return best


def draw_detection(frame, detection: Detection) -> None:
    x1, y1, x2, y2 = detection.xyxy
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 0), 2)
    text = f"{detection.label} {detection.confidence:.2f}"
    cv2.putText(
        frame,
        text,
        (x1, max(24, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 180, 0),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fruit sorter using YOLOv8 and ESP32")
    parser.add_argument("--model", required=True, help="Path to YOLOv8 .pt model, for example best.pt")
    parser.add_argument("--port", required=True, help="ESP32 serial port, for example COM5")
    parser.add_argument("--camera", default=0, type=int, help="Camera index")
    parser.add_argument("--baud", default=115200, type=int, help="Serial baud rate")
    parser.add_argument("--conf", default=0.65, type=float, help="Detection confidence threshold")
    parser.add_argument("--cooldown", default=2.5, type=float, help="Seconds between sort commands")
    parser.add_argument("--imgsz", default=640, type=int, help="YOLO inference image size")
    args = parser.parse_args()

    model = YOLO(args.model)
    ser = open_serial(args.port, args.baud)
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    last_label: str | None = None
    last_sent_at = 0.0

    print("Running. Press q to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame read failed")
                break

            results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
            detection = best_detection(results[0], model.names, args.conf)

            now = time.monotonic()
            if detection is not None:
                draw_detection(frame, detection)

                cooldown_done = now - last_sent_at >= args.cooldown
                changed_label = detection.label != last_label
                if cooldown_done or changed_label:
                    print(f"Detected: {detection.label} ({detection.confidence:.2f})")
                    send_fruit_command(ser, detection.label)
                    last_label = detection.label
                    last_sent_at = now

            cv2.imshow("Fruit sorter", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        ser.close()


if __name__ == "__main__":
    main()
