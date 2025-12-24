import sys
import os
import csv
import cv2
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QSizePolicy,
    QMessageBox, QScrollArea
)
from PyQt5.QtCore import QTimer, Qt, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPainter
from ultralytics import YOLO
import pytesseract
import hashlib

# Configure pytesseract path if needed (Windows)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

IMG_FOLDER = "parked_vehicles"
LOG_FILE = "parking_log.csv"
os.makedirs(IMG_FOLDER, exist_ok=True)

model = YOLO('yolo12x.pt')  # Adjust model path as needed


def average_color(image):
    if image.size == 0:
        return (0, 0, 0)
    avg_color_row = np.average(image, axis=0)
    avg_color = np.average(avg_color_row, axis=0)
    return tuple(map(int, avg_color))


def bgr_to_hex(color_tuple):
    return '#{:02x}{:02x}{:02x}'.format(color_tuple[2], color_tuple[1], color_tuple[0])


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
    boxBArea = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))
    denom = float(boxAArea + boxBArea - interArea)
    if denom == 0:
        return 0
    return interArea / denom


def ocr_license_plate(plate_img):
    try:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(thresh, config=config)
        return text.strip()
    except Exception:
        return ''


def write_log(data):
    file_exists = os.path.isfile(LOG_FILE)
    header = [
        "first_seen",
        "last_seen",
        "image_file",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "vehicle_type",
        "avg_color_hex",
        "license_plate",
    ]
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)


def estimate_empty_parking_spots(vehicles_boxes, min_gap_pixels=60):
    empty_spots = []
    if len(vehicles_boxes) < 2:
        return empty_spots
    vehicles_boxes_sorted = sorted(vehicles_boxes, key=lambda b: b[0])
    for i in range(len(vehicles_boxes_sorted) - 1):
        left_box = vehicles_boxes_sorted[i]
        right_box = vehicles_boxes_sorted[i + 1]
        gap = right_box[0] - left_box[2]
        if gap >= min_gap_pixels:
            empty_spots.append((left_box[2], right_box[0]))
    return empty_spots


def vehicle_hash(bbox, label, plate_text):
    data = f"{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}_{label}_{plate_text}"
    return hashlib.sha256(data.encode()).hexdigest()


class LogEntryWidget(QWidget):
    def __init__(self, log_entry):
        super().__init__()
        self.log_entry = log_entry
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(15)

        img_label = QLabel()
        img_path = os.path.join(IMG_FOLDER, self.log_entry['image_file'])
        pixmap = QPixmap(img_path)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(200, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            img_label.setPixmap(pixmap)
        img_label.setFixedSize(200, 140)
        layout.addWidget(img_label)

        info_layout = QVBoxLayout()
        font_bold = QFont()
        font_bold.setBold(True)

        type_label = QLabel(f"Type: {self.log_entry['vehicle_type']}")
        type_label.setFont(font_bold)
        plate_label = QLabel(f"Plate: {self.log_entry['license_plate'] or 'N/A'}")
        first_seen_label = QLabel(f"First Seen: {self.log_entry['first_seen'][11:19]}")
        last_seen_label = QLabel(f"Last Seen: {self.log_entry['last_seen'][11:19]}")
        color_label = QLabel(f"Avg. Color: {self.log_entry['avg_color_hex']}")
        color_label.setStyleSheet(f"background-color: {self.log_entry['avg_color_hex']}; border: 1px solid #ccc; padding: 3px; border-radius: 3px;")

        for lbl in [type_label, plate_label, first_seen_label, last_seen_label, color_label]:
            info_layout.addWidget(lbl)

        layout.addLayout(info_layout)
        self.setLayout(layout)


class ParkingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Parking Detection with Logs & Images")
        self.previous_vehicles = {}
        self.setMinimumSize(1100, 720)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.logs_container = QWidget()
        self.logs_layout = QVBoxLayout()
        self.logs_container.setLayout(self.logs_layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.logs_container)
        self.scroll_area.setFixedWidth(420)

        layout = QHBoxLayout()
        layout.addWidget(self.video_label, 3)
        layout.addWidget(self.scroll_area, 1)
        self.setLayout(layout)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Failed to open webcam")
            sys.exit(1)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        results = model(frame)
        detections = results[0]

        vehicles_boxes = []
        vehicles_labels = []
        vehicles_plate_texts = []

        for det in detections.boxes:
            cls_id = int(det.cls[0])
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            label = None
            if cls_id == 2:
                label = "Car"
            elif cls_id == 7:
                label = "Truck"
            elif cls_id == 8:
                plate_img = frame[y1:y2, x1:x2]
                plate_text = ocr_license_plate(plate_img)
                vehicles_boxes.append((x1, y1, x2, y2))
                vehicles_labels.append("LicensePlate")
                vehicles_plate_texts.append(plate_text)
                continue
            else:
                continue
            vehicles_boxes.append((x1, y1, x2, y2))
            vehicles_labels.append(label)
            vehicles_plate_texts.append("")

        current_hashes = {}

        to_remove = []
        now = datetime.now()
        for key, data in self.previous_vehicles.items():
            data['seen_this_frame'] = False

        for bbox, label, plate_text in zip(vehicles_boxes, vehicles_labels, vehicles_plate_texts):
            if label not in ['Car', 'Truck']:
                continue
            matched_key = None
            for key, data in self.previous_vehicles.items():
                if iou(bbox, data['bbox']) > 0.5:
                    matched_key = key
                    break
            if matched_key:
                data = self.previous_vehicles[matched_key]
                data['bbox'] = bbox
                data['last_seen'] = now
                data['plate_text'] = plate_text or data['plate_text']
                data['seen_this_frame'] = True
                current_hashes[matched_key] = True
            else:
                h = vehicle_hash(bbox, label, plate_text)
                avg_col = average_color(frame[bbox[1]:bbox[3], bbox[0]:bbox[2]])
                avg_hex = bgr_to_hex(avg_col)
                img_name = f"{label.lower()}_{now.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                img_path = os.path.join(IMG_FOLDER, img_name)
                try:
                    cv2.imwrite(img_path, frame[bbox[1]:bbox[3], bbox[0]:bbox[2]])
                except Exception as e:
                    print(f"Error saving image: {e}")
                    continue
                self.previous_vehicles[h] = {
                    'bbox': bbox,
                    'label': label,
                    'plate_text': plate_text,
                    'avg_color_hex': avg_hex,
                    'first_seen': now,
                    'last_seen': now,
                    'image_file': img_name,
                    'seen_this_frame': True
                }
                current_hashes[h] = True

        disappear_time = 2.0
        for key, data in list(self.previous_vehicles.items()):
            if not data['seen_this_frame']:
                delta = (now - data['last_seen']).total_seconds()
                if delta > disappear_time:
                    log_entry = {
                        "first_seen": data['first_seen'].isoformat(),
                        "last_seen": data['last_seen'].isoformat(),
                        "image_file": data['image_file'],
                        "bbox_x1": data['bbox'][0],
                        "bbox_y1": data['bbox'][1],
                        "bbox_x2": data['bbox'][2],
                        "bbox_y2": data['bbox'][3],
                        "vehicle_type": data['label'],
                        "avg_color_hex": data['avg_color_hex'],
                        "license_plate": data['plate_text']
                    }
                    write_log(log_entry)
                    self.add_log_item(log_entry)
                    to_remove.append(key)

        for key in to_remove:
            del self.previous_vehicles[key]

        for bbox, label, plate_text in zip(vehicles_boxes, vehicles_labels, vehicles_plate_texts):
            x1, y1, x2, y2 = bbox
            if label == "Car":
                color = (0, 255, 0)
            elif label == "Truck":
                color = (0, 128, 255)
            else:
                color = (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = label
            if plate_text:
                text += f": {plate_text}"
            cv2.putText(frame, text, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        empty_spots = estimate_empty_parking_spots(vehicles_boxes)
        if vehicles_boxes:
            baseline = int(np.median([b[3] for b in vehicles_boxes]))
        else:
            baseline = frame.shape[0] - 10
        for left_x, right_x in empty_spots:
            y2 = baseline + 10
            y1 = y2 - 40
            cv2.rectangle(frame, (left_x, y1), (right_x, y2), (0, 255, 255), 2)
            cv2.putText(frame, "Empty Spot", (left_x, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pixmap)

    def add_log_item(self, log_entry):
        log_widget = LogEntryWidget(log_entry)
        self.logs_layout.addWidget(log_widget)
        self.scroll_area = self.layout().itemAt(1).widget()
        self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())

    def closeEvent(self, event):
        self.cap.release()
        event.accept()


if __name__ == "__main__":
    IMG_FOLDER = os.path.join(os.getcwd(), "parked_vehicles")
    os.makedirs(IMG_FOLDER, exist_ok=True)
    app = QApplication(sys.argv)
    window = ParkingApp()
    window.show()
    sys.exit(app.exec_())
