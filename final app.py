import sys
import os
import csv
import cv2
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QSizePolicy,
    QMessageBox, QScrollArea, QLineEdit, QPushButton
)
from PyQt5.QtCore import QTimer, Qt, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont
from ultralytics import YOLO
from paddleocr import PaddleOCR
import hashlib

# Configure PaddleOCR: English, with text line orientation enabled, no logs
ocr_model = PaddleOCR(use_textline_orientation=True, lang='en')

IMG_FOLDER = "parked_vehicles"
LOG_FILE = "parking_log.csv"
os.makedirs(IMG_FOLDER, exist_ok=True)

model = YOLO('yolo12l.pt')  # Adjust path if needed


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


def preprocess_plate_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize preserving aspect ratio, height=60 pixels (more detail)
    h, w = gray.shape
    new_h = 60
    new_w = int(w * (new_h / h))
    resized = cv2.resize(gray, (new_w, new_h))
    # Denoising
    denoised = cv2.fastNlMeansDenoising(resized, None, h=30)
    # Contrast enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    return enhanced


def ocr_license_plate_paddle(img):
    proc_img = preprocess_plate_image(img)
    result = ocr_model.ocr(proc_img, cls=True)
    if result and len(result) > 0:
        text = result[0][1][0]
        cleaned = ''.join(filter(str.isalnum, text))
        return cleaned.upper()
    else:
        return ''


def write_log(data):
    file_exists = os.path.isfile(LOG_FILE)
    header = ["first_seen", "last_seen", "image_file", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
              "vehicle_type", "avg_color_hex", "license_plate"]
    with open(LOG_FILE, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)


def estimate_empty_parking_spots(boxes, min_gap=60):
    spots = []
    if len(boxes) < 2:
        return spots
    sorted_boxes = sorted(boxes, key=lambda b: b[0])
    for i in range(len(sorted_boxes) - 1):
        gap = sorted_boxes[i + 1][0] - sorted_boxes[i][2]
        if gap >= min_gap:
            spots.append((sorted_boxes[i][2], sorted_boxes[i + 1][0]))
    return spots


def vehicle_hash(box, label, plate):
    string = f"{box[0]}_{box[1]}_{box[2]}_{box[3]}_{label}_{plate}"
    return hashlib.sha256(string.encode()).hexdigest()


class LogEntryWidget(QWidget):
    def __init__(self, log):
        super().__init__()
        self.log = log
        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(15)

        image_label = QLabel()
        image_path = os.path.join(IMG_FOLDER, self.log['image_file'])
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(200, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            image_label.setPixmap(pixmap)
        image_label.setFixedSize(200, 140)
        layout.addWidget(image_label)

        info_layout = QVBoxLayout()
        font_bold = QFont()
        font_bold.setBold(True)

        type_label = QLabel(f"Type: {self.log['vehicle_type']}")
        type_label.setFont(font_bold)
        plate_label = QLabel(f"Plate: {self.log['license_plate'] or 'N/A'}")
        first_seen_label = QLabel(f"First Seen: {self.log['first_seen'][11:19]}")
        last_seen_label = QLabel(f"Last Seen: {self.log['last_seen'][11:19]}")
        color_label = QLabel(f"Avg Color: {self.log['avg_color_hex']}")
        color_label.setStyleSheet(f"background-color: {self.log['avg_color_hex']}; border: 1px solid #ccc; padding: 3px; border-radius: 3px;")

        for w in [type_label, plate_label, first_seen_label, last_seen_label, color_label]:
            info_layout.addWidget(w)

        layout.addLayout(info_layout)
        self.setLayout(layout)


class ParkingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Parking Detection with License Plate OCR")
        self.previous_vehicles = {}
        self.setMinimumSize(1200, 720)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.logs_container = QWidget()
        self.logs_layout = QVBoxLayout()
        self.logs_container.setLayout(self.logs_layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.logs_container)
        self.scroll_area.setFixedWidth(440)

        # Search bar and button
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter license plate to search...")
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.search_license_plate)

        # Layout for search
        search_layout = QHBoxLayout()
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.search_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(search_layout)
        main_layout.addWidget(self.video_label, 3)
        main_layout.addWidget(self.scroll_area, 1)
        self.setLayout(main_layout)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Cannot open webcam")
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
                plate_text = ocr_license_plate_paddle(plate_img)
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
            color = (0, 255, 0) if label == "Car" else (0, 128, 255) if label == "Truck" else (255, 0, 0)
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
        self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum())

    def search_license_plate(self):
        search_text = self.search_input.text().strip().upper()
        self.clear_logs()
        if not search_text:
            for key, data in self.previous_vehicles.items():
                self.add_log_item(data)
            return

        for key, data in self.previous_vehicles.items():
            if data['plate_text'] and search_text in data['plate_text']:
                self.add_log_item(data)

    def clear_logs(self):
        for i in reversed(range(self.logs_layout.count())):
            widget = self.logs_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

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
