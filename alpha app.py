import os
import cv2
import csv
import numpy as np
from datetime import datetime
from flask import Flask, render_template, Response, send_from_directory
from ultralytics import YOLO
import pytesseract

app = Flask(__name__)

# Directory for vehicle images and CSV in current folder
IMG_DIR = os.path.join(os.path.abspath('.'), 'parked_vehicles')
CSV_LOG_FILE = os.path.join(os.path.abspath('.'), 'parking_log.csv')

# Create parked_vehicles directory if it doesn't exist
os.makedirs(IMG_DIR, exist_ok=True)

# Load your YOLO model (make sure the model file path is correct)
model = YOLO('yolo12n.pt')

# Configure pytesseract to point to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def average_color(image):
    if image.size == 0:
        return (0, 0, 0)
    avg_color_per_row = np.average(image, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
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
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(thresh, config=config)
    return text.strip()

def write_log(data, csv_file=CSV_LOG_FILE):
    file_exists = os.path.isfile(csv_file)
    header = ['timestamp', 'image_file', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'vehicle_type', 'avg_color_hex', 'license_plate']
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

def read_logs(csv_file=CSV_LOG_FILE):
    if not os.path.isfile(csv_file):
        return []
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        logs = list(reader)
    return logs[::-1]  # reverse for newest first

# Add route to serve vehicle images from parked_vehicles directory
@app.route('/images/<path:filename>')
def images(filename):
    return send_from_directory(IMG_DIR, filename)

def generate_frames():
    cap = cv2.VideoCapture(0)
    previous_logged_boxes = []
    while True:
        success, frame = cap.read()
        if not success:
            break
        results = model(frame)
        detections = results[0]

        vehicles_boxes = []
        vehicles_labels = []
        vehicles_plate_texts = []
        current_logged_boxes = []

        for det in detections.boxes:
            cls_id = int(det.cls[0])
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            label = None
            if cls_id == 2:
                label = 'Car'
            elif cls_id == 7:
                label = 'Truck'
            elif cls_id == 8:
                plate_img = frame[y1:y2, x1:x2]
                plate_text = ocr_license_plate(plate_img)
                vehicles_boxes.append((x1, y1, x2, y2))
                vehicles_labels.append('LicensePlate')
                vehicles_plate_texts.append(plate_text)
                continue
            else:
                continue
            vehicles_boxes.append((x1, y1, x2, y2))
            vehicles_labels.append(label)
            vehicles_plate_texts.append('')

        for bbox, label in zip(vehicles_boxes, vehicles_labels):
            if label not in ['Car', 'Truck']:
                continue
            if any(iou(bbox, prev_box) > 0.5 for prev_box in previous_logged_boxes):
                continue
            x1, y1, x2, y2 = bbox
            vehicle_img = frame[y1:y2, x1:x2].copy()
            if vehicle_img.size == 0:
                continue
            avg_col = average_color(vehicle_img)
            avg_hex = bgr_to_hex(avg_col)
            timestamp = datetime.now()
            timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S_%f')
            img_name = f'{label.lower()}_{timestamp_str}.jpg'
            img_path = os.path.join(IMG_DIR, img_name)
            cv2.imwrite(img_path, vehicle_img)
            plate_text = ''
            min_dist = float('inf')
            for pbox, plabel, ptext in zip(vehicles_boxes, vehicles_labels, vehicles_plate_texts):
                if plabel == 'LicensePlate':
                    v_cx = (x1 + x2) / 2
                    p_cx = (pbox[0] + pbox[2]) / 2
                    dist = abs(v_cx - p_cx)
                    if dist < min_dist and pbox[1] > y2:
                        min_dist = dist
                        plate_text = ptext
            log_entry = {
                'timestamp': timestamp.isoformat(),
                'image_file': img_name,  # Save filename in CSV
                'bbox_x1': x1, 'bbox_y1': y1,
                'bbox_x2': x2, 'bbox_y2': y2,
                'vehicle_type': label,
                'avg_color_hex': avg_hex,
                'license_plate': plate_text
            }
            write_log(log_entry)
            current_logged_boxes.append(bbox)

        previous_logged_boxes = current_logged_boxes

        # Draw bounding boxes and labels
        for bbox, label, plate_text in zip(vehicles_boxes, vehicles_labels, vehicles_plate_texts):
            x1, y1, x2, y2 = bbox
            color = (0, 255, 0) if label == 'Car' else (0, 128, 255) if label == 'Truck' else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = label
            if plate_text:
                text += f": {plate_text}"
            cv2.putText(frame, text, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# Routes and views --------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logs')
def logs():
    logs = read_logs()
    return render_template('logs.html', logs=logs)

if __name__ == '__main__':
    app.run(debug=True)
