import cv2
import numpy as np
import os
import csv
from datetime import datetime
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pytesseract

# Configure pytesseract path if needed, uncomment and modify:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def average_color(image):
    """Compute average color in BGR format for the image."""
    if image.size == 0:
        return (0, 0, 0)
    avg_color_per_row = np.average(image, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return tuple(map(int, avg_color))  # (B,G,R)

def bgr_to_hex(color_tuple):
    return '#{:02x}{:02x}{:02x}'.format(color_tuple[2], color_tuple[1], color_tuple[0])

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def write_log(csv_file, data, header=None):
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists and header:
            writer.writeheader()
        writer.writerow(data)

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

def estimate_parking_spots(vehicles_boxes, min_gap_pixels=60):
    if not vehicles_boxes:
        return []
    vehicles_boxes = sorted(vehicles_boxes, key=lambda b: b[0])
    parking_spots = []
    for i in range(len(vehicles_boxes) - 1):
        gap = vehicles_boxes[i+1][0] - vehicles_boxes[i][2]
        if gap >= min_gap_pixels:
            parking_spots.append((vehicles_boxes[i][2], vehicles_boxes[i+1][0]))
    return parking_spots

def ocr_license_plate(plate_img):
    # Convert to grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    # Optional: thresholding or denoising for better OCR
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # OCR config to recognize alphanumeric only
    config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(thresh, config=config)
    text = text.strip().replace('\n', '').replace('\r', '')
    return text

def main():
    model = YOLO('yolo12m.pt')  # Your model weights with cars, trucks, plates classes

    cap = cv2.VideoCapture(0)  # Use webcam or replace with video file path

    ensure_dir('parked_vehicles')
    csv_log_file = 'parking_log.csv'
    csv_header = ['timestamp', 'image_file', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'vehicle_type', 'avg_color_hex', 'license_plate']

    previous_logged_boxes = []

    plt.ion()
    fig, ax = plt.subplots()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or failed")
            break

        results = model(frame)
        detections = results[0]

        vehicles_boxes = []
        vehicles_labels = []
        vehicles_plate_texts = []
        current_logged_boxes = []

        # Collect detected vehicles and license plates info
        for det in detections.boxes:
            cls_id = int(det.cls[0])
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            label = None

            # Classes: 2=car,7=truck,8=license_plate (assuming 8 is plate)
            if cls_id == 2:
                label = 'Car'
            elif cls_id == 7:
                label = 'Truck'
            elif cls_id == 8:
                # Crop license plate and OCR it
                plate_img = frame[y1:y2, x1:x2]
                plate_text = ocr_license_plate(plate_img)
                # Store plate with bbox for association
                vehicles_boxes.append((x1, y1, x2, y2))
                vehicles_labels.append('LicensePlate')
                vehicles_plate_texts.append(plate_text)
                continue
            else:
                continue  # skip other classes

            vehicles_boxes.append((x1, y1, x2, y2))
            vehicles_labels.append(label)
            vehicles_plate_texts.append('')  # empty for normal vehicles

        # For vehicle logging: match plates to nearest vehicle bbox horizontally
        for i, (bbox, label) in enumerate(zip(vehicles_boxes, vehicles_labels)):
            # Only log vehicles (Car, Truck)
            if label not in ['Car', 'Truck']:
                continue

            # Skip duplicate logs
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
            img_path = os.path.join('parked_vehicles', img_name)
            cv2.imwrite(img_path, vehicle_img)

            # Find license plate text for this vehicle by horizontal proximity
            plate_text = ''
            min_dist = float('inf')
            for pbox, p_label, ptext in zip(vehicles_boxes, vehicles_labels, vehicles_plate_texts):
                if p_label == 'LicensePlate':
                    # horizontal distance between centers
                    v_cx = (x1 + x2) / 2
                    p_cx = (pbox[0] + pbox[2]) / 2
                    dist = abs(v_cx - p_cx)
                    if dist < min_dist and pbox[1] > y2:  # plate below vehicle bbox roughly
                        min_dist = dist
                        plate_text = ptext

            log_entry = {
                'timestamp': timestamp.isoformat(),
                'image_file': img_path,
                'bbox_x1': x1,
                'bbox_y1': y1,
                'bbox_x2': x2,
                'bbox_y2': y2,
                'vehicle_type': label,
                'avg_color_hex': avg_hex,
                'license_plate': plate_text
            }
            write_log(csv_log_file, log_entry, header=csv_header)
            print(f"Logged {label} at {timestamp_str}, plate: {plate_text}, color: {avg_hex}, saved: {img_path}")

            current_logged_boxes.append(bbox)

        previous_logged_boxes = current_logged_boxes

        # Draw bounding boxes and labels
        for bbox, label in zip(vehicles_boxes, vehicles_labels):
            x1, y1, x2, y2 = bbox
            if label == 'Car':
                color = (0, 255, 0)  # green
            elif label == 'Truck':
                color = (0, 128, 255)  # orange
            else:  # license plate
                color = (255, 0, 0)  # blue
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Compute baseline and median height for parking spots
        vehicle_only_boxes = [bbox for bbox, lbl in zip(vehicles_boxes, vehicles_labels) if lbl in ['Car', 'Truck']]
        if vehicle_only_boxes:
            bottoms = [box[3] for box in vehicle_only_boxes]
            baseline = int(np.median(bottoms))
            heights = [box[3] - box[1] for box in vehicle_only_boxes]
            median_height = int(np.median(heights)) if heights else 40
        else:
            baseline = frame.shape[0] - 50
            median_height = 40

        # Draw parking spots
        parking_spots = estimate_parking_spots(vehicle_only_boxes)
        for left_x, right_x in parking_spots:
            box_height = int(median_height * 0.6)
            y2 = baseline + 10
            y1 = y2 - box_height
            color = (0, 255, 255)  # cyan
            cv2.rectangle(frame, (left_x, y1), (right_x, y2), color, 2)
            cv2.putText(frame, 'Empty Spot', (left_x, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Show frame
        ax.clear()
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        plt.pause(0.001)

        if plt.get_fignums() == []:
            break

    cap.release()
    plt.close()

if __name__ == '__main__':
    main()
