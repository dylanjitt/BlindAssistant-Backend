from ultralytics import YOLO
import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import json,os,uuid
import matplotlib.pyplot as plt
import cv2
import numpy as np
import easyocr
import json
import torch

from src.config import get_settings

SETTINGS = get_settings()

Imgmodel = YOLO(SETTINGS.model_minibusSign)
# ✅ Initialize EasyOCR with GPU enabled (CUDA)
reader = easyocr.Reader(['es'], gpu=True)  # Make sure torch is using CUDA

def is_close(bbox1, bbox2, threshold=30):
    x1, y1, x2, y2 = bbox1
    x1_, y1_, x2_, y2_ = bbox2
    center1 = ((x1 + x2) / 2, (y1 + y2) / 2)
    center2 = ((x1_ + x2_) / 2, (y1_ + y2_) / 2)
    distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
    return distance < threshold

# Remove duplicate or overlapping boxes by confidence
def remove_duplicates(boxes, confidences):
    unique_boxes = []
    used_indices = set()

    for i in range(len(boxes)):
        if i in used_indices:
            continue
        for j in range(i + 1, len(boxes)):
            if is_close(boxes[i], boxes[j]):
                if confidences[i] >= confidences[j]:
                    used_indices.add(j)
                else:
                    used_indices.add(i)
        unique_boxes.append(boxes[i])

    return unique_boxes

class MiniBusSign:
    def showMinibusSign(self,img, n=None, output_json=SETTINGS.logSign_file):
        img_orig = cv2.imread(img) if isinstance(img, str) else img.copy()
        results = Imgmodel(img_orig)  # Your minibus sign detection model

        boxes = []
        confidences = []
        labels = []
        letreros_detectados = []

        # Step 1: Collect all detected boxes
        for result in results:
            for bbox in result.boxes:
                conf = bbox.conf[0].item()
                class_id = int(bbox.cls[0].item())
                label = result.names[class_id]
                if label.lower() != "letrero":
                    continue

                x1, y1, x2, y2 = map(int, bbox.xyxy[0])
                boxes.append((x1-n, y1-n, x2+n, y2+n))
                confidences.append(conf)
                labels.append(label)

        # Step 2: Remove duplicates
        unique_boxes = remove_duplicates(boxes, confidences)

        # Step 3: OCR using EasyOCR on each sign
        individual_text_found = False

        for idx, (x1, y1, x2, y2) in enumerate(unique_boxes):
            cropped_sign = img_orig[y1:y2, x1:x2]
            ocr_result = reader.readtext(cropped_sign, detail=0)
            text = " ".join(ocr_result).strip()

            if text:
                individual_text_found = True

            letrero_info = {
                "box": [x1, y1, x2, y2],
                "text": text
            }
            letreros_detectados.append(letrero_info)

            # Draw box and text
            cv2.rectangle(img_orig, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_orig, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Step 4: Group OCR if no individual detections worked
        if not individual_text_found and unique_boxes:
            x1_all = min([b[0] for b in unique_boxes])
            y1_all = min([b[1] for b in unique_boxes])
            x2_all = max([b[2] for b in unique_boxes])
            y2_all = max([b[3] for b in unique_boxes])

            full_area_crop = img_orig[y1_all:y2_all, x1_all:x2_all]
            group_ocr = reader.readtext(full_area_crop, detail=0)
            full_text = " ".join(group_ocr).strip()

            letreros_detectados.append({
                "group_box": [x1_all, y1_all, x2_all, y2_all],
                "group_text": full_text
            })

            cv2.rectangle(img_orig, (x1_all, y1_all), (x2_all, y2_all), (0, 255, 0), 2)
            cv2.putText(img_orig, "[GROUP OCR]", (x1_all, y1_all - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Save JSON
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(letreros_detectados, f, ensure_ascii=False, indent=2)

        # Show image inline
        img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 8))
        plt.imshow(img_rgb)
        plt.axis("off")
        plt.title("Minibus Signs Detected")
        plt.show()

        return letreros_detectados