from ultralytics import YOLO
import cv2
import easyocr
import matplotlib.pyplot as plt
import re
from difflib import SequenceMatcher
import numpy as np
from src.models import Billete
import json, os, uuid
from src.config import get_settings

SETTINGS = get_settings()

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load YOLO model 
Imgmodel = YOLO(SETTINGS.model_moneyDet)  # Path to your YOLO model

# Function to clean and process strings
def clean_string(s):
    return re.sub(r'[{}\[\]()]', '', s)

def is_similar(word, target_word, threshold=0.5):
    ratio = SequenceMatcher(None, word, target_word).ratio()
    return ratio > threshold

def addBs(lista, n):
    val = 0.0
    values = {'1': 1.0, '2': 2.0, '5': 5.0, '10': 10.0, '20': 20.0, '50': 50.0, '100': 100.0, '200': 200.0}
    ctvsMode = False
    for item in lista:
        words = item.split(' ')
        for word in words:
            if word in values and not ctvsMode:
                val = values[word]
            elif word == 'CENTAVOS':
                val /= 100
                ctvsMode = True
            elif is_similar(word, 'BOLIVIANO') and word.endswith('O'):
                val = 1.0
    return val

# Helper function to check if two bounding boxes are close
def is_close(bbox1, bbox2, threshold=30):
    x1, y1, x2, y2 = bbox1
    x1_, y1_, x2_, y2_ = bbox2
    # Calculate the center points of both bounding boxes
    center1 = ((x1 + x2) / 2, (y1 + y2) / 2)
    center2 = ((x1_ + x2_) / 2, (y1_ + y2_) / 2)
    # Calculate Euclidean distance between the centers
    distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
    return distance < threshold

# Function to remove duplicate or close bounding boxes
def remove_duplicates(boxes, confidences):
    unique_boxes = []
    used_indices = set()
    
    for i in range(len(boxes)):
        if i in used_indices:
            continue
        for j in range(i + 1, len(boxes)):
            if is_close(boxes[i], boxes[j]):
                # Keep the box with the higher confidence
                if confidences[i] >= confidences[j]:
                    used_indices.add(j)
                else:
                    used_indices.add(i)
        unique_boxes.append(boxes[i])

    return unique_boxes

# Main function to analyze image
class BilleteDetector:
    def showImg(self, img, output_json=SETTINGS.logMoney_file):
        n = 0
        img_orig = cv2.imread(img) if isinstance(img, str) else img.copy()
        
        results = Imgmodel(img_orig)
        reader = easyocr.Reader(['es'], gpu=True)
        blue_shade = 255

        boxes = []
        confidences = []
        labels = []
        detecciones = []  # Lista para almacenar las detecciones de billetes

        for result in results:
            for bbox in result.boxes:
                conf = bbox.conf[0].item()
                class_id = int(bbox.cls[0].item())
                x1, y1, x2, y2 = map(int, bbox.xyxy[0])

                boxes.append((x1, y1, x2, y2))
                confidences.append(conf)
                labels.append(result.names[class_id])

        unique_boxes = remove_duplicates(boxes, confidences)

        for i, (x1, y1, x2, y2) in enumerate(unique_boxes):
            conf = confidences[i]
            label = labels[i]

            if conf >= 0.6:
                valueDetected = int(label.split('-')[0])  # Extrae el valor del nombre de la clase
                n += valueDetected
                cv2.rectangle(img_orig, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(img_orig, f"{label} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Crear un objeto Billete y añadirlo a la lista de detecciones
                deteccion = Billete(value=valueDetected, position=[[x1, y1], [x2, y2]])
                detecciones.append(deteccion.dict())  # Convertir a dict para JSON
                
            else:
                img_crop = img_orig[y1:y2, x1:x2]
                result_ocr = reader.readtext(img_crop)
                words = []
                cv2.rectangle(img_orig, (x1, y1), (x2, y2), (blue_shade, 0, 0), 2)
                
                for r in result_ocr:
                    bbox_ocr, text, score = r
                    if text in {'1', '2', '5', '10', '20', '50', '100', '200'} or is_similar(clean_string(text), 'BOLIVIANOS') or is_similar(clean_string(text), 'CENTAVOS'):
                        x1_ocr, y1_ocr = int(bbox_ocr[0][0] + x1), int(bbox_ocr[0][1] + y1)
                        x2_ocr, y2_ocr = int(bbox_ocr[2][0] + x1), int(bbox_ocr[2][1] + y1)
                        cv2.rectangle(img_orig, (x1_ocr, y1_ocr), (x2_ocr, y2_ocr), (blue_shade, 0, 0), 2)
                        cv2.putText(img_orig, text, (x1_ocr, y1_ocr - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (blue_shade, 0, 0), 2)
                        words.append(text)

                if words:
                    ocr_value = addBs(words, n)
                    n += ocr_value
                    deteccion = Billete(value=ocr_value, position=[[x1, y1], [x2, y2]])
                    detecciones.append(deteccion.dict())

                blue_shade = max(0, blue_shade - 15)

        img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.show()

        # Guardar las detecciones en un archivo JSON
        with open(output_json, 'w') as f:
            json.dump(detecciones, f, indent=4)
        
        self.clear_gpu_cache()

        return n, img_orig

    def clear_gpu_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU cache cleared.")
        else:
            print("CUDA not available, nothing to clear.")


    def describe_positions(self, json_file=SETTINGS.logMoney_file):
        if not os.path.exists(json_file):
            return "Detection file not found."

        with open(json_file, 'r') as f:
            detections = json.load(f)

        if not detections:
            return "No banknotes were detected, try again."

        if len(detections) == 1:
            return f"{int(detections[0]['value'])} Bolivianos"

        # Get center points of banknotes
        banknotes = []
        for det in detections:
            x1, y1 = det["position"][0]
            x2, y2 = det["position"][1]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            banknotes.append({
                "value": int(det["value"]),
                "center": (cx, cy)
            })

        # Sort by vertical position first, then horizontal
        banknotes.sort(key=lambda b: (b["center"][1], b["center"][0]))

        # Determine if arrangement is primarily vertical or horizontal
        xs = [b["center"][0] for b in banknotes]
        ys = [b["center"][1] for b in banknotes]
        
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        
        # Check if arrangement forms a grid (multiple rows and columns)
        is_grid = False
        if len(banknotes) > 3:
            # Count distinct x and y positions to detect grid pattern
            distinct_x = len({round(x, -2) for x in xs})  # rounded to handle small variations
            distinct_y = len({round(y, -2) for y in ys})
            is_grid = distinct_x > 1 and distinct_y > 1

        descriptions = []
        
        if is_grid:
            # Handle grid layout (multiple columns and rows)
            # Group by columns first (based on x position)
            x_sorted = sorted(banknotes, key=lambda b: b["center"][0])
            
            # Find natural column breaks
            x_positions = [b["center"][0] for b in x_sorted]
            x_diff = [x_positions[i+1] - x_positions[i] for i in range(len(x_positions)-1)]
            avg_x_diff = sum(x_diff) / len(x_diff)
            
            columns = []
            current_column = [x_sorted[0]]
            
            for i in range(1, len(x_sorted)):
                if x_diff[i-1] > avg_x_diff * 1.5:  # Significant gap indicates new column
                    columns.append(current_column)
                    current_column = [x_sorted[i]]
                else:
                    current_column.append(x_sorted[i])
            columns.append(current_column)
            
            # Describe each column's notes
            for col_idx, column in enumerate(columns):
                column.sort(key=lambda b: b["center"][1])  # Sort vertically within column
                
                # Determine column position (left/right or left/center/right)
                col_positions = ["left", "right"] if len(columns) == 2 else ["left", "center", "right"]
                col_pos = col_positions[col_idx] if col_idx < len(col_positions) else f"column {col_idx+1}"
                
                # Describe vertical positions in this column
                if len(column) == 1:
                    descriptions.append(f"{column[0]['value']} Bolivianos at {col_pos}")
                else:
                    vert_positions = ["top", "bottom"] if len(column) == 2 else ["top", "middle", "bottom"]
                    for i, note in enumerate(column):
                        if i < len(vert_positions):
                            pos = f"{vert_positions[i]} {col_pos}"
                        else:
                            pos = f"position {i+1} in {col_pos}"
                        descriptions.append(f"{note['value']} Bolivianos at {pos}")
        else:
            # Handle single column/row arrangements
            if y_range > x_range:  # Primarily vertical arrangement
                if len(banknotes) == 2:
                    positions = ["top", "bottom"]
                else:  # 3 or more vertically
                    positions = ["top", "middle", "bottom"] if len(banknotes) == 3 else [f"position {i+1}" for i in range(len(banknotes))]
                
                for i, note in enumerate(banknotes):
                    if i < len(positions):
                        pos = positions[i]
                    else:
                        pos = f"position {i+1}"
                    descriptions.append(f"{note['value']} Bolivianos at {pos}")
            else:  # Primarily horizontal arrangement
                if len(banknotes) == 2:
                    positions = ["left", "right"]
                else:  # 3 or more horizontally
                    positions = ["left", "center", "right"] if len(banknotes) == 3 else [f"position {i+1}" for i in range(len(banknotes))]
                
                for i, note in enumerate(banknotes):
                    if i < len(positions):
                        pos = positions[i]
                    else:
                        pos = f"position {i+1}"
                    descriptions.append(f"{note['value']} Bolivianos at {pos}")

        return ", ".join(descriptions)

