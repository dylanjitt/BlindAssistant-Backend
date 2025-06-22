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

from difflib import SequenceMatcher

KNOWN_TERMS = [
    "OBRAJES", "CALACOTO", "SAN MIGUEL", "LOS PINOS", "ACHUMANI","STADIUM","PZA TRIANGULAR"
    "IRPAVI", "LAS LOMAS", "COMPLEJO","LAS LOMAS","ROSALES", "ARCE", "PRADO", "PEREZ",
    "MIRAFLORES", "V FATIMA", "H OBRERO", "MONTES", "PANDO","AV BUSCH","CEMENTERIO",
    "COTA COTA", "CHASQUIPAMPA", "MUNAYPATA", "CEJA", "LA FLORIDA","VITA","UMSA","U.M.S.A"
    "MALLASA", "MALLASILLA", "ACHOCALLA", "EST MAYOR","SAN PEDRO","PEDRO","OBELISCO",
    "COLOMBIA","PEREZ V","RODRIGUEZ","TELEFERICO","PEDREGAL","ALMENDROS","MEGA CENTER","IRPAVI 2","BAJO IRPAVI"
    "BAJO","ALTO","P ESTUDIANTE","EST CENTRAL","TERMINAL"
    "SEGUENCOMA","FONDO","CALLE 16","6 AGOSTO","6 DE AGOSTO"
]

def get_best_match(word: str, candidates=KNOWN_TERMS) -> str:
    """
    Compare a word against a list of candidate terms using SequenceMatcher
    and return the most similar known term.
    """
    word_up = word.upper()
    best_term = None
    best_score = 0.0
    for term in candidates:
        score = SequenceMatcher(None, word_up, term).ratio()
        if score > best_score:
            best_score = score
            best_term = term
    return best_term if best_term is not None else word


def readerImg(img):
    reader = easyocr.Reader(['es'], gpu=True)
    detected = reader.readtext(img, detail=0)

    # 2. Match each word to the known list
    matched = []
    for word in detected:
        match = get_best_match(word)
        matched.append(match)
    #return matched
    return " ".join(matched).strip()

class MiniBusSign:
    
    #NEW FUNCTION
    def showMinibusSign(self,img, offset=10):
        img_orig = cv2.imread(img) if isinstance(img, str) else img.copy()
        img_with_boxes = img_orig.copy()
        results = Imgmodel(img_orig)
        boxes = []
        confidences = []
        labels = []

        if len(results)==0:
            return "no se encontraron letreros, intente nuevamente"
        # Gather all detection results for "letrero"
        for result in results:
            for bbox in result.boxes:
                conf = bbox.conf[0].item()
                class_id = int(bbox.cls[0].item())
                label = result.names[class_id]
                if label.lower() != "letrero":
                    continue

                x1, y1, x2, y2 = map(int, bbox.xyxy[0])
                boxes.append((x1, y1, x2, y2))
                confidences.append(conf)

        # Remove duplicates based on distance threshold
        unique_boxes = remove_duplicates(boxes, confidences)

        # Draw bounding boxes on a copy of the original image
        for idx, (x1, y1, x2, y2) in enumerate(unique_boxes):
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_with_boxes, f"#{idx + 1}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Process and display each cropped/resized sign
        for idx, (x1, y1, x2, y2) in enumerate(unique_boxes):
            cropped_sign = img_orig[y1:y2, x1:x2]
            target_w, target_h = x2 - x1, y2 - y1
            resized_sign=cropped_sign

            signText= readerImg(resized_sign)
            labels.append(signText)
        
        if len(labels)==0:
            return "no se encontraron letreros, intente nuevamente"

        self.clear_gpu_cache() 
        #print(labels)   
            
        return " ".join(labels).strip()
    
    def clear_gpu_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU cache cleared.")
        else:
            print("CUDA not available, nothing to clear.")