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

        return n, img_orig

    def describe_positions(self, json_file=SETTINGS.logMoney_file):
        if not os.path.exists(json_file):
            return "Detection file not found."

        with open(json_file, 'r') as f:
            detections = json.load(f)

        if not detections:
            return "No banknotes were detected, try again."

        if len(detections) == 1:
            return f"{int(detections[0]['value'])} Bolivianos"

        # Calculate center points
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

        # Sort by vertical position (y), then horizontal (x)
        banknotes.sort(key=lambda b: (b["center"][1], b["center"][0]))

        # Calculate horizontal range to determine if vertically stacked
        xs = [b["center"][0] for b in banknotes]
        x_range = max(xs) - min(xs)

        is_vertical_stack = x_range < 50  # Adjust threshold if needed

        # Define vertical thresholds
        ys = [b["center"][1] for b in banknotes]
        y_min, y_max = min(ys), max(ys)
        y_th1 = y_min + (y_max - y_min) / 3
        y_th2 = y_min + 2 * (y_max - y_min) / 3

        def get_vertical_name(cy):
            if cy < y_th1:
                return "top"
            elif cy < y_th2:
                return "middle"
            else:
                return "bottom"

        def get_full_position_name(cx, cy):
            vertical = get_vertical_name(cy)
            # Define horizontal thresholds only if needed
            x_min, x_max = min(xs), max(xs)
            x_th1 = x_min + (x_max - x_min) / 3
            x_th2 = x_min + 2 * (x_max - x_min) / 3
            horizontal = (
                "left" if cx < x_th1 else
                "center" if cx < x_th2 else
                "right"
            )
            if vertical == "middle" and horizontal == "center":
                return "center"
            return f"{vertical} {horizontal}".strip()
        # Group banknotes by vertical zones
        zone_map = {"top": [], "middle": [], "bottom": []}
        for b in banknotes:
            cy = b["center"][1]
            zone = get_vertical_name(cy)
            zone_map[zone].append(b)

        # Build description with conditional horizontal detail
        descriptions = []
        for zone, items in zone_map.items():
            if len(items) == 1:
                # Only one item: just mention the vertical zone
                val = items[0]['value']
                descriptions.append(f"{val} Bolivianos on {zone}")
            else:
                # Multiple items: include full position (e.g., "top left", "top right")
                # Sort again horizontally
                x_min = min(b["center"][0] for b in items)
                x_max = max(b["center"][0] for b in items)
                x_th1 = x_min + (x_max - x_min) / 3
                x_th2 = x_min + 2 * (x_max - x_min) / 3

                for b in items:
                    cx = b["center"][0]
                    horizontal = (
                        "left" if cx < x_th1 else
                        "center" if cx < x_th2 else
                        "right"
                    )
                    pos = f"{zone} {horizontal}".strip()
                    descriptions.append(f"{b['value']} Bolivianos on {pos}")


        return ", ".join(descriptions)





class LLM:
  # genera un archivo de audio al cual podemos selecciona si queremos un audio en ingles o español
#   def gen_oudia(self,text, spanish=False):
#     aud_path = 'audio/audio.wav'

#     if spanish:
#         # Usar TTS para español
#         tts = TTS(SETTINGS.tts_spanish)
#         tts.tts_to_file(text=text, file_path=aud_path)
#     else:
#         # Usar Coqui TTS para inglés
#         tts = TTS(SETTINGS.tts_english)
#         tts.tts_to_file(text=text, file_path=aud_path)
#     return aud_path
  
  modelo=SETTINGS.llm
  #seleccionamos con qué dispositivo queremos ejecutar nuestro LLM, en mac: mps, y en Pcs con windows o linux que tengan una grafica nvidia, cambiar mps por cuda
  device = "cuda" if torch.backends.cuda.is_built() else "cpu"
  print(device)
  tokenizer = AutoTokenizer.from_pretrained(modelo)#.to(device)
  model = AutoModelForCausalLM.from_pretrained(
      modelo,
      # device_map="auto",
      # torch_dtype=torch.bfloat16,
  ).to(device)


#   def spanish_tr(self,input_text):


#     input_text = f"""
#     Translate the following text From English to spanish:
#     "{input_text}"
#     Translation: """


#     print(self.device)
#     print(input_text)
#     input_ids = self.tokenizer(input_text, return_tensors="pt").to("cuda")#.to("cuda")#.to("cpu")#
#     #model.to("cpu")#comment this if in colab
#     outputs = self.model.generate(**input_ids, max_new_tokens=150,
#                             do_sample=True,
#                             temperature=0.5,  # Make output less random
#                             top_p=0.8,        # Use nucleus sampling
#                             top_k=60 )
#     ans=self.tokenizer.decode(outputs[0])

#     ans=ans.split("Translation:")[1].strip()
#     ans=ans.split("\n")[0]
#     return ans
  


  def generate_response(self,spanish):
    if os.path.exists(SETTINGS.logMoney_file) and os.path.getsize(SETTINGS.logMoney_file) > 0:
          with open(SETTINGS.logMoney_file, "r") as file:
              try:
                  billetesData = json.load(file)
                  print(len(billetesData))
                  billetesData_str = json.dumps(billetesData, indent=4) 
              except json.JSONDecodeError:
                  print(json.JSONDecodeError)
    input_text = """
  Given a JSON list of banknotes with their values and bounding box coordinates, analyze the position coordinates of each banknote to determine its approximate location (e.g., "top left," "bottom right," "center"), relative to the other banknotes. After describing the locations of all banknotes, calculate the total sum and include it in the final sentence in the format: "So, in total, you have (total) Bolivianos." Use this format for each entry and make the description clear and concise. Here are a few examples to guide the model:

  Example 1:
  
  [
    {
        "value": 100.0,
        "position": [[300, 300], [600, 600]]
    }
  ]

  Response: The only banknote detected in the image is a 100 Bolivianos.

  Example 2:

  [
      {
          "value": 20.0,
          "position": [[173, 219], [601, 426]]
      },
      {
          "value": 20.0,
          "position": [[613, 221], [1025, 428]]
      },
      {
          "value": 10.0,
          "position": [[80, 100], [300, 300]]
      }
  ]
  Response: At the center left, there's a 20 Bolivianos banknote. Toward the center right, you’ll find another 20 Bolivianos banknote. In the top left, there’s a 10 Bolivianos banknote. So, in total, you have 50 Bolivianos.

  Example 3:

  [
      {
          "value": 100.0,
          "position": [[150, 450], [600, 800]]
      },
      {
          "value": 50.0,
          "position": [[650, 450], [1000, 800]]
      }
  ]
  Response: On the left, there’s a 100 Bolivianos banknote. Toward the right, you’ll find a 50 Bolivianos banknote. So, in total, you have 150 Bolivianos.

  Example 4:

  [
      {
          "value": 20.0,
          "position": [[300, 300], [450, 400]]
      },
      {
          "value": 10.0,
          "position": [[600, 50], [750, 200]]
      }
  ]
  Response: At the center, there’s a 20 Bolivianos banknote. Toward the top right, you’ll find a 10 Bolivianos banknote. So, in total, you have 30 Bolivianos.

  Example 5:

  [
      {
          "value": 50.0,
          "position": [[169, 433], [610, 665]]
      },
      {
          "value": 10.0,
          "position": [[600, 11], [1005, 221]]
      },
      {
          "value": 10.0,
          "position": [[176, 0], [604, 221]]
      },
      {
          "value": 50.0,
          "position": [[636, 452], [1046, 677]]
      },
      {
          "value": 20.0,
          "position": [[173, 219], [601, 426]]
      },
      {
          "value": 20.0,
          "position": [[613, 221], [1025, 428]]
      }
  ]
  Response: On the bottom left, there’s a 50 Bolivianos banknote. Toward the top right, you’ll find a 10 Bolivianos banknote. In the top left, there’s another 10 Bolivianos banknote. At the center right, there’s a 50 Bolivianos banknote. Toward the center left, there’s a 20 Bolivianos banknote. Finally, at the center right, you’ll find another 20 Bolivianos banknote. So, in total, you have 160 Bolivianos.

  Example 6: 

  [
    {
        "value": 5.0,
        "position": [[50, 50], [200, 150]]
    },
    {
        "value": 10.0,
        "position": [[400, 100], [600, 300]]
    },
    {
        "value": 50.0,
        "position": [[150, 500], [600, 800]]
    }
  ]

  Response: In the top left, there's a 5 Bolivianos banknote. Toward the top right, you’ll find a 10 Bolivianos banknote. At the bottom center, there’s a 50 Bolivianos banknote. So, in total, you have 65 Bolivianos.

  Example 7: 

  [
    {
        "value": 20.0,
        "position": [[200, 600], [500, 900]]
    },
    {
        "value": 10.0,
        "position": [[650, 200], [850, 400]]
    },
    {
        "value": 100.0,
        "position": [[300, 100], [800, 300]]
    },
    {
        "value": 5.0,
        "position": [[50, 50], [200, 100]]
    }
]

  Response: At the bottom center, there's a 20 Bolivianos banknote. Toward the center right, you’ll find a 10 Bolivianos banknote. In the top center, there’s a 100 Bolivianos banknote. Finally, at the very top left, there’s a 5 Bolivianos banknote. So, in total, you have 135 Bolivianos.
  
  Example 8:

  [
    {
        "value": 10.0,
        "position": [[150, 250], [400, 450]]
    },
    {
        "value": 20.0,
        "position": [[450, 250], [700, 450]]
    },
    {
        "value": 50.0,
        "position": [[200, 500], [600, 800]]
    }
  ]

  Response: On the center left, there’s a 10 Bolivianos banknote. Toward the center right, there’s a 20 Bolivianos banknote. At the bottom center, there’s a 50 Bolivianos banknote. So, in total, you have 80 Bolivianos.


  Example 9:"""+billetesData_str+"""

  Response:
  """



    input_ids = self.tokenizer(input_text, return_tensors="pt").to("cuda")#.to("cpu")#.to('cuda)

    outputs = self.model.generate(**input_ids, max_new_tokens=len(billetesData)*20,
                            temperature=0.5,  # Make output less random
                            top_p=0.8,        # Use nucleus sampling
                            top_k=60 )
    ans=self.tokenizer.decode(outputs[0])
    print(ans)
    if "Response:" in ans:
      generated_text = ans.split("Response:")[-1].strip()

    generated_text=re.split(r"<\|end_of_text\|>|Example|Question:", generated_text)[0].strip()
    print(generated_text)
    # if spanish:
    #   generated_text=self.spanish_tr(generated_text)


    return generated_text