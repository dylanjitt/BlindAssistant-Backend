from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
from io import BytesIO
import re

class LLM:
    def __init__(self, model_id="Qwen/Qwen2.5-VL-7B-Instruct"):
        print(f"Loading model: {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id,use_fast=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def describe_image(self, image_bytes: bytes, prompt: str) -> str:
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception:
            raise ValueError("Invalid image file.")

        # ✅ Insert <|image|> placeholder explicitly
        final_prompt = f"<|user|>\n<|image_1|>\n{prompt}\n<|end|>\n<|assistant|>\n"

        # 🧠 Pass prompt string and images as expected by Qwen processor
        inputs = self.processor(
            text=final_prompt,
            images=[image],
            return_tensors="pt",
            
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

        result = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return re.sub(r'\s+', ' ', result).strip()


