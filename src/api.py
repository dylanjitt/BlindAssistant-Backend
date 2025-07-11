from fastapi import FastAPI, Depends, Query,UploadFile, Form, File, HTTPException, status, Depends
#from llama_index.core.agent import ReActAgent
from src.config import get_settings
from src.moneyDetector import BilleteDetector
from src.minibusSignDetector import MiniBusSign
from src.llm import LLM
from functools import cache
from fastapi.responses import Response,JSONResponse,FileResponse,StreamingResponse
import io,os
from PIL import Image, UnidentifiedImageError
import numpy as np
import cv2
import ollama
import base64
import requests

SETTINGS = get_settings()

app = FastAPI(title=SETTINGS.api_name, version=SETTINGS.revision)


@cache
def get_bill_detector() -> BilleteDetector:
    print("Creating model...")
    return BilleteDetector()

# @cache
# def get_llm()-> LLM:
#   print('getting LLM')
#   return LLM()


@cache
def get_minibusSignDetector()-> MiniBusSign:
    print("creating SingFinderModel...")
    return MiniBusSign()

@app.post("/BolivianMoneyDetector")
def detect_and_summarize(
    file: UploadFile = File(...),
    detector: BilleteDetector = Depends(get_bill_detector),
    spanish: bool = False
) -> JSONResponse:
    img_stream = io.BytesIO(file.file.read())
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Not an image"
        )

    try:
        img_obj = Image.open(img_stream)
        img_np = np.array(img_obj)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Image format not supported"
        )

    total, img_det = detector.showImg(img_bgr)
    generated_text = detector.describe_positions(spanish)

    return JSONResponse(content={"description": generated_text})

@app.post("/minibusSignDetector")
def detect_minibus_sign(
    file: UploadFile = File(...),
    sign_model: MiniBusSign = Depends(get_minibusSignDetector)
) -> JSONResponse:
    img_stream = io.BytesIO(file.file.read())
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Not an image"
        )

    try:
        img_obj = Image.open(img_stream)
        img_np = np.array(img_obj)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Image format not supported"
        )

    result = sign_model.showMinibusSign(img_bgr)
    print(result)
    return JSONResponse(content={"description":result})

@app.post("/ollamaVision")
async def ollama_vision_endpoint(
     prompt: str = "Describe the following image in detail, the elements on it like if you were to tell a blind person, If the image has a person, describe looks, hair, skin tone and clothing, if it is an objects (or various objects) describe them, what they are or where are those from, their colors, and if on the image you find any form of text, read it in a separate paragraph, look for text specifically in spanish, the response should be straigntforward and IN SPANISH, en español",
     file: UploadFile = File(...)
 ):
     if not file.content_type.startswith("image/"):
         raise HTTPException(status_code=400, detail="Only image files are allowed.")

     # Read and encode the image
     img_bytes = await file.read()
     img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    

    # Use ollama.generate for vision tasks
     response = ollama.generate(
         model=SETTINGS.llm,
         prompt=prompt,
         images=[img_base64],
         keep_alive="10h",
         stream=False,
         options={
        "temperature": 0.7,
        "top_p": 0.9,
        "num_predict": 300
        },
     )
     print(response['response'])

     return {"description": response['response']}
    

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.app:app", port=8080, host="0.0.0.0")