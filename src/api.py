from fastapi import FastAPI, Depends, Query,UploadFile, File, HTTPException, status, Depends
#from llama_index.core.agent import ReActAgent
from src.config import get_settings
from src.moneyDetector import BilleteDetector,LLM#, gen_oudia
from src.minibusSignDetector import MiniBusSign
from functools import cache
from fastapi.responses import Response,JSONResponse,FileResponse
import io,os
from PIL import Image, UnidentifiedImageError
import numpy as np
import cv2

SETTINGS = get_settings()

app = FastAPI(title=SETTINGS.api_name, version=SETTINGS.revision)


@cache
def get_bill_detector() -> BilleteDetector:
    print("Creating model...")
    return BilleteDetector()

@cache
def get_llm()-> LLM:
  print('getting LLM')
  return LLM()

@cache
def get_minibusSignDetector()-> MiniBusSign:
    print("creating SingFinderModel...")
    return MiniBusSign()

@app.post("/BolivianMoneyDetector")
def detect_and_summarize(
    file: UploadFile = File(...),
    detector: BilleteDetector = Depends(get_bill_detector),
    llm: LLM = Depends(get_llm),
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
    generated_text = llm.generate_response(spanish=spanish)

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

    result = sign_model.showMinibusSign(img_bgr,n=10)
    return JSONResponse(content=result)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.app:app", port=8080, host="0.0.0.0")