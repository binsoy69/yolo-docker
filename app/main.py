from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# Load model once at startup
model = YOLO("denom-cls-v2.pt") 

@app.get("/")
def root():
    return {"message": "YOLOv11 Classifier API is running."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image into memory
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Run prediction (classification)
        results = model(image)
        probs = results[0].probs  # classification confidence and classes

        if probs is None:
            return JSONResponse(status_code=400, content={"error": "Model did not return classification results."})

        top_class_idx = int(probs.top1)
        top_class_name = model.names[top_class_idx]
        confidence = float(probs.top1conf)

        return {
            "class": top_class_name,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
