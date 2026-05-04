from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import tensorflow as tf

app = FastAPI(title="Image Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_NAMES = ["cat", "frog", "deer"]

model = tf.keras.models.load_model("classification.keras")
INPUT_SHAPE = model.input_shape[1:3]


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(INPUT_SHAPE)
    arr = np.array(img, dtype="float32") / 255.0
    if len(model.input_shape) == 2:
        arr = arr.flatten()
    return np.expand_dims(arr, axis=0)


@app.get("/")
def root():
    return {"status": "ok", "model_input": str(INPUT_SHAPE), "classes": CLASS_NAMES}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    x = preprocess_image(image_bytes)
    probs = model.predict(x, verbose=0)[0].tolist()
    pred_idx = int(np.argmax(probs))
    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence": round(probs[pred_idx] * 100, 2),
        "probabilities": {cls: round(p * 100, 2) for cls, p in zip(CLASS_NAMES, probs)},
    }
