from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# Для Keras 3
os.environ['KERAS_BACKEND'] = 'tensorflow'

app = FastAPI(title="Multi-Model Classification API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# ЗАГРУЗКА МОДЕЛЕЙ (с обработкой ошибок)
# ============================================

print("🔄 Загрузка моделей...")

# Модель 1: Классификация животных (Keras)
ANIMAL_MODEL_PATH = "animal_model_savedmodel"  # Папка, не файл!

try:
    print(f"🔄 Загрузка модели из {ANIMAL_MODEL_PATH}...")
    animal_model = tf.keras.models.load_model(ANIMAL_MODEL_PATH, compile=False)
    print("✅ Модель животных загружена!")
except Exception as e:
    print(f"❌ Ошибка: {e}")
    raise

# Модель 2: MNIST цифры (TFLite)
MNIST_MODEL_PATH = "mnist_model.tflite"

try:
    print(f"🔄 Загрузка модели MNIST из {MNIST_MODEL_PATH}...")
    mnist_interpreter = tf.lite.Interpreter(model_path=MNIST_MODEL_PATH)
    mnist_interpreter.allocate_tensors()
    mnist_input_details = mnist_interpreter.get_input_details()
    mnist_output_details = mnist_interpreter.get_output_details()
    print(f"✅ Модель MNIST успешно загружена!")
except FileNotFoundError:
    print(f"❌ ОШИБКА: Файл {MNIST_MODEL_PATH} не найден!")
    raise
except Exception as e:
    print(f"❌ ОШИБКА при загрузке MNIST модели: {e}")
    raise

print("🎉 Все модели загружены успешно!")

# ============================================
# ФУНКЦИИ ПРЕДОБРАБОТКИ
# ============================================

def preprocess_animal_image(image: Image.Image):
    """Предобработка для модели животных"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_size = 224  # Измени под размер своей модели!
    image = image.resize((img_size, img_size))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, 0)
    
    return img_array

def preprocess_mnist_image(image: Image.Image):
    """Предобработка для MNIST модели"""
    if image.mode != 'L':
        image = image.convert('L')
    
    image = image.resize((28, 28), Image.LANCZOS)
    img_array = np.array(image)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, -1)
    img_array = np.expand_dims(img_array, 0)
    
    return img_array

# ============================================
# ЭНДПОИНТЫ API
# ============================================

@app.get("/")
async def root():
    return {
        "message": "Multi-Model Classification API",
        "available_models": ["animals", "mnist"],
        "endpoints": {
            "/predict/animals": "Классификация животных",
            "/predict/mnist": "Классификация цифр MNIST",
            "/docs": "API документация"
        }
    }

@app.post("/predict/animals")
async def predict_animal(file: UploadFile = File(...)):
    """Классификация животных"""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        processed_image = preprocess_animal_image(image)
        predictions = animal_model.predict(processed_image, verbose=0)
        predicted_class_idx = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        ANIMAL_CLASSES = ["cat", "dog", "bird", "horse"]  # ЗАМЕНИ НА СВОИ!
        predicted_class = ANIMAL_CLASSES[predicted_class_idx] if predicted_class_idx < len(ANIMAL_CLASSES) else f"class_{predicted_class_idx}"
        
        return JSONResponse(content={
            "model": "animals",
            "predicted_class": predicted_class,
            "class_index": predicted_class_idx,
            "confidence": round(confidence, 4),
            "all_probabilities": [float(p) for p in predictions[0]],
            "status": "success"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")

@app.post("/predict/mnist")
async def predict_mnist(file: UploadFile = File(...)):
    """Классификация цифр MNIST"""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        processed_image = preprocess_mnist_image(image)
        
        mnist_interpreter.set_tensor(mnist_input_details[0]['index'], processed_image)
        mnist_interpreter.invoke()
        predictions = mnist_interpreter.get_tensor(mnist_output_details[0]['index'])
        
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        return JSONResponse(content={
            "model": "mnist",
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "all_probabilities": [float(p) for p in predictions[0]],
            "status": "success"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")

@app.get("/classes/animals")
async def get_animal_classes():
    return {"classes": ["cat", "frog", "deer"]}  # ЗАМЕНИ НА СВОИ!

@app.get("/classes/mnist")
async def get_mnist_classes():
    return {"classes": list(range(10)), "count": 10}