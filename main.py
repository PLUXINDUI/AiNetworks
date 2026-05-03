from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

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

# Загрузка моделей
print("🔄 Загрузка моделей...")

# Модель 1: Классификация животных (Keras)
ANIMAL_MODEL_PATH = "classification.keras"
animal_model = tf.keras.models.load_model(ANIMAL_MODEL_PATH, compile=False)
print(f"✓ Модель животных загружена: {ANIMAL_MODEL_PATH}")

# Модель 2: MNIST цифры (TFLite)
MNIST_MODEL_PATH = "mnist_model.tflite"
mnist_interpreter = tf.lite.Interpreter(model_path=MNIST_MODEL_PATH)
mnist_interpreter.allocate_tensors()
mnist_input_details = mnist_interpreter.get_input_details()
mnist_output_details = mnist_interpreter.get_output_details()
print(f"✓ Модель MNIST загружена: {MNIST_MODEL_PATH}")

# Названия классов для животных (замени на свои!)
ANIMAL_CLASSES = ["cat", "dog", "bird", "horse"]  # Пример

def preprocess_animal_image(image: Image.Image):
    """Предобработка для модели животных"""
    # Приводим к RGB если нужно
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Ресайз под вход модели (измени размер под свою модель!)
    img_size = 224  # или 128, 150 и т.д. - какой у тебя?
    image = image.resize((img_size, img_size))
    
    # В массив и нормализация
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, 0)  # Добавляем batch dimension
    
    return img_array

def preprocess_mnist_image(image: Image.Image):
    """Предобработка для MNIST модели"""
    # Конвертируем в оттенки серого
    if image.mode != 'L':
        image = image.convert('L')
    
    # Ресайз до 28x28
    image = image.resize((28, 28), Image.LANCZOS)
    
    # В массив и инвертируем (если нужно)
    img_array = np.array(image)
    
    # Нормализация
    img_array = img_array.astype('float32') / 255.0
    
    # Добавляем канал и batch dimension
    img_array = np.expand_dims(img_array, -1)
    img_array = np.expand_dims(img_array, 0)
    
    return img_array

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
        
        # Предобработка
        processed_image = preprocess_animal_image(image)
        
        # Предсказание
        predictions = animal_model.predict(processed_image, verbose=0)
        predicted_class_idx = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        # Получаем название класса
        if predicted_class_idx < len(ANIMAL_CLASSES):
            predicted_class = ANIMAL_CLASSES[predicted_class_idx]
        else:
            predicted_class = f"class_{predicted_class_idx}"
        
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
        
        # Предобработка
        processed_image = preprocess_mnist_image(image)
        
        # Предсказание через TFLite
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
    """Список классов животных"""
    return {
        "classes": ANIMAL_CLASSES,
        "count": len(ANIMAL_CLASSES)
    }

@app.get("/classes/mnist")
async def get_mnist_classes():
    """Список классов MNIST"""
    return {
        "classes": list(range(10)),
        "count": 10,
        "description": "Digits 0-9"
    }