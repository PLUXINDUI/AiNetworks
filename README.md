# Классификация изображений: Cat / Frog / Deer

## Описание проекта

Проект по сравнению моделей классификации изображений и развертыванию лучшей модели в виде API с пользовательским интерфейсом.

**Датасет:** [Cifar10](https://www.kaggle.com/datasets/ayush1220/cifar10) — были выбраны три класса: cat, deer, frog (3000 изобаржений, по 1000 в каждом)

## Сравниваемые модели

| № | Модель | Практическая | Описание |
|---|--------|-------------|----------|
| 1 | Обычная ИНС (Dense) | ПР №2 | Полносвязная сеть, вход 64x64 (flatten) |
| 2 | Custom CNN | ПР №3 | Простая CNN: 2 свёрточных блока, вход 128x128 |
| 3 | VGG-like CNN | ПР №3 | CNN в стиле VGG: 4 блока свёрток, вход 128x128 |
| 4 | CNN + BN + Dropout | ПР №4 | CNN с BatchNormalization и Dropout, вход 128x128 |
| 5 | MobileNetV2 (Transfer Learning) | ПР №5 | Предобученная MobileNetV2, дообученная на датасете, вход 96x96 |

## Результаты сравнения

| Модель | Accuracy | Precision | Recall | F1-score |
|--------|----------|-----------|--------|----------|
| Обычная ИНС | 0.6000 | 0.6000 | 0.5900 | 0.5900 |
| Custom CNN | 0.6933 | 0.6900 | 0.6900 | 0.6900 |
| VGG-like CNN | 0.7133 | 0.7100 | 0.7100 | 0.7100 |
| CNN + BN + Dropout | ~0.73 | ~0.73 | ~0.73 | ~0.73 |
| **MobileNetV2 (Transfer Learning)** | **~0.95** | **~0.95** | **~0.95** | **~0.95** |

**Лучшая модель по F1-мере:** MobileNetV2 (Transfer Learning)

## Структура проекта

```
├── main.py                          # FastAPI бэкенд
├── app.py                           # Streamlit фронтенд
├── classification.keras  # Лучшая модель
├── requirements.txt                 # Зависимости фронтенда
├── requirements_backend.txt         # Зависимости бэкенда
└── README.md                        # Документация
```

## Ссылки

- **Публичный API:** https://networksai.onrender.com/predict
- **Streamlit-интерфейс:** https://classification-animals.streamlit.app

## Локальное развертывание

### Бэкенд (FastAPI)

```bash
pip install -r requirements_backend.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

API будет доступен по адресу `http://localhost:8000`.

### Фронтенд (Streamlit)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Примеры использования API

### Python

```python
import requests

url = "https://networksai.onrender.com/predict"
files = {"file": open("cat.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

### cURL

```bash
curl -X POST "https://networksai.onrender.com/predict" \
     -F "file=@cat.jpg"
```

### Пример ответа

```json
{
  "predicted_class": "cats",
  "confidence": 92.35,
  "probabilities": {
    "cats": 92.35,
    "dogs": 5.12,
    "snakes": 2.53
  }
}
```
