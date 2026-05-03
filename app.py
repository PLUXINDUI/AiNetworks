import streamlit as st
import requests
from PIL import Image
import io
import numpy as np

st.set_page_config(page_title="Multi-Model Classifier", layout="wide")
st.title(" Мульти-модельный классификатор")

# Выбор модели
model_choice = st.radio(
    "Выберите модель:",
    ["🐾 Классификация животных", "🔢 Распознавание цифр (MNIST)"],
    horizontal=True
)

# URL API
API_URL = st.text_input(
    "URL API:",
    value="http://localhost:8000"
)

# Определяем эндпоинт
if "животных" in model_choice:
    endpoint = "/predict/animals"
    st.info("📸 Загрузите изображение животного (кошка, собака и т.д.)")
else:
    endpoint = "/predict/mnist"
    st.info("✏️ Загрузите изображение цифры (0-9) или нарисуйте её")

# Загрузка файла
uploaded_file = st.file_uploader(
    "Выберите изображение",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    col1, col2 = st.columns(2)
    
    with col1:
        original = Image.open(uploaded_file)
        st.image(original, caption="Оригинал", use_container_width=True)
    
    # Кнопка классификации
    if st.button("🔮 Классифицировать", type="primary"):
        with st.spinner("Обработка..."):
            try:
                # Подготовка файла
                img_byte_arr = io.BytesIO()
                original.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # Отправка запроса
                files = {"file": ("image.png", img_byte_arr, "image/png")}
                response = requests.post(
                    f"{API_URL}{endpoint}",
                    files=files,
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()
                
                if result.get("status") == "success":
                    with col2:
                        st.success(f"✅ Предсказание: **{result['predicted_class']}**")
                        st.metric("Уверенность", f"{result['confidence']:.2%}")
                    
                    # График вероятностей
                    st.subheader("📊 Вероятности по классам")
                    probs = result['all_probabilities']
                    
                    if "animals" in result.get("model", ""):
                        classes = result.get("classes", [f"Class {i}" for i in range(len(probs))])
                    else:
                        classes = list(range(len(probs)))
                    
                    probs_df = st.dataframe(
                        {"Класс": classes, "Вероятность": [f"{p:.2%}" for p in probs]},
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    # Бар-чарт
                    chart_data = {"Вероятность": probs}
                    st.bar_chart(chart_data)
                    
                else:
                    st.error("❌ Ошибка в ответе API")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Ошибка соединения: {e}")
            except Exception as e:
                st.error(f"❌ Ошибка: {e}")

# Боковая панель с информацией
with st.sidebar:
    st.header("ℹ️ Информация")
    st.write(f"**API URL:** {API_URL}")
    st.write(f"**Модель:** {endpoint}")
    
    if st.button("📡 Проверить API"):
        try:
            response = requests.get(f"{API_URL}/", timeout=5)
            if response.status_code == 200:
                st.success("✅ API доступен!")
                st.json(response.json())
            else:
                st.error(f"❌ Статус: {response.status_code}")
        except Exception as e:
            st.error(f"❌ Ошибка: {e}")