import streamlit as st
import requests
from PIL import Image
import io
import numpy as np
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Multi-Model Classifier", layout="wide")
st.title("🤖 Мульти-модельный классификатор")

# Выбор модели
model_choice = st.radio(
    "Выберите задачу:",
    ["🐾 Классификация животных", " Распознавание цифр (MNIST)"],
    horizontal=True
)

# URL API
API_URL = st.text_input("URL вашего API (Render):", value="http://localhost:8000")

# Определяем эндпоинт
endpoint = "/predict/animal" if "животных" in model_choice else "/predict/digit"

# ------------------------------------------
# 🐾 РЕЖИМ: ЖИВОТНЫЕ (загрузка файла)
# ------------------------------------------
if "животных" in model_choice:
    st.info("📸 Загрузите фотографию животного (кошка, лягушка, олень)")
    uploaded_file = st.file_uploader("Выберите изображение", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        with col1:
            original = Image.open(uploaded_file)
            st.image(original, caption="Оригинал", use_container_width=True)
            
        if st.button("🔮 Классифицировать животное", type="primary"):
            with st.spinner("Анализирую..."):
                try:
                    img_byte_arr = io.BytesIO()
                    original.save(img_byte_arr, format='PNG')
                    files = {"file": ("animal.png", img_byte_arr.getvalue(), "image/png")}
                    response = requests.post(f"{API_URL}{endpoint}", files=files, timeout=30)
                    response.raise_for_status()
                    result = response.json()
                    
                    if result.get("status") == "success":
                        with col2:
                            st.success(f"✅ Это: **{result['predicted_class']}**")
                            st.metric("Уверенность", f"{result['confidence']:.2%}")
                            st.bar_chart({"Вероятность": result['all_probabilities']})
                except Exception as e:
                    st.error(f"❌ Ошибка: {e}")

# ------------------------------------------
# 🔢 РЕЖИМ: MNIST (холст для рисования)
# ------------------------------------------
else:
    st.info("️ Нарисуйте цифру от 0 до 9 на чёрном холсте белой кистью")
    
    # Настройки кисти
    col1, col2 = st.columns(2)
    with col1:
        stroke_width = st.slider("Толщина кисти", 5, 30, 15)
    with col2:
        stroke_color = st.color_picker("Цвет кисти", "#FFFFFF")
        bg_color = st.color_picker("Цвет фона", "#000000")
        
    # Холст
    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="mnist_canvas",
        update_streamlit=True,
    )
    
    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Ваш рисунок", use_container_width=True)
            
        with col2:
            if st.button("🔮 Распознать цифру", type="primary"):
                with st.spinner("Распознаю..."):
                    try:
                        # Приводим к формату MNIST (28x28, grayscale)
                        if img.mode != 'L':
                            img = img.convert('L')
                        img = img.resize((28, 28), Image.LANCZOS)
                        
                        img_byte_arr = io.BytesIO()
                        img.save(img_byte_arr, format='PNG')
                        
                        files = {"file": ("digit.png", img_byte_arr.getvalue(), "image/png")}
                        response = requests.post(f"{API_URL}{endpoint}", files=files, timeout=30)
                        response.raise_for_status()
                        result = response.json()
                        
                        if result.get("status") == "success":
                            st.success(f"✅ Цифра: **{result['predicted_class']}**")
                            st.metric("Уверенность", f"{result['confidence']:.2%}")
                            st.bar_chart({"Вероятность": result['all_probabilities']})
                    except Exception as e:
                        st.error(f"❌ Ошибка: {e}")

# Боковая панель
with st.sidebar:
    st.header("ℹ️ Статус")
    if st.button("📡 Проверить API"):
        try:
            resp = requests.get(f"{API_URL}/", timeout=5)
            if resp.status_code == 200:
                st.success("✅ API работает!")
                st.json(resp.json())
            else:
                st.error(f" Код ответа: {resp.status_code}")
        except Exception as e:
            st.error(f" Не удалось подключиться: {e}")