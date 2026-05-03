import streamlit as st
import requests
from PIL import Image
import io
import numpy as np
from streamlit_drawable_canvas import st_canvas
import pandas as pd 

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
        
        if st.button("🔮 Классифицировать", type="primary"):
            with st.spinner("Анализирую..."):
                try:
                    img_byte_arr = io.BytesIO()
                    original.save(img_byte_arr, format='PNG')
                    files = {"file": ("image.png", img_byte_arr.getvalue(), "image/png")}
                    
                    # Определяем тип модели
                    model_type = "animal" if "животных" in model_choice else "digit"
                    
                    response = requests.post(f"{API_URL}{endpoint}", files=files, timeout=30)
                    response.raise_for_status()
                    result = response.json()
                    
                    # 🎨 Красивое отображение результата
                    display_prediction_result(result, model_type)
                    
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

def display_prediction_result(result, model_type):
    """Красивое отображение результатов классификации"""
    
    if result.get("status") != "success":
        st.error("❌ Ошибка в ответе API")
        return
    
    predicted_class = result['predicted_class']
    confidence = result['confidence']
    probabilities = result['all_probabilities']
    
    # Заголовок с результатом
    st.markdown("### 🎯 Результат классификации")
    
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col2:
        # Большая карточка с предсказанием
        st.markdown(f"""
        <div style='background-color: #4CAF50; padding: 20px; border-radius: 10px; text-align: center; margin: 10px 0;'>
            <h2 style='color: white; margin: 0;'>{predicted_class}</h2>
            <p style='color: white; margin: 5px 0 0 0;'>📊 Уверенность: {confidence:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Метрики
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Предсказанный класс", str(predicted_class))
    with col2:
        st.metric("Уверенность модели", f"{confidence:.2%}")
    with col3:
        st.metric("Количество классов", len(probabilities))
    
    # Визуализация распределения вероятностей
    st.markdown("### 📈 Распределение вероятностей по классам")
    
    # Создаём DataFrame для отображения
    if model_type == "animals":
        # Для животных используем названия классов
        classes = result.get('classes', [f"Class {i}" for i in range(len(probabilities))])
    else:
        # Для MNIST просто цифры
        classes = [f"Цифра {i}" for i in range(len(probabilities))]
    
    df_probs = pd.DataFrame({
        'Класс': classes,
        'Вероятность': probabilities,
        'Процент': [f"{p:.2%}" for p in probabilities]
    }).sort_values('Вероятность', ascending=False)
    
    # Отображаем таблицу с топ-5 классами
    st.markdown("#### 📋 Топ-5 наиболее вероятных классов")
    st.dataframe(
        df_probs.head(5),
        column_config={
            "Класс": "Класс",
            "Вероятность": st.column_config.ProgressColumn(
                "Вероятность",
                format="%.2%%",
                min_value=0,
                max_value=1,
            ),
            "Процент": "Процент"
        },
        hide_index=True,
        use_container_width=True
    )
    
    # Горизонтальный bar chart
    col1, col2 = st.columns([3, 1])
    with col1:
        st.bar_chart(
            df_probs.set_index('Класс')['Вероятность'],
            horizontal=True,
            color="#4CAF50"
        )
    
    # Круговая диаграмма для топ-5
    with col2:
        if len(probabilities) >= 3:
            st.markdown("#### 🥇 Топ-3")
            top3 = df_probs.head(3)
            for idx, row in top3.iterrows():
                st.write(f" **{row['Класс']}**: {row['Процент']}")
    
    # Подробная информация
    with st.expander("📊 Подробное распределение по всем классам"):
        st.write("Все вероятности:")
        for i, (cls, prob) in enumerate(zip(classes, probabilities)):
            bar_width = int(prob * 50)
            bar = "█" * bar_width + "░" * (50 - bar_width)
            st.text(f"{cls:15} |{bar}| {prob:.2%}")