import streamlit as st
import requests
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import canvas
import io
import plotly.graph_objects as go

API_URL = st.sidebar.text_input(
    "API URL",
    value="https://classification-api-ipqh.onrender.com/predict",
    help="URL вашего FastAPI бэкенда",
)

st.title("Классификатор изображений: Cats / Dogs / Snakes")
st.markdown("Загрузите изображение или нарисуйте на холсте для классификации.")

CLASS_NAMES = ["cats", "dogs", "snakes"]

tab1, tab2 = st.tabs(["Загрузить изображение", "Нарисовать на холсте"])

image_bytes = None

with tab1:
    uploaded_file = st.file_uploader("Выберите изображение", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Загруженное изображение", width=300)
        image_bytes = uploaded_file.getvalue()

with tab2:
    st.markdown("Нарисуйте животное (кошку, собаку или змею):")
    canvas_result = canvas(
        stroke_width=8,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=256,
        width=256,
        drawing_mode="freedraw",
        key="canvas",
    )
    if canvas_result.image_data is not None:
        img_arr = canvas_result.image_data[:, :, :3].astype(np.uint8)
        if img_arr.sum() > 0:
            img_pil = Image.fromarray(img_arr)
            buf = io.BytesIO()
            img_pil.save(buf, format="PNG")
            image_bytes = buf.getvalue()

if st.button("Классифицировать", type="primary", disabled=image_bytes is None):
    with st.spinner("Отправка на сервер..."):
        try:
            resp = requests.post(
                API_URL,
                files={"file": ("image.png", image_bytes, "image/png")},
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()

            st.success(
                f"**Предсказание:** {result['predicted_class']}  "
                f"(**{result['confidence']}%** уверенность)"
            )

            probs = result["probabilities"]
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=list(probs.keys()),
                        y=list(probs.values()),
                        marker_color=["#FF6B6B", "#4ECDC4", "#45B7D1"],
                        text=[f"{v}%" for v in probs.values()],
                        textposition="auto",
                    )
                ]
            )
            fig.update_layout(
                title="Распределение вероятностей по классам",
                yaxis_title="Вероятность (%)",
                yaxis_range=[0, 105],
            )
            st.plotly_chart(fig, use_container_width=True)

        except requests.exceptions.ConnectionError:
            st.error("Не удалось подключиться к API. Проверьте URL и доступность сервера.")
        except Exception as e:
            st.error(f"Ошибка: {e}")
