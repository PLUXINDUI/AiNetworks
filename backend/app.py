import streamlit as st
import requests
import plotly.graph_objects as go

API_URL = st.sidebar.text_input(
    "API URL",
    value="https://classification-api-ipqh.onrender.com/predict",
    help="URL вашего FastAPI бэкенда",
)

st.title("Классификатор изображений: Cats / Dogs / Snakes")
st.markdown("Загрузите изображение для классификации.")

uploaded_file = st.file_uploader("Выберите изображение", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Загруженное изображение", width=300)
    image_bytes = uploaded_file.getvalue()

    if st.button("Классифицировать", type="primary"):
        with st.spinner("Отправка на сервер (первый запрос может занять до 2 минут)..."):
            try:
                resp = requests.post(
                    API_URL,
                    files={"file": ("image.png", image_bytes, "image/png")},
                    timeout=120,
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
