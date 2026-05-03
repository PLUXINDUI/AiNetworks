import streamlit as st
import requests
from PIL import Image
import io
import numpy as np
import pandas as pd
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Multi-Model Classifier", layout="wide")
st.title("Мульти-модельный классификатор")

# Выбор модели
model_choice = st.radio(
    "Выберите задачу:",
    ["🐾 Классификация животных", "🔢 Распознавание цифр (MNIST)"],
    horizontal=True
)

# URL API
API_URL = st.text_input("URL вашего API (Render):", value="https://ainetworks.onrender.com")

# ============================================
# 🐾 РЕЖИМ: ЖИВОТНЫЕ
# ============================================
if "животных" in model_choice:
    st.markdown("### 📸 Загрузите фотографию животного (кошка, лягушка, олень)")
    st.info("💡 Поддерживаются форматы: PNG, JPG, JPEG")
    
    uploaded_file = st.file_uploader("Выберите изображение", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📷 Оригинал:")
            original = Image.open(uploaded_file)
            st.image(original, caption="Загруженное изображение", use_container_width=True)
        
        # Кнопка классификации
        if st.button("🔮 Классифицировать животное", type="primary"):
            with st.spinner("🔄 Анализирую изображение..."):
                try:
                    # Подготовка файла
                    img_byte_arr = io.BytesIO()
                    original.save(img_byte_arr, format='PNG')
                    
                    # Отправка на API
                    endpoint = "/predict/animal"
                    files = {"file": ("animal.png", img_byte_arr.getvalue(), "image/png")}
                    response = requests.post(f"{API_URL}{endpoint}", files=files, timeout=30)
                    response.raise_for_status()
                    result = response.json()
                    
                    if result.get("status") == "success":
                        predicted_class = result['predicted_class']
                        confidence = result['confidence']
                        probabilities = result['all_probabilities']
                        class_index = result.get('class_index', 0)
                        
                        with col2:
                            st.markdown("#### 🎯 Результат:")
                            
                            # 🎨 Красивая карточка с предсказанием
                            st.markdown(f"""
                            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                        padding: 20px; border-radius: 10px; text-align: center; margin: 10px 0;'>
                                <h1 style='color: white; margin: 0; font-size: 48px;'>{predicted_class}</h1>
                                <p style='color: white; margin: 10px 0 0 0; font-size: 18px;'>
                                    📊 Уверенность: {confidence:.1%}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Метрики
                            col_m1, col_m2, col_m3 = st.columns(3)
                            with col_m1:
                                st.metric("Предсказанный класс", str(predicted_class))
                            with col_m2:
                                st.metric("Уверенность", f"{confidence:.2%}")
                            with col_m3:
                                st.metric("Индекс класса", class_index)
                        
                        # Визуализация вероятностей (на всю ширину)
                        st.markdown("### 📈 Распределение вероятностей по классам")
                        
                        # Получаем названия классов из API или генерируем
                        classes = result.get('classes', [f"Class {i}" for i in range(len(probabilities))])
                        
                        # Создаём DataFrame
                        df_probs = pd.DataFrame({
                            'Класс': classes,
                            'Вероятность': probabilities,
                            'Процент': [f"{p:.2%}" for p in probabilities]
                        }).sort_values('Вероятность', ascending=False)
                        
                        # Топ-5 в таблице с прогресс-барами
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
                        col_chart1, col_chart2 = st.columns([3, 1])
                        with col_chart1:
                            st.bar_chart(
                                df_probs.set_index('Класс')['Вероятность'],
                                horizontal=True,
                                color="#667eea"
                            )
                        
                        # Топ-3 с медалями
                        with col_chart2:
                            st.markdown("#### 🏆 Топ-3")
                            top3 = df_probs.head(3)
                            medals = ["🥇", "", "🥉"]
                            for idx, (medal, row) in enumerate(zip(medals, top3.iterrows())):
                                row_data = row[1]
                                st.write(f"{medal} **{row_data['Класс']}**: {row_data['Процент']}")
                        
                        # Подробное распределение
                        with st.expander("📊 Подробное распределение по всем классам"):
                            st.write("Все вероятности:")
                            for cls, prob in zip(classes, probabilities):
                                bar_width = int(prob * 50)
                                bar = "█" * bar_width + "░" * (50 - bar_width)
                                st.text(f"{cls:20} |{bar}| {prob:.2%}")
                        
                        # Дополнительная информация
                        with st.expander("ℹ️ Техническая информация"):
                            st.json({
                                "predicted_class": predicted_class,
                                "class_index": class_index,
                                "confidence": f"{confidence:.4f}",
                                "all_probabilities": probabilities
                            })
                            
                    else:
                        st.error("❌ Ошибка в ответе API")
                        st.json(result)
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ Не удалось подключиться к API")
                    st.warning(f"Проверьте URL: {API_URL}")
                    st.info("💡 Убедитесь, что API запущен на Render и доступен")
                except requests.exceptions.Timeout:
                    st.error("❌ Превышено время ожидания ответа от API")
                    st.warning("💡 API может быть перегружен или модель большая")
                except Exception as e:
                    st.error(f"❌ Ошибка: {type(e).__name__}: {e}")
                    import traceback
                    st.code(traceback.format_exc())

# ============================================
# 🔢 РЕЖИМ: MNIST (холст для рисования)
# ============================================
else:
    st.markdown("### ✏️ Нарисуйте цифру от 0 до 9")
    st.info("💡 Рисуйте белой кистью на чёрном фоне")
    
    # Настройки кисти
    col_settings1, col_settings2 = st.columns(2)
    with col_settings1:
        stroke_width = st.slider("Толщина кисти", 5, 30, 15)
    with col_settings2:
        stroke_color = st.color_picker("Цвет кисти", "#FFFFFF")
        bg_color = st.color_picker("Цвет фона", "#000000")
    
    # Холст для рисования
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
    
    # Кнопка распознавания
    if st.button("🔮 Распознать цифру", type="primary"):
        if canvas_result.image_data is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📝 Ваш рисунок:")
                img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
                st.image(img, caption="Нарисованная цифра", use_container_width=True)
            
            with col2:
                st.markdown("#### 🎯 Результат:")
                
                with st.spinner("🔄 Распознаю..."):
                    try:
                        # Предобработка
                        if img.mode != 'L':
                            img = img.convert('L')
                        img = img.resize((28, 28), Image.LANCZOS)
                        
                        img_byte_arr = io.BytesIO()
                        img.save(img_byte_arr, format='PNG')
                        
                        # Отправка на API
                        endpoint = "/predict/digit"
                        files = {"file": ("digit.png", img_byte_arr.getvalue(), "image/png")}
                        response = requests.post(f"{API_URL}{endpoint}", files=files, timeout=30)
                        response.raise_for_status()
                        result = response.json()
                        
                        if result.get("status") == "success":
                            predicted_class = result['predicted_class']
                            confidence = result['confidence']
                            probabilities = result['all_probabilities']
                            
                            # 🎨 Красивое отображение
                            st.markdown(f"""
                            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                        padding: 20px; border-radius: 10px; text-align: center; margin: 10px 0;'>
                                <h1 style='color: white; margin: 0; font-size: 48px;'>{predicted_class}</h1>
                                <p style='color: white; margin: 10px 0 0 0; font-size: 18px;'>
                                    📊 Уверенность: {confidence:.1%}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Метрики
                            col_m1, col_m2, col_m3 = st.columns(3)
                            with col_m1:
                                st.metric("Предсказанная цифра", str(predicted_class))
                            with col_m2:
                                st.metric("Уверенность", f"{confidence:.2%}")
                            with col_m3:
                                top_3_indices = np.argsort(probabilities)[-3:][::-1]
                                st.metric("Топ-3", f"{top_3_indices[0]}, {top_3_indices[1]}, {top_3_indices[2]}")
                            
                            # Визуализация
                            st.markdown("### 📈 Распределение вероятностей")
                            
                            classes = [f"Цифра {i}" for i in range(10)]
                            df_probs = pd.DataFrame({
                                'Класс': classes,
                                'Вероятность': probabilities,
                                'Процент': [f"{p:.2%}" for p in probabilities]
                            }).sort_values('Вероятность', ascending=False)
                            
                            st.markdown("#### 📋 Топ-5 наиболее вероятных цифр")
                            st.dataframe(
                                df_probs.head(5),
                                column_config={
                                    "Класс": "Цифра",
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
                            
                            st.bar_chart(
                                df_probs.set_index('Класс')['Вероятность'],
                                horizontal=True,
                                color="#667eea"
                            )
                            
                            st.markdown("#### 🏆 Топ-3 кандидата")
                            top3 = df_probs.head(3)
                            medals = ["🥇", "🥈", ""]
                            for idx, (medal, row) in enumerate(zip(medals, top3.iterrows())):
                                row_data = row[1]
                                st.write(f"{medal} **{row_data['Класс']}**: {row_data['Процент']} "
                                       f"({'█' * int(row_data['Вероятность']*20)}{'░' * (20-int(row_data['Вероятность']*20))})")
                            
                            with st.expander("📊 Подробное распределение по всем цифрам"):
                                for i, (cls, prob) in enumerate(zip(classes, probabilities)):
                                    bar_width = int(prob * 50)
                                    bar = "█" * bar_width + "░" * (50 - bar_width)
                                    st.text(f"{cls:10} |{bar}| {prob:.2%}")
                            
                        else:
                            st.error("❌ Ошибка в ответе API")
                            st.json(result)
                            
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Не удалось подключиться к API")
                        st.warning(f"Проверьте URL: {API_URL}")
                    except requests.exceptions.Timeout:
                        st.error("❌ Превышено время ожидания")
                    except Exception as e:
                        st.error(f"❌ Ошибка: {type(e).__name__}: {e}")
        else:
            st.warning("⚠️ Сначала нарисуйте цифру на холсте!")

# ============================================
# БОКОВАЯ ПАНЕЛЬ
# ============================================
with st.sidebar:
    st.header("ℹ️ Статус системы")
    
    st.markdown("---")
    st.markdown("**Текущий режим:**")
    st.write("🐾 Животные" if "животных" in model_choice else "🔢 MNIST")
    
    st.markdown("---")
    st.markdown("**API:**")
    st.write(f"🔗 {API_URL}")
    
    if st.button("📡 Проверить доступность API", type="primary"):
        try:
            with st.spinner("Проверка..."):
                resp = requests.get(f"{API_URL}/", timeout=5)
                if resp.status_code == 200:
                    st.success("✅ API работает!")
                    with st.expander("📋 Информация об API"):
                        st.json(resp.json())
                else:
                    st.error(f"❌ Статус: {resp.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("❌ Не удалось подключиться")
            st.warning("Проверьте URL API")
        except Exception as e:
            st.error(f"❌ Ошибка: {e}")
    
    st.markdown("---")
    st.markdown("**💡 Советы:**")
    if "животных" in model_choice:
        st.write("• Загружайте чёткие фото")
        st.write("• Животное должно быть в кадре")
        st.write("• Хорошее освещение")
    else:
        st.write("• Рисуйте крупно")
        st.write("• Белым цветом")
        st.write("• По центру холста")
    
    st.markdown("---")
    st.caption("Multi-Model Classifier v1.0")