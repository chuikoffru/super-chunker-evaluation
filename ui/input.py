"""
Компонент для ввода и обработки текста
"""

import streamlit as st
from sample_texts import get_sample_text, get_available_samples
from analyzer import TextStatistics
from config import ALLOWED_FILE_TYPES

def render_text_input() -> str:
    """
    Отрисовывает компонент для ввода текста
    
    Returns:
        Введенный пользователем текст
    """
    st.header("Входной текст")
    
    # Выбор источника текста
    text_source = st.radio(
        "Источник текста:",
        ["Образец текста", "Вставить текст", "Загрузить файл"]
    )
    
    input_text = ""
    
    if text_source == "Образец текста":
        input_text = _render_sample_text_input()
    elif text_source == "Вставить текст":
        input_text = _render_manual_text_input()
    elif text_source == "Загрузить файл":
        input_text = _render_file_upload_input()
    
    # Отображение статистики текста
    if input_text:
        _render_text_statistics(input_text)
    
    return input_text

def _render_sample_text_input() -> str:
    """Отрисовывает выбор образца текста"""
    sample_type = st.selectbox(
        "Выберите тип образца:",
        get_available_samples(),
        index=0
    )
    
    input_text = get_sample_text(sample_type or "Искусственный интеллект")
    st.text_area("Текст:", value=input_text, height=300, disabled=True)
    
    return input_text

def _render_manual_text_input() -> str:
    """Отрисовывает поле для ручного ввода текста"""
    input_text = st.text_area("Введите или вставьте текст:", height=300)
    return input_text

def _render_file_upload_input() -> str:
    """Отрисовывает загрузку файла"""
    uploaded_file = st.file_uploader(
        "Выберите текстовый файл",
        type=ALLOWED_FILE_TYPES,
        help=f"Поддерживаемые форматы: {', '.join(ALLOWED_FILE_TYPES)}"
    )
    
    input_text = ""
    if uploaded_file is not None:
        try:
            input_text = str(uploaded_file.read(), "utf-8")
            st.text_area("Загруженный текст:", value=input_text, height=300, disabled=True)
            st.success(f"Файл '{uploaded_file.name}' успешно загружен!")
        except UnicodeDecodeError:
            st.error("Ошибка при чтении файла. Убедитесь, что файл в формате UTF-8.")
        except Exception as e:
            st.error(f"Ошибка при загрузке файла: {str(e)}")
    
    return input_text

def _render_text_statistics(text: str):
    """Отображает статистику текста"""
    st.subheader("Статистика текста")
    
    text_stats = TextStatistics.get_text_stats(text)
    
    if text_stats:
        # Отображаем статистику в 2 колонки
        col1, col2 = st.columns(2)
        
        stats_items = list(text_stats.items())
        mid_point = len(stats_items) // 2
        
        # Первая колонка - первая половина метрик
        with col1:
            for key, value in stats_items[:mid_point]:
                st.metric(key, value)
        
        # Вторая колонка - вторая половина метрик
        with col2:
            for key, value in stats_items[mid_point:]:
                st.metric(key, value)
    else:
        st.warning("Не удалось получить статистику текста")

def validate_text_input(text: str) -> bool:
    """
    Проверяет корректность введенного текста
    
    Args:
        text: Текст для проверки
        
    Returns:
        True если текст корректный, False иначе
    """
    if not text or not text.strip():
        st.error("Пожалуйста, введите текст для чанкования.")
        return False
    
    if len(text.strip()) < 10:
        st.warning("Текст очень короткий. Результаты чанкования могут быть неинформативными.")
    
    return True 

def render_sidebar_text_input() -> str:
    """
    Отрисовывает компонент для ввода текста в sidebar
    
    Returns:
        Введенный пользователем текст
    """
    # Выбор источника текста
    text_source = st.sidebar.radio(
        "Источник текста:",
        ["Образец текста", "Вставить текст", "Загрузить файл"]
    )
    
    input_text = ""
    
    if text_source == "Образец текста":
        input_text = _render_sidebar_sample_text_input()
    elif text_source == "Вставить текст":
        input_text = _render_sidebar_manual_text_input()
    elif text_source == "Загрузить файл":
        input_text = _render_sidebar_file_upload_input()
    
    # Отображение статистики текста в компактном виде
    if input_text:
        _render_sidebar_text_statistics(input_text)
    
    return input_text

def _render_sidebar_sample_text_input() -> str:
    """Отрисовывает выбор образца текста в sidebar"""
    sample_type = st.sidebar.selectbox(
        "Выберите тип образца:",
        get_available_samples(),
        index=0
    )
    
    input_text = get_sample_text(sample_type or "Искусственный интеллект")
    
    # Показываем превью текста
    preview_text = input_text[:200] + "..." if len(input_text) > 200 else input_text
    st.sidebar.text_area("Превью:", value=preview_text, height=100, disabled=True)
    
    return input_text

def _render_sidebar_manual_text_input() -> str:
    """Отрисовывает поле для ручного ввода текста в sidebar"""
    input_text = st.sidebar.text_area("Введите или вставьте текст:", height=200)
    return input_text

def _render_sidebar_file_upload_input() -> str:
    """Отрисовывает загрузку файла в sidebar"""
    uploaded_file = st.sidebar.file_uploader(
        "Выберите текстовый файл",
        type=ALLOWED_FILE_TYPES,
        help=f"Поддерживаемые форматы: {', '.join(ALLOWED_FILE_TYPES)}"
    )
    
    input_text = ""
    if uploaded_file is not None:
        try:
            input_text = str(uploaded_file.read(), "utf-8")
            
            # Показываем превью загруженного текста
            preview_text = input_text[:200] + "..." if len(input_text) > 200 else input_text
            st.sidebar.text_area("Превью загруженного:", value=preview_text, height=100, disabled=True)
            st.sidebar.success(f"✅ '{uploaded_file.name}' загружен!")
        except UnicodeDecodeError:
            st.sidebar.error("Ошибка при чтении файла. Убедитесь, что файл в формате UTF-8.")
        except Exception as e:
            st.sidebar.error(f"Ошибка при загрузке файла: {str(e)}")
    
    return input_text

def _render_sidebar_text_statistics(text: str):
    """Отображает компактную статистику текста в sidebar"""
    text_stats = TextStatistics.get_text_stats(text)
    
    if text_stats:
        st.sidebar.markdown("**📊 Статистика:**")
        
        # Отображаем ключевые метрики в компактном виде
        key_stats = ['Символов', 'Слов', 'Предложений', 'Абзацев']
        for stat_name in key_stats:
            if stat_name in text_stats:
                st.sidebar.text(f"{stat_name}: {text_stats[stat_name]}")
    else:
        st.sidebar.warning("Не удалось получить статистику текста")