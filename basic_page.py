"""
Страница базового чанкования - основной функционал SuperChunker
"""

import streamlit as st

from config import initialize_session_state
from ui import (
    validate_text_input,
    render_results_section,
    run_chunking_process
)
from ui.sidebar import render_basic_sidebar_with_input

def show_basic_page():
    """Отображает страницу базового чанкования"""
    
    # Инициализация состояния
    initialize_session_state()
    
    # Заголовок
    st.title("✂️ SuperChunker")
    st.markdown("### Инструмент для тестирования различных методов чанкования текстов")
    
    # Боковая панель с вводом текста и настройками в expander'ах
    input_text, selected_methods, method_configs = render_basic_sidebar_with_input()
    
    # Основная область на всю ширину
    # Компонент результатов
    render_results_section()
    
    # Кнопка запуска чанкования
    if st.button("🚀 Запустить чанкование", type="primary", use_container_width=True):
        if not validate_text_input(input_text):
            return
            
        if not selected_methods:
            st.error("Пожалуйста, выберите хотя бы один метод чанкования.")
            return
        
        # Выполняем чанкование
        success = run_chunking_process(selected_methods, method_configs, input_text)
        
        if success:
            st.rerun()  # Обновляем интерфейс для отображения результатов 