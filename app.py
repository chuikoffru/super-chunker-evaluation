"""
Главное приложение SuperChunker - инструмент для тестирования методов чанкования текстов
"""

import streamlit as st

from config import setup_page_config, initialize_session_state
from ui import (
    render_text_input, 
    validate_text_input,
    render_sidebar, 
    render_results_section,
    run_chunking_process
)

def main():
    """Основная функция приложения"""
    
    # Настройка страницы и инициализация состояния
    setup_page_config()
    initialize_session_state()
    
    # Заголовок приложения
    st.title("✂️ SuperChunker")
    st.markdown("### Инструмент для тестирования различных методов чанкования текстов")
    
    # Основная компоновка: боковая панель + основная область
    with st.sidebar:
        selected_methods, method_configs = render_sidebar()
    
    # Основная область в две колонки
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Компонент ввода текста
        input_text = render_text_input()
    
    with col2:
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

if __name__ == "__main__":
    main() 