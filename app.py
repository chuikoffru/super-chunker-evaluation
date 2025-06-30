"""
Главное приложение SuperChunker - инструмент для тестирования методов чанкования текстов
"""

import streamlit as st
from config import setup_page_config

# Импорт страниц
from basic_page import show_basic_page
from advanced_page import show_advanced_page
from documentation_page import show_documentation_page
from debug_chunking_page import show_debug_page

def main():
    """Основная функция приложения"""
    
    # Настройка страницы
    setup_page_config()
    
    # Определяем страницы
    pages = [
        st.Page(show_documentation_page, title="Документация", icon="📚", default=True),
        st.Page(show_basic_page, title="Базовое чанкование", icon="🏠"),
        st.Page(show_advanced_page, title="Продвинутый анализ", icon="🧠"),
        st.Page(show_debug_page, title="Отладка", icon="🔍")
    ]
    
    # Создаем навигацию
    pg = st.navigation(pages)
    
    # Запускаем выбранную страницу
    pg.run()

if __name__ == "__main__":
    main() 