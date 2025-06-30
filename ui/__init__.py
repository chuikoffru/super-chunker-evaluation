"""
Пакет UI компонентов для SuperChunker

Содержит модульные компоненты интерфейса:
    input: Компонент для ввода и обработки текста
    sidebar: Боковая панель с параметрами чанкеров  
    results: Отображение результатов чанкования
    advanced_semantic: Продвинутое семантическое чанкование
"""

from .input import render_text_input, validate_text_input
from .sidebar import render_sidebar
from .results import render_results_section, run_chunking_process

__all__ = [
    'render_text_input',
    'validate_text_input',
    'render_sidebar', 
    'render_results_section',
    'run_chunking_process'
] 