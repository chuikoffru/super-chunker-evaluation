"""
Компонент боковой панели с настройками чанкеров
"""

import streamlit as st
from typing import Dict, Any, List
from chunkers import CHUNKERS
from config import (
    DEFAULT_METHODS, MAX_CHUNK_SIZE, MAX_OVERLAP, MAX_SENTENCES_PER_CHUNK,
    MAX_PARAGRAPHS_PER_CHUNK, MAX_SIMILARITY_THRESHOLD, MIN_SIMILARITY_THRESHOLD,
    MAX_TIMEOUT, MAX_RETRIES, SPACY_MODELS, TOKENIZER_MODELS, EMBEDDING_MODELS,
    DEFAULT_SEPARATORS
)
from ui.input import render_sidebar_text_input

# Базовые методы чанкования (без семантических)
BASIC_METHODS = ['Символы', 'Рекурсивный', 'Токены', 'Предложения', 'Абзацы', 'Фиксированный размер']

def render_sidebar() -> tuple[List[str], Dict[str, Dict[str, Any]]]:
    """
    Отрисовывает боковую панель с настройками
    
    Returns:
        Кортеж из выбранных методов и их конфигураций
    """
    st.sidebar.header("Настройки чанкования")
    
    # Выбор методов чанкования
    selected_methods = st.sidebar.multiselect(
        "Выберите методы для тестирования:",
        list(CHUNKERS.keys()),
        default=DEFAULT_METHODS,
        help="Выберите один или несколько методов для сравнения"
    )
    
    if not selected_methods:
        st.sidebar.warning("Выберите хотя бы один метод чанкования.")
        return [], {}
    
    # Настройки для каждого выбранного метода
    method_configs = {}
    for method in selected_methods:
        st.sidebar.markdown(f"#### {method}")
        params = _render_chunker_params(method)
        method_configs[method] = params
    
    return selected_methods, method_configs

def render_basic_sidebar() -> tuple[List[str], Dict[str, Dict[str, Any]]]:
    """
    Отрисовывает боковую панель с настройками для базовых методов (без семантических)
    Использует чекбоксы вместо multiselect
    
    Returns:
        Кортеж из выбранных методов и их конфигураций
    """
    st.sidebar.header("⚙️ Настройки чанкования")
    
    # Выбор методов чанкования с помощью чекбоксов
    st.sidebar.subheader("Методы чанкования")
    selected_methods = []
    
    # Отображаем чекбоксы для каждого базового метода
    for method in BASIC_METHODS:
        # Устанавливаем по умолчанию первые три метода
        default_value = method in ['Символы', 'Рекурсивный', 'Токены']
        
        if st.sidebar.checkbox(
            method, 
            value=default_value, 
            key=f"basic_checkbox_{method}",
            help=_get_method_description(method)
        ):
            selected_methods.append(method)
    
    if not selected_methods:
        st.sidebar.warning("Выберите хотя бы один метод чанкования.")
        return [], {}
    
    # Настройки для каждого выбранного метода
    st.sidebar.markdown("---")
    st.sidebar.subheader("Параметры методов")
    
    method_configs = {}
    for method in selected_methods:
        with st.sidebar.expander(f"⚙️ {method}"):
            params = _render_basic_chunker_params(method)
            method_configs[method] = params
    
    return selected_methods, method_configs

def _get_method_description(method_name: str) -> str:
    """Возвращает описание для метода чанкования"""
    descriptions = {
        'Символы': 'Разбиение по количеству символов - простой и быстрый',
        'Рекурсивный': 'Умное разбиение с приоритизацией разделителей',
        'Токены': 'Разбиение по токенам для совместимости с LLM',
        'Предложения': 'Разбиение на группы предложений',
        'Абзацы': 'Разбиение на группы абзацев',
        'Фиксированный размер': 'Чанки одинакового размера'
    }
    return descriptions.get(method_name, f"Метод {method_name}")

def render_basic_sidebar_with_input() -> tuple[str, List[str], Dict[str, Dict[str, Any]]]:
    """
    Отрисовывает полную боковую панель с вводом текста и настройками чанкования в expander'ах
    
    Returns:
        Кортеж из (входной_текст, выбранные_методы, конфигурации_методов)
    """
    # Expander для ввода текста
    with st.sidebar.expander("📝 Входной текст", expanded=True):
        input_text = render_sidebar_text_input()
    
    # Expander для настроек чанкования  
    with st.sidebar.expander("⚙️ Настройки чанкования", expanded=True):
        # Выбор методов чанкования с помощью чекбоксов
        st.markdown("**Методы чанкования:**")
        selected_methods = []
        
        # Отображаем чекбоксы для каждого базового метода
        for method in BASIC_METHODS:
            # Устанавливаем по умолчанию первые три метода
            default_value = method in ['Символы', 'Рекурсивный', 'Токены']
            
            if st.checkbox(
                method, 
                value=default_value, 
                key=f"basic_checkbox_exp_{method}",
                help=_get_method_description(method)
            ):
                selected_methods.append(method)
        
        if not selected_methods:
            st.warning("Выберите хотя бы один метод чанкования.")
            return input_text, [], {}
        
        # Настройки для каждого выбранного метода
        st.markdown("---")
        st.markdown("**Параметры методов:**")
        
        method_configs = {}
        for method in selected_methods:
            with st.expander(f"⚙️ {method}"):
                params = _render_basic_chunker_params_in_expander(method)
                method_configs[method] = params
    
    return input_text, selected_methods, method_configs

def _render_basic_chunker_params_in_expander(method_name: str) -> Dict[str, Any]:
    """Отрисовывает параметры для базовых чанкеров внутри expander'а (без st.sidebar)"""
    params = {}
    
    if method_name == 'Символы':
        params = _render_character_chunker_params_in_expander(method_name)
    elif method_name == 'Рекурсивный':
        params = _render_recursive_chunker_params_in_expander(method_name)
    elif method_name == 'Токены':
        params = _render_token_chunker_params_in_expander(method_name)
    elif method_name == 'Предложения':
        params = _render_sentence_chunker_params_in_expander(method_name)
    elif method_name == 'Абзацы':
        params = _render_paragraph_chunker_params_in_expander(method_name)
    elif method_name == 'Фиксированный размер':
        params = _render_fixed_size_chunker_params_in_expander(method_name)
    
    return params

def _render_character_chunker_params_in_expander(method_name: str) -> Dict[str, Any]:
    """Параметры для символьного чанкера внутри expander'а"""
    chunk_size = st.slider(
        "Размер чанка (символы)", 
        100, MAX_CHUNK_SIZE, 1000, 
        key=f"{method_name}_size_exp",
        help="Максимальное количество символов в чанке"
    )
    chunk_overlap = st.slider(
        "Перекрытие", 
        0, min(MAX_OVERLAP, chunk_size//2), 200, 
        key=f"{method_name}_overlap_exp",
        help="Количество символов перекрытия между чанками"
    )
    separator = st.text_input(
        "Разделитель", 
        value="\n\n", 
        key=f"{method_name}_sep_exp",
        help="Разделитель для поиска границ чанков"
    )
    
    return {
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'separator': separator
    }

def _render_recursive_chunker_params_in_expander(method_name: str) -> Dict[str, Any]:
    """Параметры для рекурсивного чанкера внутри expander'а"""
    chunk_size = st.slider(
        "Размер чанка (символы)", 
        100, MAX_CHUNK_SIZE, 1000, 
        key=f"{method_name}_size_exp"
    )
    chunk_overlap = st.slider(
        "Перекрытие", 
        0, min(MAX_OVERLAP, chunk_size//2), 200, 
        key=f"{method_name}_overlap_exp"
    )
    separators_text = st.text_area(
        "Разделители (по одному на строку)", 
        value=DEFAULT_SEPARATORS, 
        key=f"{method_name}_seps_exp",
        help="Разделители в порядке приоритета"
    )
    separators = [s for s in separators_text.split('\n') if s is not None]
    
    return {
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'separators': separators
    }

def _render_token_chunker_params_in_expander(method_name: str) -> Dict[str, Any]:
    """Параметры для токенного чанкера внутри expander'а"""
    chunk_size = st.slider(
        "Размер чанка (токены)", 
        50, 2048, 512, 
        key=f"{method_name}_size_exp"
    )
    chunk_overlap = st.slider(
        "Перекрытие (токены)", 
        0, min(200, chunk_size//2), 50, 
        key=f"{method_name}_overlap_exp"
    )
    model_name = st.selectbox(
        "Модель токенизации",
        TOKENIZER_MODELS,
        key=f"{method_name}_model_exp",
        help="Модель для подсчета токенов"
    )
    
    return {
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'model_name': model_name
    }

def _render_sentence_chunker_params_in_expander(method_name: str) -> Dict[str, Any]:
    """Параметры для предложенческого чанкера внутри expander'а"""
    sentences_per_chunk = st.slider(
        "Предложений в чанке", 
        1, MAX_SENTENCES_PER_CHUNK, 5, 
        key=f"{method_name}_sentences_exp"
    )
    overlap_sentences = st.slider(
        "Перекрытие (предложения)", 
        0, min(5, sentences_per_chunk-1), 1, 
        key=f"{method_name}_overlap_exp"
    )
    
    return {
        'sentences_per_chunk': sentences_per_chunk,
        'overlap_sentences': overlap_sentences
    }

def _render_paragraph_chunker_params_in_expander(method_name: str) -> Dict[str, Any]:
    """Параметры для абзацного чанкера внутри expander'а"""
    paragraphs_per_chunk = st.slider(
        "Абзацев в чанке", 
        1, MAX_PARAGRAPHS_PER_CHUNK, 3, 
        key=f"{method_name}_paragraphs_exp"
    )
    overlap_paragraphs = st.slider(
        "Перекрытие (абзацы)", 
        0, min(3, paragraphs_per_chunk-1), 1, 
        key=f"{method_name}_overlap_exp"
    )
    
    return {
        'paragraphs_per_chunk': paragraphs_per_chunk,
        'overlap_paragraphs': overlap_paragraphs
    }

def _render_fixed_size_chunker_params_in_expander(method_name: str) -> Dict[str, Any]:
    """Параметры для чанкера фиксированного размера внутри expander'а"""
    chunk_size = st.slider(
        "Размер чанка", 
        100, MAX_CHUNK_SIZE, 1000, 
        key=f"{method_name}_size_exp"
    )
    overlap = st.slider(
        "Перекрытие", 
        0, min(MAX_OVERLAP, chunk_size//2), 100, 
        key=f"{method_name}_overlap_exp"
    )
    
    return {
        'chunk_size': chunk_size,
        'overlap': overlap
    }

def _render_chunker_params(method_name: str) -> Dict[str, Any]:
    """Отрисовывает параметры для конкретного чанкера"""
    params = {}
    
    if method_name == 'Символы':
        params = _render_character_chunker_params(method_name)
    elif method_name == 'Рекурсивный':
        params = _render_recursive_chunker_params(method_name)
    elif method_name == 'Токены':
        params = _render_token_chunker_params(method_name)
    elif method_name == 'Предложения':
        params = _render_sentence_chunker_params(method_name)
    elif method_name == 'Абзацы':
        params = _render_paragraph_chunker_params(method_name)
    elif method_name == 'Spacy семантический':
        params = _render_spacy_semantic_chunker_params(method_name)
    elif method_name == 'OpenAI семантический':
        params = _render_openai_semantic_chunker_params(method_name)
    elif method_name == 'Фиксированный размер':
        params = _render_fixed_size_chunker_params(method_name)
    
    return params

def _render_basic_chunker_params(method_name: str) -> Dict[str, Any]:
    """Отрисовывает параметры для базовых чанкеров (без семантических)"""
    params = {}
    
    if method_name == 'Символы':
        params = _render_character_chunker_params(method_name)
    elif method_name == 'Рекурсивный':
        params = _render_recursive_chunker_params(method_name)
    elif method_name == 'Токены':
        params = _render_token_chunker_params(method_name)
    elif method_name == 'Предложения':
        params = _render_sentence_chunker_params(method_name)
    elif method_name == 'Абзацы':
        params = _render_paragraph_chunker_params(method_name)
    elif method_name == 'Фиксированный размер':
        params = _render_fixed_size_chunker_params(method_name)
    
    return params

def _render_character_chunker_params(method_name: str) -> Dict[str, Any]:
    """Параметры для символьного чанкера"""
    chunk_size = st.sidebar.slider(
        "Размер чанка (символы)", 
        100, MAX_CHUNK_SIZE, 1000, 
        key=f"{method_name}_size",
        help="Максимальное количество символов в чанке"
    )
    chunk_overlap = st.sidebar.slider(
        "Перекрытие", 
        0, min(MAX_OVERLAP, chunk_size//2), 200, 
        key=f"{method_name}_overlap",
        help="Количество символов перекрытия между чанками"
    )
    separator = st.sidebar.text_input(
        "Разделитель", 
        value="\n\n", 
        key=f"{method_name}_sep",
        help="Разделитель для поиска границ чанков"
    )
    
    return {
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'separator': separator
    }

def _render_recursive_chunker_params(method_name: str) -> Dict[str, Any]:
    """Параметры для рекурсивного чанкера"""
    chunk_size = st.sidebar.slider(
        "Размер чанка (символы)", 
        100, MAX_CHUNK_SIZE, 1000, 
        key=f"{method_name}_size"
    )
    chunk_overlap = st.sidebar.slider(
        "Перекрытие", 
        0, min(MAX_OVERLAP, chunk_size//2), 200, 
        key=f"{method_name}_overlap"
    )
    separators_text = st.sidebar.text_area(
        "Разделители (по одному на строку)", 
        value=DEFAULT_SEPARATORS, 
        key=f"{method_name}_seps",
        help="Разделители в порядке приоритета"
    )
    separators = [s for s in separators_text.split('\n') if s is not None]
    
    return {
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'separators': separators
    }

def _render_token_chunker_params(method_name: str) -> Dict[str, Any]:
    """Параметры для токенного чанкера"""
    chunk_size = st.sidebar.slider(
        "Размер чанка (токены)", 
        50, 2048, 512, 
        key=f"{method_name}_size"
    )
    chunk_overlap = st.sidebar.slider(
        "Перекрытие (токены)", 
        0, min(200, chunk_size//2), 50, 
        key=f"{method_name}_overlap"
    )
    model_name = st.sidebar.selectbox(
        "Модель токенизации",
        TOKENIZER_MODELS,
        key=f"{method_name}_model",
        help="Модель для подсчета токенов"
    )
    
    return {
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'model_name': model_name
    }

def _render_sentence_chunker_params(method_name: str) -> Dict[str, Any]:
    """Параметры для предложенческого чанкера"""
    sentences_per_chunk = st.sidebar.slider(
        "Предложений в чанке", 
        1, MAX_SENTENCES_PER_CHUNK, 5, 
        key=f"{method_name}_sentences"
    )
    overlap_sentences = st.sidebar.slider(
        "Перекрытие (предложения)", 
        0, min(5, sentences_per_chunk-1), 1, 
        key=f"{method_name}_overlap"
    )
    
    return {
        'sentences_per_chunk': sentences_per_chunk,
        'overlap_sentences': overlap_sentences
    }

def _render_paragraph_chunker_params(method_name: str) -> Dict[str, Any]:
    """Параметры для абзацного чанкера"""
    paragraphs_per_chunk = st.sidebar.slider(
        "Абзацев в чанке", 
        1, MAX_PARAGRAPHS_PER_CHUNK, 3, 
        key=f"{method_name}_paragraphs"
    )
    overlap_paragraphs = st.sidebar.slider(
        "Перекрытие (абзацы)", 
        0, min(3, paragraphs_per_chunk-1), 1, 
        key=f"{method_name}_overlap"
    )
    
    return {
        'paragraphs_per_chunk': paragraphs_per_chunk,
        'overlap_paragraphs': overlap_paragraphs
    }

def _render_spacy_semantic_chunker_params(method_name: str) -> Dict[str, Any]:
    """Параметры для Spacy семантического чанкера"""
    similarity_threshold = st.sidebar.slider(
        "Порог схожести", 
        MIN_SIMILARITY_THRESHOLD, MAX_SIMILARITY_THRESHOLD, 0.7, 0.05, 
        key=f"{method_name}_threshold",
        help="Минимальная схожесть для объединения предложений"
    )
    max_chunk_size = st.sidebar.slider(
        "Максимальный размер чанка", 
        500, 3000, 1000, 
        key=f"{method_name}_max_size"
    )
    model_name = st.sidebar.selectbox(
        "Spacy модель",
        SPACY_MODELS,
        key=f"{method_name}_model",
        help="Модель Spacy для эмбеддингов"
    )
    
    return {
        'similarity_threshold': similarity_threshold,
        'max_chunk_size': max_chunk_size,
        'model_name': model_name
    }

def _render_openai_semantic_chunker_params(method_name: str) -> Dict[str, Any]:
    """Параметры для OpenAI семантического чанкера"""
    similarity_threshold = st.sidebar.slider(
        "Порог схожести", 
        MIN_SIMILARITY_THRESHOLD, MAX_SIMILARITY_THRESHOLD, 0.7, 0.05, 
        key=f"{method_name}_threshold"
    )
    max_chunk_size = st.sidebar.slider(
        "Максимальный размер чанка", 
        500, 3000, 1000, 
        key=f"{method_name}_max_size"
    )
    
    # API настройки
    with st.sidebar.expander("🔧 API настройки", expanded=False):
        api_url = st.text_input(
            "API URL", 
            value="https://api.openai.com/v1/embeddings",
            key=f"{method_name}_api_url",
            help="URL для API эмбеддингов"
        )
        api_key = st.text_input(
            "API ключ", 
            type="password",
            help="Оставьте пустым для использования переменной окружения OPENAI_API_KEY",
            key=f"{method_name}_api_key"
        )
        model_name = st.selectbox(
            "Модель эмбеддингов",
            EMBEDDING_MODELS,
            key=f"{method_name}_model"
        )
        timeout = st.slider(
            "Таймаут (сек)", 
            5, MAX_TIMEOUT, 30, 
            key=f"{method_name}_timeout"
        )
        max_retries = st.slider(
            "Количество попыток", 
            1, MAX_RETRIES, 3, 
            key=f"{method_name}_retries"
        )
    
    return {
        'similarity_threshold': similarity_threshold,
        'max_chunk_size': max_chunk_size,
        'api_url': api_url,
        'api_key': api_key,
        'model_name': model_name,
        'timeout': timeout,
        'max_retries': max_retries
    }

def _render_fixed_size_chunker_params(method_name: str) -> Dict[str, Any]:
    """Параметры для чанкера фиксированного размера"""
    chunk_size = st.sidebar.slider(
        "Размер чанка", 
        100, MAX_CHUNK_SIZE, 1000, 
        key=f"{method_name}_size"
    )
    overlap = st.sidebar.slider(
        "Перекрытие", 
        0, min(MAX_OVERLAP, chunk_size//2), 100, 
        key=f"{method_name}_overlap"
    )
    
    return {
        'chunk_size': chunk_size,
        'overlap': overlap
    } 