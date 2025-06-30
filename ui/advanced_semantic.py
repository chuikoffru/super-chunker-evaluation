"""
UI модуль для продвинутого семантического чанкования
"""

import streamlit as st
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple

from sample_texts import SAMPLE_TEXTS
from chunkers.registry import get_chunker_by_name

def load_test_texts() -> List[Dict[str, str]]:
    """Загружает тестовые тексты разного размера с названиями"""
    texts = []
    
    # Добавляем примеры из sample_texts.py (это словарь)
    for title, text in SAMPLE_TEXTS.items():
        if isinstance(text, str) and len(text) > 100:
            texts.append({"name": title, "text": text})
    
    # Добавляем тексты разной длины для тестирования масштабируемости
    base_text = """
    Искусственный интеллект представляет собой одну из самых революционных технологий современности.
    Машинное обучение и глубокие нейронные сети позволяют компьютерам выполнять задачи, которые ранее
    требовали человеческого интеллекта. Обработка естественного языка открывает новые возможности
    для анализа и понимания текстов.
    """
    
    # Короткий текст
    texts.append({"name": "Тестовый текст (короткий)", "text": base_text})
    
    # Средний текст
    texts.append({"name": "Тестовый текст (средний)", "text": base_text * 5})
    
    # Длинный текст
    texts.append({"name": "Тестовый текст (длинный)", "text": base_text * 20})
    
    # Очень длинный текст
    texts.append({"name": "Тестовый текст (очень длинный)", "text": base_text * 50})
    
    return texts

def render_chunk_visual(chunk: str, chunk_index: int, method: str = ""):
    """Отображает один чанк в простом и понятном формате"""
    word_count = len(chunk.split())
    char_count = len(chunk)
    
    # Простой заголовок с метриками
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"**📄 Чанк {chunk_index + 1}**")
    with col2:
        st.markdown(f"*{word_count} слов*")
    with col3:
        st.markdown(f"*{char_count} символов*")
    
    # Отображаем чанк с помощью st.code
    st.code(chunk, language="text", wrap_lines=True)

def render_chunking_results():
    """Отображает результаты чанкования для всех тестовых текстов"""
    
    # Проверяем наличие результатов чанкования в session_state
    if ('chunking_results' not in st.session_state or 
        not st.session_state.chunking_results):
        st.info("🔍 Запустите чанкование, чтобы увидеть результаты для тестовых текстов")
        return
    
    st.header("📋 Результаты чанкования тестовых текстов")
    st.markdown("Просмотрите как каждый метод разбивает тестовые тексты на чанки")
    
    # Загружаем тестовые тексты
    test_texts = load_test_texts()
    available_methods = list(st.session_state.chunking_results.keys())
    
    # Создаем выбор текста и метода в колонках
    col1, col2 = st.columns(2)
    
    with col1:
        selected_text_idx = st.selectbox(
            "📖 Выберите тестовый текст:",
            range(len(test_texts)),
            format_func=lambda i: f"{test_texts[i]['name']} ({len(test_texts[i]['text'].split())} слов)"
        )
    
    with col2:
        selected_method = st.selectbox(
            "🔧 Выберите метод чанкования:",
            available_methods,
            format_func=lambda x: x.title()
        )
    
    if selected_text_idx is not None and selected_method:
        # Получаем выбранные данные
        text_data = test_texts[selected_text_idx]
        title = text_data["name"]
        text_content = text_data["text"]
        
        # Ищем результаты для выбранного текста и метода
        method_results = st.session_state.chunking_results[selected_method]
        selected_result = None
        
        for result in method_results:
            if result['text_name'] == title:
                selected_result = result
                break
        
        if not selected_result:
            st.error("❌ Результаты чанкования для выбранного текста не найдены")
            return
        
        # Показываем информацию о тексте
        st.markdown("---")
        st.subheader(f"📖 {title}")
        
        # Статистика исходного текста
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📝 Символов", len(text_content))
        with col2:
            st.metric("📝 Слов", len(text_content.split()))
        with col3:
            sentences = len([s for s in text_content.split('.') if s.strip()])
            st.metric("📝 Предложений", sentences)
        
        # Показываем исходный текст
        st.markdown("**📝 Исходный текст:**")
        st.code(text_content, language="text", wrap_lines=True)
        
        # Разделитель
        st.markdown("---")
        st.subheader(f"🔧 Результат чанкования методом: {selected_method.title()}")
        
        chunks = selected_result['chunks']
        
        if chunks:
            # Статистика чанкования
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Количество чанков", len(chunks))
            with col2:
                avg_chunk_size = sum(len(chunk.split()) for chunk in chunks) / len(chunks)
                st.metric("📏 Средний размер", f"{avg_chunk_size:.1f} слов")
            with col3:
                total_words = sum(len(chunk.split()) for chunk in chunks)
                original_words = len(text_content.split())
                coverage = (total_words / original_words) * 100 if original_words > 0 else 0
                st.metric("🎯 Покрытие", f"{coverage:.1f}%")
            
            st.markdown("---")
            
            # Отображаем каждый чанк отдельно
            for chunk_idx, chunk in enumerate(chunks):
                render_chunk_visual(chunk, chunk_idx, selected_method)
                if chunk_idx < len(chunks) - 1:  # Добавляем разделитель между чанками
                    st.markdown("---")
                    
        else:
            st.error(f"❌ Не удалось получить результаты чанкования для метода {selected_method}")

def get_chunks_for_text_and_method(text: str, method_name: str, params: Optional[Dict[str, Any]] = None) -> Tuple[List[str], Dict[str, Any]]:
    """Получает чанки для текста и метода с отладочной информацией"""
    
    if params is None:
        params = {}
    
    start_time = time.time()
    
    # Добавляем отладочную информацию
    st.write(f"🔧 **Отладка**: Запуск метода `{method_name}` с параметрами:")
    st.json(params)
    
    try:
        chunker = get_chunker_by_name(method_name, params)
        
        if chunker is None:
            st.error(f"❌ Chunker {method_name} не найден!")
            return [], {}
        
        st.write(f"✅ **Chunker создан**: {type(chunker).__name__}")
        
        # Получаем чанки
        chunks = chunker.chunk_text(text)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Сохраняем параметры chunker'а для отображения
        chunker_params = getattr(chunker, 'get_params', lambda: {})()
        
        # Отладочная информация о результатах
        st.write(f"⏱️ **Время обработки**: {processing_time:.2f} сек")
        st.write(f"📊 **Результат**: {len(chunks)} чанков")
        
        if len(chunks) > 0:
            st.write(f"📏 **Размеры чанков**: {[len(chunk) for chunk in chunks]}")
        
        # Проверяем различные факторы 
        if hasattr(chunker, '_get_embeddings_api'):
            st.write("🧠 **API эмбеддинги**: доступны")
        elif hasattr(chunker, '_load_spacy_model'):
            st.write("🔄 **Fallback**: используется Spacy")
        
        metadata = {
            'processing_time': processing_time,
            'chunker_type': type(chunker).__name__,
            'chunker_params': chunker_params,
            'chunks_count': len(chunks),
            'chunk_sizes': [len(chunk) for chunk in chunks]
        }
        
        return chunks, metadata
        
    except Exception as e:
        st.error(f"❌ **Ошибка в чанковании**: {str(e)}")
        st.exception(e)  # Показываем полный stack trace
        return [], {}

def display_chunking_results(chunks: List[str], metadata: Dict[str, Any], method_name: str):
    """Отображает результаты чанкования с метаданными"""
    st.subheader(f"📋 Результаты для метода: {method_name}")
    
    # Отображаем метаданные
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("⏱️ Время", f"{metadata.get('processing_time', 0):.2f}с")
    
    with col2:
        st.metric("📊 Чанков", metadata.get('chunks_count', 0))
    
    with col3:
        avg_size = sum(metadata.get('chunk_sizes', [0])) / max(len(metadata.get('chunk_sizes', [1])), 1)
        st.metric("📏 Ср. размер", f"{int(avg_size)} симв.")
    
    with col4:
        # Показываем источник эмбеддингов
        if metadata.get('cache_info'):
            cache_info = metadata['cache_info']
            if cache_info['cached_texts'] > 0:
                st.metric("🎯 Эмбеддинги", "Из кэша")
            else:
                st.metric("🔄 Эмбеддинги", "Новые")
        else:
            st.metric("🧠 Эмбеддинги", "API/Spacy")
    
    # Показываем информацию о кэше если доступна
    if metadata.get('cache_info'):
        cache_info = metadata['cache_info']
        with st.expander("🧠 Информация о кэше эмбеддингов"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Кэшированных текстов", cache_info['cached_texts'])
            with col2:
                st.metric("Всего предложений", cache_info['total_sentences'])
            with col3:
                st.metric("Размер кэша", f"{cache_info['cache_size_mb']:.2f} MB")
    
    # Показываем параметры chunker'а
    if metadata.get('chunker_params'):
        with st.expander("⚙️ Использованные параметры"):
            st.json(metadata['chunker_params'])
    
    # Отображаем чанки
    if chunks:
        for i, chunk in enumerate(chunks, 1):
            with st.expander(f"Чанк {i} ({len(chunk)} символов)"):
                st.text(chunk)
    else:
        st.warning("❌ Чанки не созданы") 