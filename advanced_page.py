"""
Страница продвинутого семантического чанкования
"""

import streamlit as st
import time
import pandas as pd
import hashlib
import json
from typing import Dict, Any, List

# Импортируем функции из ui/advanced_semantic.py
from ui.advanced_semantic import (
    load_test_texts, 
    render_chunking_results
)

# Импортируем функцию создания chunker'ов
from chunkers.registry import get_chunker_by_name

def get_params_hash(params: Dict[str, Any]) -> str:
    """Создает хэш параметров для отслеживания изменений"""
    # Убираем API ключ из хэша для безопасности, но включаем остальные параметры
    params_for_hash = {k: v for k, v in params.items() if k != 'api_key'}
    # Преобразуем в строку и создаем хэш
    params_str = json.dumps(params_for_hash, sort_keys=True)
    return hashlib.md5(params_str.encode()).hexdigest()

# Названия семантических методов для UI
SEMANTIC_METHODS = {
    'Weighted Cumulative': 'Weighted Cumulative - улучшенное кумулятивное с взвешенным усреднением',
    'Optimal Dynamic Programming': 'Optimal DP - глобально оптимальное чанкование через ДП',
    'Multi-Scale Analysis': 'Multi-Scale - многоуровневый анализ с иерархией',
    'Topic-Aware Clustering': 'Topic-Aware - тематическая кластеризация',
    'Advanced Cumulative': 'Advanced Cumulative - улучшенное кумулятивное (v1)',
    'Advanced Hierarchical': 'Advanced Hierarchical - иерархическое (v1)',
    'Advanced Adaptive': 'Advanced Adaptive - адаптивное (v1)',
    'OpenAI Semantic': 'OpenAI Semantic - базовый семантический',
    'Spacy Semantic': 'Spacy Semantic - Spacy семантический',
}

def render_sidebar():
    """Рендерит sidebar с выбором методов и настройками"""
    with st.sidebar:
        st.markdown("## 🧠 Продвинутое чанкование")
        
        # Выбор методов
        st.markdown("### Выбор методов")
        selected_methods = []
        
        # Группируем методы для удобства
        st.markdown("**🌟 Революционные методы (v2):**")
        for method in ['Weighted Cumulative', 'Optimal Dynamic Programming', 
                      'Multi-Scale Analysis', 'Topic-Aware Clustering']:
            if st.checkbox(SEMANTIC_METHODS[method], key=method):
                selected_methods.append(method)
        
        st.markdown("**🔧 Базовые продвинутые методы (v1):**")
        for method in ['Advanced Cumulative', 'Advanced Hierarchical', 'Advanced Adaptive']:
            if st.checkbox(SEMANTIC_METHODS[method], key=method):
                selected_methods.append(method)
        
        st.markdown("**🏛️ Классические семантические методы:**")
        for method in ['OpenAI Semantic', 'Spacy Semantic']:
            if st.checkbox(SEMANTIC_METHODS[method], key=method):
                selected_methods.append(method)
        
        # Общие настройки
        st.markdown("---")
        st.markdown("### ⚙️ Настройки API")
        
        with st.expander("🔑 API Конфигурация"):
            api_key = st.text_input(
                "OpenAI API Key", 
                type="password",
                help="Требуется для семантических методов с OpenAI"
            )
            
            model_name = st.selectbox(
                "Модель для эмбеддингов",
                ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"],
                help="Модель OpenAI для создания эмбеддингов"
            )
            
            timeout = st.slider("Timeout (сек)", 10, 60, 30)
            max_retries = st.slider("Количество попыток", 1, 5, 3)
        
        # Настройки для революционных методов
        st.markdown("---")
        with st.expander("⚡ Настройки революционных методов"):
            window_size = st.slider(
                "Размер окна анализа",
                min_value=2, max_value=10, value=3,
                help="Размер скользящего окна для анализа контекста"
            )
            
            min_coherence = st.slider(
                "Минимальная связность",
                min_value=0.0, max_value=1.0, value=0.3, step=0.05,
                help="Минимальная внутренняя связность чанка"
            )
            
            enable_clustering = st.checkbox(
                "Включить кластеризацию",
                value=True,
                help="Использовать тематическую кластеризацию в Topic-Aware методе"
            )
        
        # Стандартные настройки чанкования
        st.markdown("---")
        with st.expander("📏 Стандартные настройки"):
            max_chunk_size = st.slider("Макс. размер чанка", 500, 3000, 1000)
            similarity_threshold = st.slider("Порог схожести", 0.1, 0.9, 0.7, 0.05)
        
        # Информация об автоматическом сбросе результатов
        st.markdown("---")
        st.info("💡 **Автообновление**: Результаты автоматически сбрасываются при изменении любых настроек")
        
        return {
            'selected_methods': selected_methods,
            'api_key': api_key,
            'model_name': model_name,
            'timeout': timeout,
            'max_retries': max_retries,
            'max_chunk_size': max_chunk_size,
            'similarity_threshold': similarity_threshold,
            'window_size': window_size,
            'min_coherence': min_coherence,
            'enable_clustering': enable_clustering,
        }

def show_advanced_page():
    """Главная функция страницы продвинутого анализа"""
    
    st.title("🧠 Продвинутое семантическое чанкование")
    st.markdown("---")
    
    # Отрисовываем боковую панель с настройками
    params = render_sidebar()
    
    # Проверяем изменения параметров и очищаем результаты при необходимости
    current_params_hash = get_params_hash(params)
    
    # Инициализируем или проверяем хэш параметров
    if 'params_hash' not in st.session_state:
        st.session_state.params_hash = current_params_hash
    elif st.session_state.params_hash != current_params_hash:
        # Параметры изменились - очищаем результаты
        if 'chunking_results' in st.session_state:
            del st.session_state.chunking_results
            st.warning("🔄 Настройки изменились! Предыдущие результаты очищены. Нажмите '🚀 Запустить чанкование' для получения новых результатов с обновленными параметрами.")
        st.session_state.params_hash = current_params_hash

    # Информационная панель
    with st.expander("ℹ️ О продвинутых методах чанкования", expanded=False):
        st.markdown("""
        **Продвинутые методы семантического чанкования** используют современные алгоритмы для интеллектуального разбиения текста:
        
        - 🌟 **Weighted Cumulative**: Экспоненциальное взвешенное усреднение с sliding window анализом
        - 🔍 **Optimal DP**: Глобально оптимальное чанкование через динамическое программирование  
        - 🏗️ **Multi-Scale**: Иерархический анализ на уровне документа, секций и предложений
        - 🎯 **Topic-Aware**: Тематическая кластеризация с использованием машинного обучения
        
        Каждый метод оптимизирован для разных типов текстов и задач.
        """)
    
    # Отображение текущих параметров 
    if params['selected_methods']:
        with st.expander("⚙️ Текущие настройки", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔧 Основные параметры:**")
                st.write(f"• Методы: {', '.join(params['selected_methods'])}")
                st.write(f"• Макс. размер чанка: {params['max_chunk_size']}")
                st.write(f"• Порог схожести: {params['similarity_threshold']}")
                st.write(f"• Timeout: {params['timeout']}с")
                st.write(f"• Попыток: {params['max_retries']}")
            
            with col2:
                st.markdown("**⚡ Революционные настройки:**")
                st.write(f"• Размер окна: {params['window_size']}")
                st.write(f"• Мин. связность: {params['min_coherence']}")
                st.write(f"• Кластеризация: {'Вкл' if params['enable_clustering'] else 'Выкл'}")
                st.write(f"• Модель: {params['model_name']}")
                api_status = "Установлен" if params['api_key'] else "Не установлен"
                st.write(f"• API ключ: {api_status}")

    # Основная кнопка запуска чанкования
    if st.button("🚀 Запустить чанкование", type="primary", use_container_width=True):
        
        if not params['selected_methods']:
            st.error("Выберите хотя бы один метод для тестирования!")
            st.stop()
        
        # Загружаем тестовые тексты
        test_texts = load_test_texts()
        
        # Создаем progress bar и status text
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Обрабатываем каждый метод
        results = {}
        
        for i, method in enumerate(params['selected_methods']):
            status_text.text(f"Обрабатываем {method}...")
            
            try:
                # Создаем chunker с пользовательскими параметрами
                chunker = get_chunker_by_name(method, params)
                
                # Получаем параметры chunker'а для сохранения
                chunker_params = chunker.get_params() if hasattr(chunker, 'get_params') else {}
                
                if chunker:
                    # Обрабатываем тестовые тексты
                    method_results = []
                    for text_item in test_texts:
                        chunks = chunker.chunk_text(text_item["text"])
                        method_results.append({
                            'text_name': text_item["name"],
                            'original_text': text_item["text"],
                            'chunks': chunks,
                            'chunk_count': len(chunks),
                            'avg_chunk_size': sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0,
                            'total_chars': len(text_item["text"]),
                            'chunks_coverage': sum(len(chunk) for chunk in chunks) / len(text_item["text"]) if text_item["text"] else 0,
                            'chunker_params': chunker_params.copy()  # Сохраняем параметры для отладки
                        })
                    
                    results[method] = method_results
                
                progress_bar.progress((i + 1) / len(params['selected_methods']))
                
            except Exception as e:
                st.error(f"Ошибка при обработке {method}: {e}")
        
        status_text.text("Чанкование завершено!")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
        
        # Сохраняем результаты в session state
        st.session_state.chunking_results = results

    # Отображаем результаты если они есть
    if 'chunking_results' in st.session_state and st.session_state.chunking_results:
        
        results = st.session_state.chunking_results
        methods = list(results.keys())
        
        st.markdown("---")
        st.markdown("## 📊 Результаты чанкования")
        
        if len(methods) == 1:
            # Один метод - отображаем результаты
            method = methods[0]
            st.subheader(f"📈 Результаты для метода: {method}")
            render_method_results(results[method])
            
        else:
            # Несколько методов - создаем табы
            tabs = st.tabs(["🏆 Сравнение"] + methods)
            
            # Таб сравнения (первый)
            with tabs[0]:
                st.subheader("🏆 Сравнительный анализ методов")
                render_comparison_table(results)
            
            # Табы для каждого метода
            for i, method in enumerate(methods):
                with tabs[i + 1]:
                    st.subheader(f"📈 Детали для {method}")
                    render_method_results(results[method])
        
        # Кнопка для сброса результатов
        if st.button("🗑️ Очистить результаты"):
            if 'chunking_results' in st.session_state:
                del st.session_state.chunking_results
            st.rerun()
    
    # Раздел с результатами чанкования тестовых текстов
    render_chunking_results()

def render_method_results(method_results: List[Dict]):
    """Отображает результаты чанкования для одного метода"""
    
    # Общая статистика
    total_texts = len(method_results)
    total_chunks = sum(result['chunk_count'] for result in method_results)
    avg_chunks_per_text = total_chunks / total_texts if total_texts > 0 else 0
    avg_chunk_size = sum(result['avg_chunk_size'] for result in method_results) / total_texts if total_texts > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Обработано текстов", total_texts)
    with col2:
        st.metric("Всего чанков", total_chunks)
    with col3:
        st.metric("Среднее чанков/текст", f"{avg_chunks_per_text:.1f}")
    with col4:
        st.metric("Средний размер чанка", f"{avg_chunk_size:.0f} символов")
    
    # Детали по каждому тексту
    st.markdown("### 📝 Детали по текстам")
    
    for result in method_results:
        with st.expander(f"📄 {result['text_name']} ({result['chunk_count']} чанков)"):
            
            # Статистика текста
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Размер текста", f"{result['total_chars']} символов")
            with col2:
                st.metric("Количество чанков", result['chunk_count'])
            with col3:
                st.metric("Покрытие", f"{result['chunks_coverage']:.1%}")
            
            # Отображение параметров chunker'а (если есть)
            if 'chunker_params' in result and result['chunker_params']:
                st.markdown("**Параметры чанкования:**")
                params_text = ", ".join([f"{k}: {v}" for k, v in result['chunker_params'].items()])
                st.code(params_text)
            
            # Отображение чанков
            st.markdown("**Чанки:**")
            for i, chunk in enumerate(result['chunks'], 1):
                st.code(f"Чанк {i} ({len(chunk)} символов):\n{chunk[:200]}{'...' if len(chunk) > 200 else ''}")

def render_comparison_table(results: Dict[str, List[Dict]]):
    """Отображает сравнительную таблицу результатов"""
    
    # Собираем статистику по методам
    comparison_data = []
    
    for method, method_results in results.items():
        total_texts = len(method_results)
        total_chunks = sum(result['chunk_count'] for result in method_results)
        avg_chunks_per_text = total_chunks / total_texts if total_texts > 0 else 0
        avg_chunk_size = sum(result['avg_chunk_size'] for result in method_results) / total_texts if total_texts > 0 else 0
        avg_coverage = sum(result['chunks_coverage'] for result in method_results) / total_texts if total_texts > 0 else 0
        
        comparison_data.append({
            'Метод': method,
            'Всего чанков': total_chunks,
            'Среднее чанков/текст': f"{avg_chunks_per_text:.1f}",
            'Средний размер чанка': f"{avg_chunk_size:.0f}",
            'Среднее покрытие': f"{avg_coverage:.1%}"
        })
    
    # Создаем DataFrame и отображаем
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True)
    
    # Дополнительная информация о параметрах
    st.markdown("### 🔧 Параметры чанкования")
    
    # Показываем параметры для каждого метода
    for method, method_results in results.items():
        if method_results and 'chunker_params' in method_results[0]:
            params = method_results[0]['chunker_params']
            
            with st.expander(f"Параметры для {method}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'similarity_threshold' in params:
                        st.metric("Порог схожести", params['similarity_threshold'])
                    if 'max_chunk_size' in params:
                        st.metric("Макс. размер чанка", params['max_chunk_size'])
                    if 'method' in params:
                        st.metric("Алгоритм", params['method'])
                
                with col2:
                    if 'window_size' in params:
                        st.metric("Размер окна", params['window_size'])
                    if 'min_coherence' in params:
                        st.metric("Мин. связность", f"{params['min_coherence']:.2f}")
                    if 'enable_clustering' in params:
                        st.metric("Кластеризация", "Вкл" if params['enable_clustering'] else "Выкл")
    
    # Графики сравнения
    st.markdown("### 📊 Графическое сравнение")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # График количества чанков
        st.bar_chart(df.set_index('Метод')['Всего чанков'])
        st.caption("Общее количество чанков по методам")
    
    with col2:
        # График среднего размера чанка
        df_chart = df.copy()
        df_chart['Размер чанка'] = df_chart['Средний размер чанка'].str.replace(' символов', '').astype(float)
        st.bar_chart(df_chart.set_index('Метод')['Размер чанка'])
        st.caption("Средний размер чанка по методам")

 