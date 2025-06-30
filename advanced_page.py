"""
Страница продвинутого семантического анализа с PSR оценкой
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any

from chunkers.semantic import AdvancedSemanticChunker
from analyzer_advanced import PSRAnalyzer
from sample_texts import SAMPLE_TEXTS

# Импортируем и запускаем все функции из ui/advanced_semantic.py
from ui.advanced_semantic import (
    load_test_texts, create_chunker, render_psr_metrics, 
    render_detailed_metrics, create_comparison_chart,
    render_chunk_visual, render_chunking_results,
    get_chunks_for_text_and_method
)

def render_sidebar():
    """Отрисовывает настройки для продвинутого семантического анализа"""
    
    st.sidebar.header("⚙️ Настройки анализа")
    
    # Выбор методов для тестирования
    st.sidebar.subheader("Методы чанкования")
    test_cumulative = st.sidebar.checkbox("Cumulative", value=True, 
                                         help="Кумулятивное накопление семантического контекста")
    test_hierarchical = st.sidebar.checkbox("Hierarchical", value=True, 
                                           help="Иерархическое чанкование по уровням")
    test_adaptive = st.sidebar.checkbox("Adaptive", value=True, 
                                       help="Адаптивный порог на основе статистики")
    
    # Общие параметры
    st.sidebar.subheader("Параметры")
    similarity_threshold = st.sidebar.slider("Порог схожести", 0.1, 1.0, 0.7, 0.05)
    max_chunk_size = st.sidebar.slider("Макс. размер чанка", 500, 3000, 1000, 100)
    
    # API настройки
    st.sidebar.subheader("API настройки")
    api_key = st.sidebar.text_input("OpenAI API ключ", type="password", 
                                   help="Оставьте пустым для использования Spacy fallback")
    api_url = st.sidebar.text_input("API URL", value="https://api.openai.com/v1/embeddings")
    model_name = st.sidebar.selectbox("Модель эмбеддингов", 
                                     ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"])
    
    # Дополнительные параметры
    with st.sidebar.expander("Дополнительные параметры"):
        timeout = st.sidebar.slider("Таймаут (сек)", 10, 120, 30)
        max_retries = st.sidebar.slider("Максимум повторов", 1, 10, 3)
        window_size = st.sidebar.slider("Размер окна", 2, 10, 3)
    
    # Возвращаем выбранные методы и параметры
    selected_methods = []
    if test_cumulative:
        selected_methods.append('cumulative')
    if test_hierarchical:
        selected_methods.append('hierarchical')
    if test_adaptive:
        selected_methods.append('adaptive')
    
    params = {
        'similarity_threshold': similarity_threshold,
        'max_chunk_size': max_chunk_size,
        'api_url': api_url,
        'api_key': api_key,
        'model_name': model_name,
        'timeout': timeout,
        'max_retries': max_retries,
        'window_size': window_size
    }
    
    return selected_methods, params

def show_advanced_page():
    """Отображает страницу продвинутого семантического анализа"""
    
    # Основная логика страницы
    st.title("🧠 Продвинутый семантический анализ")
    st.markdown("### Тестирование продвинутых методов семантического чанкования с PSR анализом")

    # Инициализируем PSR анализатор в session state
    if 'psr_analyzer' not in st.session_state:
        st.session_state.psr_analyzer = PSRAnalyzer()

    # Отрисовываем боковую панель с настройками
    selected_methods, params = render_sidebar()

    # Информационная панель
    with st.expander("ℹ️ Информация о методах", expanded=False):
        st.markdown("""
        **📈 Cumulative (Кумулятивный):**
        - Накапливает семантический контекст
        - Каждое предложение сравнивается с усредненным вектором всего чанка
        - Лучше всего для связных текстов и документов
        
        **🏗️ Hierarchical (Иерархический):**
        - Многоуровневое чанкование по структуре документа
        - Разбивка по абзацам → семантическое чанкование → объединение
        - Идеален для структурированных документов
        
        **🎯 Adaptive (Адаптивный):**
        - Динамический порог на основе статистического анализа
        - Автоматически подстраивается под характер текста
        - Универсальный подход для разных типов контента
        
        **📊 PSR Анализ:** Performance + Scalability + Reliability
        """)

    # Кнопка запуска анализа
    if st.button("🚀 Запустить PSR анализ", type="primary", use_container_width=True):
        
        if not selected_methods:
            st.error("Выберите хотя бы один метод для тестирования!")
            st.stop()
        
        # Загружаем тестовые тексты
        test_texts_data = load_test_texts()
        test_texts = [item["text"] for item in test_texts_data]  # Извлекаем только тексты для анализа
        st.info(f"Загружено {len(test_texts)} тестовых текстов")
        
        # Сохраняем параметры для последующего использования
        st.session_state.last_analysis_params = params
        
        # Создаем прогресс-бар
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Анализируем каждый метод
        for i, method in enumerate(selected_methods):
            status_text.text(f"Анализируем {method}...")
            
            try:
                # Создаем чанкер
                chunker = create_chunker(method, params)
                
                # Запускаем PSR анализ
                with st.spinner(f"Анализ {method}..."):
                    results = st.session_state.psr_analyzer.analyze_chunker(
                        chunker, test_texts, method
                    )
                
                progress_bar.progress((i + 1) / len(selected_methods))
                
            except Exception as e:
                st.error(f"Ошибка при анализе {method}: {e}")
                continue
        
        status_text.text("Анализ завершен!")
        st.success("PSR анализ выполнен успешно!")
        
        # Обновляем интерфейс для отображения результатов
        st.rerun()

    # Отображаем результаты, если они есть
    if (hasattr(st.session_state, 'psr_analyzer') and 
        hasattr(st.session_state.psr_analyzer, 'results') and 
        st.session_state.psr_analyzer.results):
        
        st.markdown("---")
        st.header("📊 Результаты PSR анализа")
        
        # Создаем табы для разных методов
        methods = list(st.session_state.psr_analyzer.results.keys())
        
        if len(methods) == 1:
            # Один метод - отображаем без табов
            method = methods[0]
            results = st.session_state.psr_analyzer.results[method]
            
            st.subheader(f"Результаты для {method}")
            render_psr_metrics(results)
            render_detailed_metrics(results)
            
        else:
            # Несколько методов - используем табы
            tabs = st.tabs(methods + ["🏆 Сравнение"])
            
            # Табы для каждого метода
            for i, method in enumerate(methods):
                with tabs[i]:
                    results = st.session_state.psr_analyzer.results[method]
                    render_psr_metrics(results)
                    render_detailed_metrics(results)
            
            # Таб сравнения
            with tabs[-1]:
                st.subheader("🏆 Сравнительный анализ")
                
                # График сравнения
                create_comparison_chart(st.session_state.psr_analyzer)
                
                # Рекомендации
                comparison = st.session_state.psr_analyzer.get_comparison_report()
                if comparison and 'recommendations' in comparison:
                    st.subheader("💡 Рекомендации")
                    for rec in comparison['recommendations']:
                        st.markdown(f"- {rec}")
                
                # Таблица рейтингов
                if comparison and 'rankings' in comparison:
                    st.subheader("🥇 Рейтинги")
                    
                    rating_cols = st.columns(4)
                    
                    for i, (category, rankings) in enumerate(comparison['rankings'].items()):
                        with rating_cols[i]:
                            st.markdown(f"**{category.title()}**")
                            for j, item in enumerate(rankings):
                                emoji = "🥇" if j == 0 else "🥈" if j == 1 else "🥉" if j == 2 else "📍"
                                st.markdown(f"{emoji} {item['method']}: {item['score']:.3f}")
        
        # Показываем результаты чанкования тестовых текстов
        st.markdown("---")
        render_chunking_results()

    # Кнопка очистки результатов
    if (hasattr(st.session_state, 'psr_analyzer') and 
        hasattr(st.session_state.psr_analyzer, 'results') and 
        st.session_state.psr_analyzer.results):
        if st.button("🗑️ Очистить результаты"):
            st.session_state.psr_analyzer = PSRAnalyzer()
            st.rerun() 