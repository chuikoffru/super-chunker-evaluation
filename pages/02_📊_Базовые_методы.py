"""
Страница детального анализа базовых методов чанкования
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any
import pandas as pd

from chunkers.registry import CHUNKERS
from analyzer import ChunkingAnalyzer
from sample_texts import SAMPLE_TEXTS

# Настройка страницы
st.set_page_config(
    page_title="Базовые методы",
    page_icon="📊",
    layout="wide"
)

def render_method_comparison():
    """Отображает сравнительный анализ всех базовых методов"""
    
    st.title("📊 Детальный анализ базовых методов чанкования")
    st.markdown("### Сравнение Character, Recursive, Token и Semantic методов")
    
    # Боковая панель с настройками
    st.sidebar.header("⚙️ Настройки анализа")
    
    # Выбор тестового текста
    st.sidebar.subheader("Тестовый текст")
    text_options = list(SAMPLE_TEXTS.keys()) + ["Собственный текст"]
    selected_text_option = st.sidebar.selectbox("Выберите текст:", text_options)
    
    if selected_text_option == "Собственный текст":
        test_text = st.sidebar.text_area("Введите ваш текст:", height=150, 
                                        placeholder="Вставьте текст для анализа...")
        if not test_text.strip():
            st.info("👈 Выберите готовый текст или введите свой в боковой панели")
            return
    else:
        test_text = SAMPLE_TEXTS[selected_text_option]
    
    # Выбор методов
    st.sidebar.subheader("Методы чанкования")
    available_methods = list(CHUNKERS.keys())
    selected_methods = []
    
    for method in available_methods:
        if st.sidebar.checkbox(method.replace('_', ' ').title(), value=True):
            selected_methods.append(method)
    
    if not selected_methods:
        st.warning("Выберите хотя бы один метод для анализа")
        return
    
    # Общие параметры
    st.sidebar.subheader("Параметры")
    chunk_size = st.sidebar.slider("Размер чанка", 100, 2000, 500, 50)
    chunk_overlap = st.sidebar.slider("Перекрытие", 0, 200, 50, 10)
    
    # Кнопка запуска анализа
    if st.sidebar.button("🚀 Запустить анализ", type="primary"):
        
        with st.spinner("Выполняется анализ..."):
            # Инициализируем анализатор
            analyzer = ChunkingAnalyzer()
            
            # Результаты для каждого метода
            results = {}
            
            for method in selected_methods:
                try:
                    # Создаем чанкер
                    chunker_class = CHUNKERS[method]
                    
                    # Настраиваем параметры в зависимости от метода
                    if method == 'Символы':  # CharacterChunker
                        chunker = chunker_class(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                    elif method == 'Рекурсивный':  # RecursiveChunker
                        chunker = chunker_class(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                    elif method == 'Токены':  # TokenChunker
                        chunker = chunker_class(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                    elif method == 'Spacy семантический':  # SpacySemanticChunker
                        chunker = chunker_class(
                            similarity_threshold=0.7,
                            max_chunk_size=chunk_size
                        )
                    elif method == 'OpenAI семантический':  # OpenAISemanticChunker
                        chunker = chunker_class(
                            similarity_threshold=0.7,
                            max_chunk_size=chunk_size
                        )
                    elif method == 'Фиксированный размер':  # FixedSizeChunker
                        chunker = chunker_class(
                            chunk_size=chunk_size
                        )
                    else:
                        # Для методов без специальных параметров (Предложения, Абзацы)
                        chunker = chunker_class()
                    
                    # Выполняем чанкование
                    chunks = chunker.chunk_text(test_text)
                    
                    # Анализируем результаты
                    analysis = analyzer.analyze_chunks(chunks, method)
                    results[method] = {
                        'chunks': chunks,
                        'analysis': analysis
                    }
                    
                except Exception as e:
                    st.error(f"Ошибка при анализе метода {method}: {e}")
                    continue
        
        # Сохраняем результаты в session state
        st.session_state.comparison_results = results
        st.session_state.test_text = test_text
        st.success("Анализ завершен!")
    
    # Отображаем результаты, если они есть
    if hasattr(st.session_state, 'comparison_results') and st.session_state.comparison_results:
        render_detailed_results()

def render_detailed_results():
    """Отображает детальные результаты сравнения"""
    
    results = st.session_state.comparison_results
    test_text = st.session_state.test_text
    
    st.markdown("---")
    st.header("📈 Результаты анализа")
    
    # Общая статистика
    st.subheader("📋 Общая статистика")
    
    stats_data = []
    for method, data in results.items():
        analysis = data['analysis']
        stats_data.append({
            'Метод': method.replace('_', ' ').title(),
            'Количество чанков': analysis['total_chunks'],
            'Средний размер': f"{analysis['avg_chunk_size']:.0f}",
            'Мин. размер': analysis['min_chunk_size'],
            'Макс. размер': analysis['max_chunk_size'],
            'Стандартное отклонение': f"{analysis['std_chunk_size']:.1f}"
        })
    
    df = pd.DataFrame(stats_data)
    st.dataframe(df, use_container_width=True)
    
    # Графики сравнения
    st.subheader("📊 Графики сравнения")
    
    # График распределения размеров
    fig_sizes = go.Figure()
    
    for method, data in results.items():
        chunks = data['chunks']
        chunk_sizes = [len(chunk) for chunk in chunks]
        
        fig_sizes.add_trace(go.Box(
            y=chunk_sizes,
            name=method.replace('_', ' ').title(),
            boxpoints='outliers'
        ))
    
    fig_sizes.update_layout(
        title="Распределение размеров чанков",
        xaxis_title="Метод",
        yaxis_title="Размер чанка (символы)",
        height=400
    )
    
    st.plotly_chart(fig_sizes, use_container_width=True)
    
    # График количества чанков vs качества
    methods = list(results.keys())
    chunk_counts = [results[m]['analysis']['total_chunks'] for m in methods]
    avg_sizes = [results[m]['analysis']['avg_chunk_size'] for m in methods]
    std_devs = [results[m]['analysis']['std_chunk_size'] for m in methods]
    
    fig_scatter = go.Figure()
    
    fig_scatter.add_trace(go.Scatter(
        x=chunk_counts,
        y=std_devs,
        mode='markers+text',
        text=[m.replace('_', ' ').title() for m in methods],
        textposition="top center",
        marker=dict(
            size=[s/20 for s in avg_sizes],
            color=['blue', 'green', 'red', 'orange'][:len(methods)],
            opacity=0.7
        ),
        name="Методы"
    ))
    
    fig_scatter.update_layout(
        title="Количество чанков vs Консистентность размеров",
        xaxis_title="Количество чанков",
        yaxis_title="Стандартное отклонение размеров",
        height=400
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Детальный просмотр чанков
    st.subheader("🔍 Детальный просмотр чанков")
    
    # Выбор метода для детального просмотра
    selected_method = st.selectbox(
        "Выберите метод для детального просмотра:",
        list(results.keys()),
        format_func=lambda x: x.replace('_', ' ').title() if x else ""
    )
    
    if selected_method and selected_method in results:
        chunks = results[selected_method]['chunks']
        
        st.markdown(f"**Метод:** {selected_method.replace('_', ' ').title()}")
        st.markdown(f"**Количество чанков:** {len(chunks)}")
        
        # Показываем каждый чанк
        for i, chunk in enumerate(chunks):
            word_count = len(chunk.split())
            char_count = len(chunk)
            
            with st.expander(f"📄 Чанк {i+1} ({word_count} слов, {char_count} символов)"):
                st.text_area(
                    f"Содержимое чанка {i+1}",
                    value=chunk,
                    height=100,
                    disabled=True,
                    key=f"detailed_chunk_{selected_method}_{i}",
                    label_visibility="collapsed"
                )
    
    # Кнопка очистки результатов
    if st.button("🗑️ Очистить результаты"):
        if hasattr(st.session_state, 'comparison_results'):
            del st.session_state.comparison_results
        if hasattr(st.session_state, 'test_text'):
            del st.session_state.test_text
        st.rerun()

# Основная логика страницы
render_method_comparison() 