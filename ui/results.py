"""
Компонент для отображения результатов чанкования
"""

import streamlit as st
import time
from typing import Dict, Any, List

def render_results_section():
    """Отрисовывает секцию с результатами чанкования"""
    st.header("Результаты чанкования")
    
    if st.session_state.chunking_results:
        _render_results_tabs()
    else:
        st.info("Запустите чанкование для просмотра результатов.")

def run_chunking_process(selected_methods: List[str], method_configs: Dict[str, Dict[str, Any]], 
                        input_text: str) -> bool:
    """
    Выполняет процесс чанкования для выбранных методов
    
    Args:
        selected_methods: Список выбранных методов
        method_configs: Конфигурации методов
        input_text: Входной текст
        
    Returns:
        True если процесс завершился успешно
    """
    from chunkers import CHUNKERS
    
    # Очистка предыдущих результатов
    st.session_state.analyzer.clear_results()
    
    # Прогресс бар и статус
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    success_count = 0
    
    for i, method in enumerate(selected_methods):
        status_text.text(f"Обрабатываем метод: {method}")
        
        try:
            # Выполняем чанкование
            success = _process_single_method(method, method_configs[method], input_text)
            if success:
                success_count += 1
                
        except Exception as e:
            st.error(f"Критическая ошибка при обработке метода {method}: {str(e)}")
        
        progress_bar.progress((i + 1) / len(selected_methods))
    
    # Завершение
    status_text.text("Чанкование завершено!")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()
    
    if success_count > 0:
        st.success(f"Успешно обработано методов: {success_count}/{len(selected_methods)}")
        return True
    else:
        st.error("Ни один метод не был обработан успешно.")
        return False

def _process_single_method(method: str, config: Dict[str, Any], input_text: str) -> bool:
    """Обрабатывает один метод чанкования"""
    from chunkers import CHUNKERS
    
    try:
        # Создаем экземпляр чанкера с параметрами
        chunker_class = CHUNKERS[method]
        
        # Разделяем параметры на конструктор и метод chunk_text
        constructor_params, chunk_method_params = _split_params(method, config)
        
        chunker = chunker_class(**constructor_params)
        
        # Выполняем чанкование
        start_time = time.time()
        chunks = chunker.chunk_text(input_text, **chunk_method_params)
        end_time = time.time()
        
        # Проверяем результат
        if not chunks:
            st.warning(f"Метод {method} вернул пустой результат.")
            return False
        
        # Анализируем результаты
        analysis = st.session_state.analyzer.analyze_chunks(chunks, method)
        analysis['processing_time'] = round(end_time - start_time, 3)
        
        st.session_state.chunking_results[method] = {
            'chunks': chunks,
            'analysis': analysis,
            'config': config
        }
        
        return True
        
    except Exception as e:
        st.error(f"Ошибка при обработке метода {method}: {str(e)}")
        return False

def _split_params(method: str, config: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Разделяет параметры на конструкторные и для метода chunk_text"""
    constructor_params = {}
    chunk_method_params = {}
    
    if method == 'Символы':
        constructor_params = {k: v for k, v in config.items() if k in ['chunk_size', 'chunk_overlap']}
        chunk_method_params = {k: v for k, v in config.items() if k in ['separator']}
    elif method == 'Рекурсивный':
        constructor_params = {k: v for k, v in config.items() if k in ['chunk_size', 'chunk_overlap']}
        chunk_method_params = {k: v for k, v in config.items() if k in ['separators']}
    else:
        # Для остальных методов все параметры идут в конструктор
        constructor_params = config
        chunk_method_params = {}
    
    return constructor_params, chunk_method_params

def _render_results_tabs():
    """Отрисовывает табы с результатами"""
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Сводка", "📈 Графики", "🔍 Детали чанков", "⚙️ Конфигурации"
    ])
    
    with tab1:
        _render_summary_tab()
    
    with tab2:
        _render_charts_tab()
    
    with tab3:
        _render_details_tab()
    
    with tab4:
        _render_configs_tab()

def _render_summary_tab():
    """Отрисовывает таб со сводкой"""
    st.subheader("Сводная таблица результатов")
    
    summary_df = st.session_state.analyzer.get_summary_table()
    
    # Добавляем время обработки
    processing_times = []
    for method in summary_df['Метод']:
        if method in st.session_state.chunking_results:
            processing_times.append(
                st.session_state.chunking_results[method]['analysis']['processing_time']
            )
        else:
            processing_times.append(0)
    
    summary_df['Время обработки (сек)'] = processing_times
    st.dataframe(summary_df, use_container_width=True)
    
    # Рекомендации
    _render_recommendations(summary_df)

def _render_recommendations(summary_df):
    """Отрисовывает рекомендации"""
    st.subheader("💡 Рекомендации")
    
    if len(summary_df) > 1:
        best_stability_idx = summary_df['Стандартное отклонение'].idxmin()
        fastest_idx = summary_df['Время обработки (сек)'].idxmin()
        
        best_method = summary_df.loc[best_stability_idx, 'Метод']
        fastest_method = summary_df.loc[fastest_idx, 'Метод']
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Наиболее стабильные размеры чанков:** {best_method}")
        with col2:
            st.info(f"**Самый быстрый метод:** {fastest_method}")
        
        # Дополнительные метрики
        if len(summary_df) >= 3:
            avg_chunks = summary_df['Количество чанков'].median()
            optimal_methods = summary_df[
                summary_df['Количество чанков'].between(avg_chunks * 0.8, avg_chunks * 1.2)
            ]
            
            if not optimal_methods.empty:
                optimal_method = optimal_methods.iloc[0]['Метод']
                st.success(f"**Оптимальное количество чанков:** {optimal_method}")

def _render_charts_tab():
    """Отрисовывает таб с графиками"""
    st.subheader("Сравнительный анализ")
    
    comparison_chart = st.session_state.analyzer.create_comparison_chart()
    if comparison_chart and comparison_chart.data:
        st.plotly_chart(comparison_chart, use_container_width=True)
    else:
        st.info("Нет данных для отображения графиков")

def _render_details_tab():
    """Отрисовывает таб с деталями чанков"""
    st.subheader("Детальный просмотр чанков")
    
    # Выбор метода для детального просмотра
    method_for_details = st.selectbox(
        "Выберите метод для просмотра:",
        list(st.session_state.chunking_results.keys()),
        help="Выберите метод для детального анализа чанков"
    )
    
    if method_for_details:
        _render_method_details(method_for_details)

def _render_method_details(method: str):
    """Отрисовывает детали для конкретного метода"""
    try:
        # Получаем правильные данные чанков из session_state
        chunks = st.session_state.chunking_results[method]['chunks']
        
        # Создаем график размеров чанков напрямую с правильными данными
        chunk_viz = _create_chunk_size_chart(chunks, method, max_chunks=20)
        st.plotly_chart(chunk_viz, use_container_width=True)
        
        # Просмотр содержимого чанков
        
        st.write(f"**Общее количество чанков:** {len(chunks)}")
        
        # Выбор чанка для просмотра
        chunk_index = st.selectbox(
            "Выберите номер чанка для просмотра:",
            range(1, len(chunks) + 1),
            format_func=lambda x: f"Чанк #{x} ({len(chunks[x-1])} символов)"
        )
        
        if chunk_index:
            selected_chunk = chunks[chunk_index - 1]
            st.text_area(
                f"Содержимое чанка #{chunk_index}:",
                value=selected_chunk,
                height=200,
                help=f"Длина: {len(selected_chunk)} символов"
            )
            
            # Дополнительная информация о чанке
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Символов", len(selected_chunk))
            with col2:
                word_count = len(selected_chunk.split())
                st.metric("Слов", word_count)
            with col3:
                sentence_count = len([s for s in selected_chunk.split('.') if s.strip()])
                st.metric("Предложений", sentence_count)
                
    except Exception as e:
        st.error(f"Ошибка при отображении деталей: {str(e)}")

def _create_chunk_size_chart(chunks: List[str], method_name: str, max_chunks: int = 20):
    """Создает график размеров чанков с правильными данными"""
    import plotly.graph_objects as go
    import plotly.express as px
    
    # Ограничиваем количество отображаемых чанков
    display_chunks = chunks[:max_chunks]
    chunk_sizes = [len(chunk) for chunk in display_chunks]
    chunk_indices = list(range(1, len(display_chunks) + 1))
    
    # Создаем превью чанков (первые 100 символов)
    chunk_previews = [
        chunk[:100] + "..." if len(chunk) > 100 else chunk 
        for chunk in display_chunks
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=chunk_indices,
        y=chunk_sizes,
        text=chunk_previews,
        textposition='none',  # Убираем текст с баров
        hovertemplate='<b>Чанк %{x}</b><br>' +
                     'Размер: %{y} символов<br>' +
                     'Превью: %{text}<br>' +
                     '<extra></extra>',
        marker_color=px.colors.sequential.Viridis
    ))
    
    fig.update_layout(
        title=f"Размеры чанков - {method_name} (первые {len(display_chunks)} чанков)",
        xaxis_title="Номер чанка",
        yaxis_title="Размер чанка (символы)",
        height=400
    )
    
    return fig

def _render_configs_tab():
    """Отрисовывает таб с конфигурациями"""
    st.subheader("Конфигурации методов")
    
    for method, result in st.session_state.chunking_results.items():
        with st.expander(f"Настройки метода: {method}"):
            config = result['config']
            analysis = result['analysis']
            
            col_config, col_metrics = st.columns(2)
            
            with col_config:
                st.write("**Параметры:**")
                for key, value in config.items():
                    if key == 'api_key' and value:
                        # Скрываем API ключ
                        st.write(f"• {key}: {'*' * len(str(value))}")
                    else:
                        st.write(f"• {key}: {value}")
            
            with col_metrics:
                st.write("**Результаты:**")
                st.metric("Количество чанков", analysis['total_chunks'])
                st.metric("Средний размер", f"{analysis['avg_chunk_size']:.1f}")
                st.metric("Время обработки", f"{analysis['processing_time']:.3f} сек") 