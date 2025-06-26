"""
Приложение Streamlit для тестирования методов чанкования текстов
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any
import time

from chunkers import CHUNKERS
from analyzer import ChunkingAnalyzer, TextStatistics
from sample_texts import get_sample_text, get_available_samples

# Конфигурация страницы
st.set_page_config(
    page_title="SuperChunker - Тестирование методов чанкования",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Инициализация состояния
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = ChunkingAnalyzer()

if 'chunking_results' not in st.session_state:
    st.session_state.chunking_results = {}



def render_chunker_params(chunker_class, method_name: str) -> Dict[str, Any]:
    """Отрисовывает параметры для выбранного чанкера"""
    params = {}
    
    if method_name == 'Символы':
        params['chunk_size'] = st.sidebar.slider(
            "Размер чанка (символы)", 100, 5000, 1000, key=f"{method_name}_size"
        )
        params['chunk_overlap'] = st.sidebar.slider(
            "Перекрытие", 0, min(500, params['chunk_size']//2), 200, key=f"{method_name}_overlap"
        )
        params['separator'] = st.sidebar.text_input(
            "Разделитель", value="\n\n", key=f"{method_name}_sep"
        )
        
    elif method_name == 'Рекурсивный':
        params['chunk_size'] = st.sidebar.slider(
            "Размер чанка (символы)", 100, 5000, 1000, key=f"{method_name}_size"
        )
        params['chunk_overlap'] = st.sidebar.slider(
            "Перекрытие", 0, min(500, params['chunk_size']//2), 200, key=f"{method_name}_overlap"
        )
        separators_text = st.sidebar.text_area(
            "Разделители (по одному на строку)", 
            value="\n\n\n\n \n", 
            key=f"{method_name}_seps"
        )
        params['separators'] = [s for s in separators_text.split('\n') if s is not None]
        
    elif method_name == 'Токены':
        params['chunk_size'] = st.sidebar.slider(
            "Размер чанка (токены)", 50, 2048, 512, key=f"{method_name}_size"
        )
        params['chunk_overlap'] = st.sidebar.slider(
            "Перекрытие (токены)", 0, min(200, params['chunk_size']//2), 50, key=f"{method_name}_overlap"
        )
        params['model_name'] = st.sidebar.selectbox(
            "Модель токенизации",
            ["gpt-3.5-turbo", "gpt-4", "text-davinci-003", "cl100k_base"],
            key=f"{method_name}_model"
        )
        
    elif method_name == 'Предложения':
        params['sentences_per_chunk'] = st.sidebar.slider(
            "Предложений в чанке", 1, 20, 5, key=f"{method_name}_sentences"
        )
        params['overlap_sentences'] = st.sidebar.slider(
            "Перекрытие (предложения)", 0, min(5, params['sentences_per_chunk']-1), 1, key=f"{method_name}_overlap"
        )
        
    elif method_name == 'Абзацы':
        params['paragraphs_per_chunk'] = st.sidebar.slider(
            "Абзацев в чанке", 1, 10, 3, key=f"{method_name}_paragraphs"
        )
        params['overlap_paragraphs'] = st.sidebar.slider(
            "Перекрытие (абзацы)", 0, min(3, params['paragraphs_per_chunk']-1), 1, key=f"{method_name}_overlap"
        )
        
    elif method_name == 'Семантический':
        params['similarity_threshold'] = st.sidebar.slider(
            "Порог схожести", 0.1, 0.9, 0.7, 0.05, key=f"{method_name}_threshold"
        )
        params['max_chunk_size'] = st.sidebar.slider(
            "Максимальный размер чанка", 500, 3000, 1000, key=f"{method_name}_max_size"
        )
        
    elif method_name == 'Фиксированный размер':
        params['chunk_size'] = st.sidebar.slider(
            "Размер чанка", 100, 5000, 1000, key=f"{method_name}_size"
        )
        params['overlap'] = st.sidebar.slider(
            "Перекрытие", 0, min(500, params['chunk_size']//2), 100, key=f"{method_name}_overlap"
        )
    
    return params

def main():
    """Основная функция приложения"""
    
    st.title("✂️ SuperChunker")
    st.markdown("### Инструмент для тестирования различных методов чанкования текстов")
    
    # Боковая панель с настройками
    st.sidebar.header("Настройки чанкования")
    
    # Выбор методов чанкования
    selected_methods = st.sidebar.multiselect(
        "Выберите методы для тестирования:",
        list(CHUNKERS.keys()),
        default=['Символы', 'Рекурсивный', 'Токены']
    )
    
    if not selected_methods:
        st.warning("Пожалуйста, выберите хотя бы один метод чанкования.")
        return
    
    # Настройки для каждого выбранного метода
    method_configs = {}
    for method in selected_methods:
        st.sidebar.markdown(f"#### {method}")
        chunker_class = CHUNKERS[method]
        params = render_chunker_params(chunker_class, method)
        method_configs[method] = params
    
    # Основная область
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Входной текст")
        
        # Выбор источника текста
        text_source = st.radio(
            "Источник текста:",
            ["Образец текста", "Вставить текст", "Загрузить файл"]
        )
        
        input_text = ""
        
        if text_source == "Образец текста":
            # Выбор типа образца текста
            sample_type = st.selectbox(
                "Выберите тип образца:",
                get_available_samples(),
                index=0
            )
            input_text = get_sample_text(sample_type or "Искусственный интеллект")
            st.text_area("Текст:", value=input_text, height=300, disabled=True)
            
        elif text_source == "Вставить текст":
            input_text = st.text_area("Введите или вставьте текст:", height=300)
            
        elif text_source == "Загрузить файл":
            uploaded_file = st.file_uploader(
                "Выберите текстовый файл",
                type=['txt', 'md', 'rtf']
            )
            if uploaded_file is not None:
                input_text = str(uploaded_file.read(), "utf-8")
                st.text_area("Загруженный текст:", value=input_text, height=300, disabled=True)
        
        # Статистика исходного текста
        if input_text:
            st.subheader("Статистика текста")
            text_stats = TextStatistics.get_text_stats(input_text)
            for key, value in text_stats.items():
                st.metric(key, value)
    
    with col2:
        st.header("Результаты чанкования")
        
        if st.button("🚀 Запустить чанкование", type="primary"):
            if not input_text.strip():
                st.error("Пожалуйста, введите текст для чанкования.")
                return
            
            # Очистка предыдущих результатов
            st.session_state.analyzer.clear_results()
            
            # Выполняем чанкование для каждого выбранного метода
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, method in enumerate(selected_methods):
                status_text.text(f"Обрабатываем метод: {method}")
                
                try:
                    # Создаем экземпляр чанкера с параметрами
                    chunker_class = CHUNKERS[method]
                    config = method_configs[method]
                    
                    # Разделяем параметры на конструктор и метод chunk_text
                    constructor_params = {}
                    chunk_method_params = {}
                    
                    if method == 'Символы':
                        constructor_params = {k: v for k, v in config.items() if k in ['chunk_size', 'chunk_overlap']}
                        chunk_method_params = {k: v for k, v in config.items() if k in ['separator']}
                    elif method == 'Рекурсивный':
                        constructor_params = {k: v for k, v in config.items() if k in ['chunk_size', 'chunk_overlap']}
                        chunk_method_params = {k: v for k, v in config.items() if k in ['separators']}
                    elif method == 'Токены':
                        constructor_params = config  # Все параметры идут в конструктор
                        chunk_method_params = {}
                    else:
                        constructor_params = config  # Для остальных методов все параметры в конструктор
                        chunk_method_params = {}
                    
                    chunker = chunker_class(**constructor_params)
                    
                    # Выполняем чанкование
                    start_time = time.time()
                    chunks = chunker.chunk_text(input_text, **chunk_method_params)
                    end_time = time.time()
                    
                    # Анализируем результаты
                    analysis = st.session_state.analyzer.analyze_chunks(chunks, method)
                    analysis['processing_time'] = round(end_time - start_time, 3)
                    
                    st.session_state.chunking_results[method] = {
                        'chunks': chunks,
                        'analysis': analysis,
                        'config': config
                    }
                    
                except Exception as e:
                    st.error(f"Ошибка при обработке метода {method}: {str(e)}")
                
                progress_bar.progress((i + 1) / len(selected_methods))
            
            status_text.text("Чанкование завершено!")
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()
        
        # Отображение результатов
        if st.session_state.chunking_results:
            
            # Вкладки для разных видов анализа
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Сводка", "📈 Графики", "🔍 Детали чанков", "⚙️ Конфигурации"
            ])
            
            with tab1:
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
                st.subheader("💡 Рекомендации")
                if len(summary_df) > 1:
                    best_method = summary_df.loc[summary_df['Стандартное отклонение'].idxmin(), 'Метод']
                    fastest_method = summary_df.loc[summary_df['Время обработки (сек)'].idxmin(), 'Метод']
                    
                    st.info(f"**Наиболее стабильные размеры чанков:** {best_method}")
                    st.info(f"**Самый быстрый метод:** {fastest_method}")
            
            with tab2:
                st.subheader("Сравнительный анализ")
                comparison_chart = st.session_state.analyzer.create_comparison_chart()
                st.plotly_chart(comparison_chart, use_container_width=True)
                
                st.subheader("Распределение размеров чанков")
                distribution_chart = st.session_state.analyzer.create_size_distribution_chart()
                st.plotly_chart(distribution_chart, use_container_width=True)
            
            with tab3:
                st.subheader("Детальный просмотр чанков")
                
                # Выбор метода для детального просмотра
                method_for_details = st.selectbox(
                    "Выберите метод для просмотра:",
                    list(st.session_state.chunking_results.keys())
                )
                
                if method_for_details:
                    # График размеров чанков
                    chunk_viz = st.session_state.analyzer.create_chunk_visualization(
                        method_for_details, max_chunks=20
                    )
                    st.plotly_chart(chunk_viz, use_container_width=True)
                    
                    # Просмотр содержимого чанков
                    chunks = st.session_state.chunking_results[method_for_details]['chunks']
                    
                    st.write(f"**Общее количество чанков:** {len(chunks)}")
                    
                    # Выбор чанка для просмотра
                    chunk_index = st.selectbox(
                        "Выберите номер чанка для просмотра:",
                        range(1, len(chunks) + 1)
                    )
                    
                    if chunk_index:
                        selected_chunk = chunks[chunk_index - 1]
                        st.text_area(
                            f"Содержимое чанка #{chunk_index} (длина: {len(selected_chunk)} символов):",
                            value=selected_chunk,
                            height=200
                        )
            
            with tab4:
                st.subheader("Конфигурации методов")
                
                for method, result in st.session_state.chunking_results.items():
                    with st.expander(f"Настройки метода: {method}"):
                        config = result['config']
                        analysis = result['analysis']
                        
                        col_config, col_metrics = st.columns(2)
                        
                        with col_config:
                            st.write("**Параметры:**")
                            for key, value in config.items():
                                st.write(f"• {key}: {value}")
                        
                        with col_metrics:
                            st.write("**Результаты:**")
                            st.metric("Количество чанков", analysis['total_chunks'])
                            st.metric("Средний размер", f"{analysis['avg_chunk_size']:.1f}")
                            st.metric("Время обработки", f"{analysis['processing_time']:.3f} сек")

if __name__ == "__main__":
    main() 