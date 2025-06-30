import streamlit as st
from ui.advanced_semantic import get_chunks_for_text_and_method, display_chunking_results

def show_debug_page():
    """Отладочная страница для тестирования семантического чанкования"""
    
    st.title("🔍 Отладка семантического чанкования")
    st.markdown("Страница для быстрого тестирования и отладки алгоритмов с кэшированием эмбеддингов")
    
    # Инициализируем состояние
    if 'cached_chunker' not in st.session_state:
        st.session_state.cached_chunker = None
    if 'last_params' not in st.session_state:
        st.session_state.last_params = None
    if 'test_results' not in st.session_state:
        st.session_state.test_results = None
    if 'comparison_results' not in st.session_state:
        st.session_state.comparison_results = {}
    
    # Информация о кэше (статическая секция)
    cache_container = st.container()
    with cache_container:
        st.markdown("### 🧠 Кэш эмбеддингов")
        col1, col2, col3, col4 = st.columns(4)
        
        if st.session_state.cached_chunker is not None:
            cache_info = st.session_state.cached_chunker.get_cache_info()
            
            with col1:
                st.metric("Кэшированных текстов", cache_info['cached_texts'])
            with col2:
                st.metric("Всего предложений", cache_info['total_sentences'])
            with col3:
                st.metric("Размер кэша", f"{cache_info['cache_size_mb']:.2f} MB")
            with col4:
                if st.button("🗑️ Очистить кэш", key="clear_cache"):
                    st.session_state.cached_chunker.clear_embeddings_cache()
                    st.session_state.test_results = None
                    st.session_state.comparison_results = {}
                    st.success("✅ Кэш очищен!")
                    st.rerun()
        else:
            with col1:
                st.metric("Кэшированных текстов", 0)
            with col2:
                st.metric("Всего предложений", 0)
            with col3:
                st.metric("Размер кэша", "0.00 MB")
            with col4:
                st.info("ℹ️ Кэш пуст")
    
    st.markdown("---")
    
    # Тестовый текст (сохраняем в session_state)
    if 'test_text' not in st.session_state:
        st.session_state.test_text = """Искусственный интеллект - это область компьютерных наук, занимающаяся созданием интеллектуальных машин.
Машинное обучение является подмножеством ИИ. Оно позволяет системам автоматически учиться на данных.
Глубокое обучение использует нейронные сети с множественными слоями. Эти сети способны выучивать сложные паттерны.
Обработка естественного языка помогает компьютерам понимать человеческий язык. Она включает анализ текста и речи.
Компьютерное зрение позволяет машинам "видеть" и интерпретировать изображения. Это важно для автономных автомобилей.
Робототехника объединяет ИИ с физическими системами. Роботы могут выполнять сложные задачи в реальном мире."""
    
    test_text = st.text_area(
        "📝 Тестовый текст:",
        value=st.session_state.test_text,
        height=200,
        key="text_input"
    )
    
    # Обновляем текст в session_state только если он изменился
    if test_text != st.session_state.test_text:
        st.session_state.test_text = test_text
        st.session_state.test_results = None  # Сбрасываем результаты при изменении текста
        st.session_state.comparison_results = {}
    
    # Настройки параметров
    settings_container = st.container()
    with settings_container:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ⚙️ Настройки")
            
            method = st.selectbox(
                "Метод чанкования:",
                ["EnhancedSemanticChunker", "SpacySemanticChunker", "OpenAISemanticChunker"],
                key="method_select"
            )
            
            enhanced_method = st.selectbox(
                "Алгоритм (для Enhanced):",
                ["weighted_cumulative", "optimal_dp", "multi_scale", "topic_aware"],
                key="enhanced_method_select"
            )
            
            similarity_threshold = st.slider("Порог схожести:", 0.1, 0.9, 0.7, 0.1, key="similarity_slider")
            max_chunk_size = st.slider("Макс. размер чанка:", 200, 2000, 1000, 100, key="chunk_size_slider")
            
        with col2:
            st.markdown("### 🔧 Дополнительные параметры")
            
            window_size = st.slider("Размер окна:", 1, 10, 3, key="window_slider")
            min_coherence = st.slider("Мин. связность:", 0.1, 0.9, 0.3, 0.1, key="coherence_slider")
            enable_clustering = st.checkbox("Включить кластеризацию", value=True, key="clustering_check")
            
            api_key = st.text_input("API ключ (оставьте пустым для Spacy):", type="password", key="api_input")
    
    # Собираем параметры
    params = {
        'method': enhanced_method,
        'similarity_threshold': similarity_threshold,
        'max_chunk_size': max_chunk_size,
        'window_size': window_size,
        'min_coherence': min_coherence,
        'enable_clustering': enable_clustering,
        'api_key': api_key,
        'model_name': 'text-embedding-ada-002',
        'timeout': 30,
        'max_retries': 3
    }
    
    # Показываем текущие параметры (без перерисовки результатов)
    with st.expander("📋 Текущие параметры", expanded=False):
        st.json(params)
    
    st.markdown("---")
    
    # Кнопки управления
    control_col1, control_col2, control_col3 = st.columns(3)
    
    with control_col1:
        if st.button("🚀 Запустить тестирование", type="primary", key="main_test"):
            run_main_test(method, params, test_text)
    
    with control_col2:
        if st.button("⚡ Быстрый тест (0.3/0.9)", key="quick_test"):
            run_quick_comparison(params, test_text)
    
    with control_col3:
        if st.button("🧪 Сравнить алгоритмы", key="algo_compare"):
            run_algorithm_comparison(params, test_text)
    
    # Область для отображения результатов (не перерисовывается при изменении параметров)
    results_container = st.container()
    with results_container:
        # Основные результаты тестирования
        if st.session_state.test_results is not None:
            st.markdown("### 📊 Результаты основного тестирования")
            display_chunking_results(
                st.session_state.test_results['chunks'],
                st.session_state.test_results['metadata'],
                st.session_state.test_results['method_name']
            )
        
        # Результаты сравнений
        if st.session_state.comparison_results:
            st.markdown("### 🔍 Результаты сравнений")
            
            for comparison_type, results in st.session_state.comparison_results.items():
                st.markdown(f"#### {comparison_type}")
                
                # Создаем табы для разных результатов
                if len(results) > 1:
                    tab_names = [f"{res['name']} ({res['chunks_count']} чанков)" for res in results]
                    tabs = st.tabs(tab_names)
                    
                    for i, (tab, result) in enumerate(zip(tabs, results)):
                        with tab:
                            st.write(f"**Параметры**: {result['params_summary']}")
                            for j, chunk in enumerate(result['chunks'], 1):
                                with st.expander(f"Чанк {j} ({len(chunk)} символов)"):
                                    st.text(chunk)
                else:
                    result = results[0]
                    st.write(f"**{result['name']}** - {result['chunks_count']} чанков")
                    st.write(f"**Параметры**: {result['params_summary']}")
                    for j, chunk in enumerate(result['chunks'], 1):
                        with st.expander(f"Чанк {j} ({len(chunk)} символов)"):
                            st.text(chunk)
    
    # Информационная секция (статическая)
    st.markdown("---")
    st.markdown("### 💡 Информация о кэшировании")
    
    st.info("""
    **🎯 Преимущества кэширования эмбеддингов:**
    
    - ⚡ **Мгновенные тесты**: изменение параметров не требует пересоздания эмбеддингов
    - 💰 **Экономия API**: одинаковый текст обрабатывается API только один раз
    - 🔄 **Быстрые сравнения**: можно мгновенно сравнивать разные алгоритмы
    - 🧠 **Умный кэш**: автоматически определяет, когда нужны новые эмбеддинги
    - 📋 **Сохранение результатов**: результаты тестов не исчезают при изменении параметров
    
    Кэш и результаты сохраняются в течение всей сессии.
    """)

@st.fragment
def run_main_test(method: str, params: dict, test_text: str):
    """Выполняет основное тестирование без перерисовки всей страницы"""
    
    with st.spinner("Обрабатываем..."):
        if method == "EnhancedSemanticChunker":
            from chunkers.semantic import EnhancedSemanticChunker
            
            # Создаем/переиспользуем кэшированный chunker
            if (st.session_state.cached_chunker is None or 
                not isinstance(st.session_state.cached_chunker, EnhancedSemanticChunker) or
                st.session_state.last_params != params):
                
                st.info("🔄 Создаем новый chunker с обновленными параметрами...")
                st.session_state.cached_chunker = EnhancedSemanticChunker(**params)
                st.session_state.last_params = params.copy()
            else:
                # Обновляем только необходимые параметры
                chunker = st.session_state.cached_chunker
                chunker.similarity_threshold = params['similarity_threshold']
                chunker.max_chunk_size = params['max_chunk_size']
                chunker.window_size = params['window_size']
                chunker.min_coherence = params['min_coherence']
                chunker.enable_clustering = params['enable_clustering']
                chunker.method = params['method']
                
                st.success("⚡ Переиспользуем кэшированный chunker!")
            
            chunks = st.session_state.cached_chunker.chunk_text(test_text)
            
            # Создаем metadata
            cache_info = st.session_state.cached_chunker.get_cache_info()
            metadata = {
                'processing_time': 0.1,
                'chunker_type': type(st.session_state.cached_chunker).__name__,
                'chunker_params': st.session_state.cached_chunker.get_params(),
                'chunks_count': len(chunks),
                'chunk_sizes': [len(chunk) for chunk in chunks],
                'cache_info': cache_info
            }
            
            # Сохраняем результаты в session_state
            st.session_state.test_results = {
                'chunks': chunks,
                'metadata': metadata,
                'method_name': method
            }
            
            st.success(f"✅ Тестирование завершено! Получено {len(chunks)} чанков")
            
        else:
            chunks, metadata = get_chunks_for_text_and_method(test_text, method, params)
            
            st.session_state.test_results = {
                'chunks': chunks,
                'metadata': metadata,
                'method_name': method
            }
            
            if chunks:
                st.success(f"✅ Тестирование завершено! Получено {len(chunks)} чанков")
            else:
                st.error("❌ Не удалось получить результаты")

@st.fragment  
def run_quick_comparison(params: dict, test_text: str):
    """Быстрое сравнение с разными порогами схожести"""
    
    if st.session_state.cached_chunker is None:
        st.warning("⚠️ Сначала запустите основное тестирование для создания кэша")
        return
    
    with st.spinner("Выполняем быстрое сравнение..."):
        results = []
        original_threshold = st.session_state.cached_chunker.similarity_threshold
        
        for threshold in [0.3, 0.9]:
            st.session_state.cached_chunker.similarity_threshold = threshold
            chunks = st.session_state.cached_chunker.chunk_text(test_text)
            
            results.append({
                'name': f"Порог {threshold}",
                'chunks': chunks,
                'chunks_count': len(chunks),
                'params_summary': f"similarity_threshold={threshold}"
            })
        
        # Восстанавливаем оригинальный порог
        st.session_state.cached_chunker.similarity_threshold = original_threshold
        
        st.session_state.comparison_results["⚡ Быстрое сравнение порогов"] = results
        st.success("✅ Быстрое сравнение завершено!")

@st.fragment
def run_algorithm_comparison(params: dict, test_text: str):
    """Сравнение разных алгоритмов"""
    
    if st.session_state.cached_chunker is None:
        st.warning("⚠️ Сначала запустите основное тестирование для создания кэша")
        return
    
    with st.spinner("Сравниваем алгоритмы..."):
        results = []
        original_method = st.session_state.cached_chunker.method
        
        methods_to_test = ["weighted_cumulative", "optimal_dp", "topic_aware"]
        
        for method in methods_to_test:
            st.session_state.cached_chunker.method = method
            chunks = st.session_state.cached_chunker.chunk_text(test_text)
            
            results.append({
                'name': method,
                'chunks': chunks,
                'chunks_count': len(chunks),
                'params_summary': f"method={method}"
            })
        
        # Восстанавливаем оригинальный метод
        st.session_state.cached_chunker.method = original_method
        
        st.session_state.comparison_results["🧪 Сравнение алгоритмов"] = results
        st.success("✅ Сравнение алгоритмов завершено!")

if __name__ == "__main__":
    show_debug_page() 