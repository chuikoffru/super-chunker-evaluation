"""
UI модуль для продвинутого семантического анализа с PSR оценкой
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any

from chunkers.semantic import AdvancedSemanticChunker
from analyzer_advanced import PSRAnalyzer
from sample_texts import SAMPLE_TEXTS

def load_test_texts() -> List[Dict[str, str]]:
    """Загружает тестовые тексты разного размера с названиями"""
    texts = []
    
    # Добавляем примеры из sample_texts.py (это словарь)
    for title, text in SAMPLE_TEXTS.items():
        if isinstance(text, str) and len(text) > 100:
            texts.append({"title": title, "text": text})
    
    # Добавляем тексты разной длины для тестирования масштабируемости
    base_text = """
    Искусственный интеллект представляет собой одну из самых революционных технологий современности.
    Машинное обучение и глубокие нейронные сети позволяют компьютерам выполнять задачи, которые ранее
    требовали человеческого интеллекта. Обработка естественного языка открывает новые возможности
    для анализа и понимания текстов.
    """
    
    # Короткий текст
    texts.append({"title": "Тестовый текст (короткий)", "text": base_text})
    
    # Средний текст
    texts.append({"title": "Тестовый текст (средний)", "text": base_text * 5})
    
    # Длинный текст
    texts.append({"title": "Тестовый текст (длинный)", "text": base_text * 20})
    
    # Очень длинный текст
    texts.append({"title": "Тестовый текст (очень длинный)", "text": base_text * 50})
    
    return texts

def create_chunker(method: str, params: Dict[str, Any]) -> AdvancedSemanticChunker:
    """Создает экземпляр продвинутого семантического чанкера"""
    return AdvancedSemanticChunker(
        method=method,
        similarity_threshold=params.get('similarity_threshold', 0.7),
        max_chunk_size=params.get('max_chunk_size', 1000),
        api_url=params.get('api_url', 'https://api.openai.com/v1/embeddings'),
        api_key=params.get('api_key', ''),
        model_name=params.get('model_name', 'text-embedding-ada-002'),
        timeout=params.get('timeout', 30),
        max_retries=params.get('max_retries', 3),
        window_size=params.get('window_size', 3)
    )

def render_psr_metrics(psr_results: Dict[str, Any]):
    """Отображает PSR метрики в виде красивых карточек"""
    psr_score = psr_results['psr_score']
    
    # Основные PSR оценки
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🚀 Performance", 
            f"{psr_score['performance_score']:.2f}",
            help="Производительность: скорость и эффективность использования ресурсов"
        )
    
    with col2:
        st.metric(
            "📈 Scalability", 
            f"{psr_score['scalability_score']:.2f}",
            help="Масштабируемость: способность обрабатывать большие объемы данных"
        )
    
    with col3:
        st.metric(
            "🎯 Reliability", 
            f"{psr_score['reliability_score']:.2f}",
            help="Надежность: качество и консистентность результатов"
        )
    
    with col4:
        grade_color = {
            "A+": "🟢", "A": "🟢", "B+": "🟡", "B": "🟡", 
            "C+": "🟠", "C": "🟠", "D": "🔴", "F": "🔴"
        }
        st.metric(
            f"{grade_color.get(psr_score['grade'], '⚪')} PSR Score", 
            f"{psr_score['total_psr_score']:.2f} ({psr_score['grade']})",
            help="Общая оценка PSR: Performance + Scalability + Reliability"
        )

def render_detailed_metrics(results: Dict[str, Any]):
    """Отображает детальные метрики в развернутом виде"""
    
    # Performance детали
    with st.expander("📊 Детали производительности"):
        perf = results['performance']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Среднее время выполнения", f"{perf['avg_execution_time']:.3f} сек")
            st.metric("Скорость обработки", f"{perf['chars_per_second']:.0f} симв/сек")
        
        with col2:
            st.metric("Максимальное время", f"{perf['max_execution_time']:.3f} сек")
            st.metric("Использование памяти", f"{perf['avg_memory_usage']:.1f} МБ")
    
    # Scalability детали
    with st.expander("📈 Детали масштабируемости"):
        scale = results['scalability']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Фактор масштабируемости", f"{scale['scalability_factor']:.3f}")
            st.metric("Максимальный размер текста", f"{scale['max_text_size_handled']:.0f} симв")
        
        with col2:
            st.metric("Коэффициент эффективности", f"{scale['efficiency_ratio']:.2f}")
            st.metric("Корреляция размер-время", f"{scale['size_correlation']:.3f}")
    
    # Reliability детали
    with st.expander("🎯 Детали надежности"):
        rel = results['reliability']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Консистентность размеров", f"{rel['chunk_size_consistency']:.3f}")
            st.metric("Точность границ", f"{rel['boundary_accuracy']:.3f}")
        
        with col2:
            st.metric("Семантическая когерентность", f"{rel['semantic_coherence']:.3f}")
            st.metric("Процент успеха", f"{rel['success_rate']*100:.1f}%")

def create_comparison_chart(analyzer: PSRAnalyzer):
    """Создает диаграмму сравнения методов"""
    comparison = analyzer.get_comparison_report()
    
    if not comparison or not comparison.get('methods'):
        st.warning("Нет данных для сравнения")
        return
    
    # Подготавливаем данные для графика
    methods = []
    performance_scores = []
    scalability_scores = []
    reliability_scores = []
    total_scores = []
    
    for method in comparison['methods']:
        result = analyzer.results[method]
        psr = result['psr_score']
        
        methods.append(method)
        performance_scores.append(psr['performance_score'])
        scalability_scores.append(psr['scalability_score'])
        reliability_scores.append(psr['reliability_score'])
        total_scores.append(psr['total_psr_score'])
    
    # Создаем subplot с несколькими графиками
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Performance', 'Scalability', 'Reliability', 'Overall PSR Score'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Performance
    fig.add_trace(
        go.Bar(x=methods, y=performance_scores, name="Performance", 
               marker_color="lightblue"),
        row=1, col=1
    )
    
    # Scalability  
    fig.add_trace(
        go.Bar(x=methods, y=scalability_scores, name="Scalability", 
               marker_color="lightgreen"),
        row=1, col=2
    )
    
    # Reliability
    fig.add_trace(
        go.Bar(x=methods, y=reliability_scores, name="Reliability", 
               marker_color="lightsalmon"),
        row=2, col=1
    )
    
    # Total PSR
    fig.add_trace(
        go.Bar(x=methods, y=total_scores, name="Total PSR", 
               marker_color="lightgoldenrodyellow"),
        row=2, col=2
    )
    
    # Обновляем layout
    fig.update_layout(
        height=600,
        title_text="Сравнение PSR метрик по методам",
        showlegend=False
    )
    
    # Устанавливаем диапазон Y от 0 до 1
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_yaxes(range=[0, 1], row=i, col=j)
    
    st.plotly_chart(fig, use_container_width=True)

def render_advanced_semantic_sidebar():
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

def render_advanced_semantic_page():
    """Главная функция для отрисовки страницы продвинутого семантического анализа"""
    
    st.title("🧠 Продвинутый семантический анализ")
    st.markdown("### Тестирование продвинутых методов семантического чанкования с PSR анализом")
    
    # Инициализируем PSR анализатор в session state
    if 'psr_analyzer' not in st.session_state:
        st.session_state.psr_analyzer = PSRAnalyzer()
    
    # Отрисовываем боковую панель с настройками
    selected_methods, params = render_advanced_semantic_sidebar()
    
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
            return
        
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
    if hasattr(st.session_state.psr_analyzer, 'results') and st.session_state.psr_analyzer.results:
        
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
    if hasattr(st.session_state.psr_analyzer, 'results') and st.session_state.psr_analyzer.results:
        if st.button("🗑️ Очистить результаты"):
            st.session_state.psr_analyzer = PSRAnalyzer()
            st.rerun()

def render_chunk_visual(chunk: str, chunk_index: int, word_count: int, text_id: str = "", method: str = ""):
    """Отображает один чанк в красивом контейнере"""
    # Создаем цветовую схему на основе индекса
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightpink', 'lightgray']
    bg_color = colors[chunk_index % len(colors)]
    
    # Обрезаем текст для предварительного просмотра
    preview_text = chunk[:200] + "..." if len(chunk) > 200 else chunk
    
    st.markdown(f"""
    <div style="
        background-color: {bg_color};
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #2E86AB;
        margin: 5px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    ">
        <div style="
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        ">
            <strong>📄 Чанк {chunk_index + 1}</strong>
            <span style="
                background-color: #2E86AB;
                color: white;
                padding: 2px 8px;
                border-radius: 12px;
                font-size: 12px;
            ">
                {word_count} слов
            </span>
        </div>
        <div style="
            font-family: 'Courier New', monospace;
            font-size: 14px;
            line-height: 1.4;
            color: #333;
        ">
            {preview_text}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Показываем полный текст в expander, если он был обрезан
    if len(chunk) > 200:
        with st.expander(f"Показать полный текст чанка {chunk_index + 1}"):
            # Создаем уникальный ключ с учетом всех контекстов
            unique_key = f"chunk_{text_id}_{method}_{chunk_index}_{abs(hash(chunk)) % 1000000}"
            st.text_area("", value=chunk, height=100, disabled=True, key=unique_key)

def render_chunking_results():
    """Отображает результаты чанкования для всех тестовых текстов"""
    
    if not (hasattr(st.session_state.psr_analyzer, 'results') and st.session_state.psr_analyzer.results):
        st.info("🔍 Запустите PSR анализ, чтобы увидеть результаты чанкования тестовых текстов")
        return
    
    st.header("📋 Результаты чанкования тестовых текстов")
    st.markdown("Просмотрите как каждый метод разбивает тестовые тексты на чанки")
    
    # Загружаем тестовые тексты с названиями
    test_texts = load_test_texts()
    
    # Получаем методы, для которых есть результаты
    available_methods = list(st.session_state.psr_analyzer.results.keys())
    
    # Показываем expander для каждого тестового текста
    for text_idx, text_data in enumerate(test_texts):
        title = text_data["title"]
        text_content = text_data["text"]
        
        # Создаем безопасный идентификатор для ключей
        safe_text_id = f"text_{text_idx}"
        
        # Подсчитываем статистику исходного текста
        char_count = len(text_content)
        word_count = len(text_content.split())
        
        with st.expander(f"📖 {title} ({word_count} слов, {char_count} символов)", expanded=False):
            
            # Показываем превью исходного текста
            st.markdown("**📝 Исходный текст:**")
            preview_text = text_content[:300] + "..." if len(text_content) > 300 else text_content
            st.markdown(f"*{preview_text}*")
            
            if len(text_content) > 300:
                with st.expander("Показать полный исходный текст"):
                    st.text_area("", value=text_content, height=150, disabled=True, 
                               key=f"original_{safe_text_id}_{abs(hash(text_content)) % 1000000}")
            
            st.markdown("---")
            
            # Создаем табы для каждого метода
            if len(available_methods) == 1:
                # Один метод - без табов
                method = available_methods[0]
                st.subheader(f"🔧 Метод: {method.title()}")
                
                # Получаем чанки для этого текста и метода
                chunks = get_chunks_for_text_and_method(text_content, method)
                
                if chunks:
                    st.markdown(f"**Количество чанков:** {len(chunks)}")
                    
                    # Отображаем каждый чанк
                    for chunk_idx, chunk in enumerate(chunks):
                        chunk_word_count = len(chunk.split())
                        render_chunk_visual(chunk, chunk_idx, chunk_word_count, 
                                          text_id=safe_text_id, method=method)
                else:
                    st.warning(f"Не удалось получить результаты чанкования для метода {method}")
            
            else:
                # Несколько методов - используем табы
                method_tabs = st.tabs([f"🔧 {method.title()}" for method in available_methods])
                
                for tab_idx, method in enumerate(available_methods):
                    with method_tabs[tab_idx]:
                        
                        # Получаем чанки для этого текста и метода
                        chunks = get_chunks_for_text_and_method(text_content, method)
                        
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
                                coverage = (total_words / word_count) * 100
                                st.metric("🎯 Покрытие", f"{coverage:.1f}%")
                            
                            st.markdown("---")
                            
                            # Отображаем каждый чанк
                            for chunk_idx, chunk in enumerate(chunks):
                                chunk_word_count = len(chunk.split())
                                render_chunk_visual(chunk, chunk_idx, chunk_word_count, 
                                                  text_id=safe_text_id, method=method)
                        else:
                            st.warning(f"Не удалось получить результаты чанкования для метода {method}")

def get_chunks_for_text_and_method(text: str, method: str) -> List[str]:
    """Получает результаты чанкования для конкретного текста и метода"""
    try:
        # Создаем чанкер с теми же параметрами, что использовались в анализе
        if hasattr(st.session_state, 'last_analysis_params'):
            params = st.session_state.last_analysis_params
        else:
            # Используем параметры по умолчанию, если анализ был запущен в предыдущей сессии
            params = {
                'similarity_threshold': 0.7,
                'max_chunk_size': 1000,
                'api_url': 'https://api.openai.com/v1/embeddings',
                'api_key': '',
                'model_name': 'text-embedding-ada-002',
                'timeout': 30,
                'max_retries': 3,
                'window_size': 3
            }
        
        chunker = create_chunker(method, params)
        chunks = chunker.chunk_text(text)
        return chunks
        
    except Exception as e:
        st.error(f"Ошибка при получении чанков: {e}")
        return [] 