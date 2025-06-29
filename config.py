"""
Конфигурация приложения SuperChunker
"""

import streamlit as st

def setup_page_config():
    """Настраивает конфигурацию страницы Streamlit"""
    st.set_page_config(
        page_title="SuperChunker - Тестирование методов чанкования",
        page_icon="✂️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def initialize_session_state():
    """Инициализирует состояние сессии"""
    from analyzer import ChunkingAnalyzer
    
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = ChunkingAnalyzer()

    if 'chunking_results' not in st.session_state:
        st.session_state.chunking_results = {}

# Дефолтные методы чанкования для выбора
DEFAULT_METHODS = ['Символы', 'Рекурсивный', 'Токены', 'Spacy семантический']

# Настройки файлов для загрузки
ALLOWED_FILE_TYPES = ['txt', 'md', 'rtf']

# Максимальные значения параметров
MAX_CHUNK_SIZE = 5000
MAX_OVERLAP = 500
MAX_SENTENCES_PER_CHUNK = 20
MAX_PARAGRAPHS_PER_CHUNK = 10
MAX_SIMILARITY_THRESHOLD = 0.9
MIN_SIMILARITY_THRESHOLD = 0.1
MAX_TIMEOUT = 60
MAX_RETRIES = 5

# Модели для различных чанкеров
SPACY_MODELS = ["ru_core_news_md", "ru_core_news_lg", "en_core_web_md"]
TOKENIZER_MODELS = ["gpt-3.5-turbo", "gpt-4", "text-davinci-003", "cl100k_base"]
EMBEDDING_MODELS = ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"]

# Разделители для рекурсивного чанкера
DEFAULT_SEPARATORS = "\n\n\n\n \n" 