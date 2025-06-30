"""
Реестр всех доступных методов чанкования
"""

from typing import Dict, Tuple, Any, Type, Optional
from .base import BaseChunker
from .basic import (
    SentenceChunker, 
    ParagraphChunker,
    CharacterChunker,
    RecursiveChunker,
    TokenChunker,
    FixedSizeChunker
)
from .semantic import (
    OpenAISemanticChunker,
    SpacySemanticChunker,
    AdvancedSemanticChunker,
    EnhancedSemanticChunker
)

# Реестр доступных чанкеров
CHUNKERS = {
    'Символы': FixedSizeChunker,  # Используем FixedSizeChunker для настоящего разбиения по символам
    'Рекурсивный': RecursiveChunker,
    'Токены': TokenChunker,
    'Предложения': SentenceChunker,
    'Абзацы': ParagraphChunker,
    'Spacy семантический': SpacySemanticChunker,
    'OpenAI семантический': OpenAISemanticChunker,
    'Символы (LangChain)': CharacterChunker,  # Переименовываем старый метод
}

# Существующий реестр методов
BASIC_METHODS = {
    'Предложения': ('basic', 'SentenceChunker', {}),
    'Абзацы': ('basic', 'ParagraphChunker', {}),
    'Символы (Fixed)': ('basic', 'FixedSizeChunker', {}),
    'Символы (LangChain)': ('basic', 'CharacterChunker', {}),
    'Рекурсивное': ('basic', 'RecursiveChunker', {}),
    'Токены': ('basic', 'TokenChunker', {}),
}

ADVANCED_SEMANTIC_METHODS = {
    'Weighted Cumulative': ('semantic', 'EnhancedSemanticChunker', {'method': 'weighted_cumulative'}),
    'Optimal Dynamic Programming': ('semantic', 'EnhancedSemanticChunker', {'method': 'optimal_dp'}),
    'Multi-Scale Analysis': ('semantic', 'EnhancedSemanticChunker', {'method': 'multi_scale'}),
    'Topic-Aware Clustering': ('semantic', 'EnhancedSemanticChunker', {'method': 'topic_aware'}),
    'OpenAI Semantic': ('semantic', 'OpenAISemanticChunker', {}),
    'Spacy Semantic': ('semantic', 'SpacySemanticChunker', {}),
    'Advanced Cumulative': ('semantic', 'AdvancedSemanticChunker', {'method': 'cumulative'}),
    'Advanced Hierarchical': ('semantic', 'AdvancedSemanticChunker', {'method': 'hierarchical'}),
    'Advanced Adaptive': ('semantic', 'AdvancedSemanticChunker', {'method': 'adaptive'}),
}

# Объединяем все в один общий реестр
CHUNKER_REGISTRY = {**BASIC_METHODS, **ADVANCED_SEMANTIC_METHODS}

# Словарь классов для создания экземпляров
CHUNKER_CLASSES = {
    'SentenceChunker': SentenceChunker,
    'ParagraphChunker': ParagraphChunker,
    'CharacterChunker': CharacterChunker,
    'RecursiveChunker': RecursiveChunker,
    'TokenChunker': TokenChunker,
    'FixedSizeChunker': FixedSizeChunker,
    'OpenAISemanticChunker': OpenAISemanticChunker,
    'SpacySemanticChunker': SpacySemanticChunker,
    'AdvancedSemanticChunker': AdvancedSemanticChunker,
    'EnhancedSemanticChunker': EnhancedSemanticChunker
}

def get_chunker_by_name(name: str, user_params: Optional[Dict[str, Any]] = None):
    """Создает экземпляр chunker по имени метода с пользовательскими параметрами"""
    if name not in CHUNKER_REGISTRY:
        raise ValueError(f"Неизвестный метод чанкования: {name}")
    
    module_name, class_name, default_params = CHUNKER_REGISTRY[name]
    
    if class_name not in CHUNKER_CLASSES:
        raise ValueError(f"Класс {class_name} не найден в реестре")
    
    chunker_class = CHUNKER_CLASSES[class_name]
    
    # Объединяем базовые параметры с пользовательскими
    final_params = default_params.copy()
    if user_params:
        # Фильтруем только существующие параметры конструктора
        import inspect
        sig = inspect.signature(chunker_class.__init__)
        valid_params = {k: v for k, v in user_params.items() if k in sig.parameters}
        final_params.update(valid_params)
    
    # Создаем экземпляр с объединенными параметрами
    return chunker_class(**final_params)

def list_available_chunkers():
    """
    Возвращает список доступных чанкеров
    
    Returns:
        Список названий доступных чанкеров
    """
    return list(CHUNKERS.keys())

def get_chunkers_by_category():
    """
    Возвращает чанкеры, сгруппированные по категориям
    
    Returns:
        Словарь с категориями чанкеров
    """
    return {
        'Базовые': {
            'Символы': FixedSizeChunker,  # Настоящее символьное разбиение
            'Рекурсивный': RecursiveChunker,
            'Токены': TokenChunker,
            'Символы (LangChain)': CharacterChunker  # Разбиение с разделителями
        },
        'Структурные': {
            'Предложения': SentenceChunker,
            'Абзацы': ParagraphChunker
        },
        'Семантические': {
            'Spacy семантический': SpacySemanticChunker,
            'OpenAI семантический': OpenAISemanticChunker
        }
    }

# Выбираем методы в зависимости от контекста
def get_basic_methods():
    return BASIC_METHODS

def get_advanced_methods():
    return ADVANCED_SEMANTIC_METHODS

def get_all_methods():
    return CHUNKER_REGISTRY 