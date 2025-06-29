"""
Реестр доступных чанкеров
"""

from .basic import (
    CharacterChunker,
    RecursiveChunker,
    TokenChunker,
    SentenceChunker,
    ParagraphChunker,
    FixedSizeChunker
)
from .semantic import (
    SpacySemanticChunker,
    OpenAISemanticChunker
)

# Реестр доступных чанкеров
CHUNKERS = {
    'Символы': CharacterChunker,
    'Рекурсивный': RecursiveChunker,
    'Токены': TokenChunker,
    'Предложения': SentenceChunker,
    'Абзацы': ParagraphChunker,
    'Spacy семантический': SpacySemanticChunker,
    'OpenAI семантический': OpenAISemanticChunker,
    'Фиксированный размер': FixedSizeChunker
}

def get_chunker_by_name(name: str):
    """
    Получает класс чанкера по его имени
    
    Args:
        name: Название чанкера
        
    Returns:
        Класс чанкера
        
    Raises:
        KeyError: Если чанкер с указанным именем не найден
    """
    if name not in CHUNKERS:
        available = ', '.join(CHUNKERS.keys())
        raise KeyError(f"Чанкер '{name}' не найден. Доступные: {available}")
    
    return CHUNKERS[name]

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
            'Символы': CharacterChunker,
            'Рекурсивный': RecursiveChunker,
            'Токены': TokenChunker,
            'Фиксированный размер': FixedSizeChunker
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