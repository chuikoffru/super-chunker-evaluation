"""
Пакет методов чанкования текстов SuperChunker

Этот пакет содержит различные методы для разбиения длинных текстов
на более короткие фрагменты (чанки) с разными стратегиями.

Модули:
    base: Базовый класс для всех чанкеров
    basic: Базовые методы чанкования (по символам, токенам и т.д.)
    semantic: Семантические методы чанкования (на основе эмбеддингов)
    registry: Реестр и утилиты для работы с чанкерами
"""

from .base import BaseChunker
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
from .registry import (
    CHUNKERS,
    get_chunker_by_name,
    list_available_chunkers,
    get_chunkers_by_category
)

__version__ = "1.1.0"
__author__ = "SuperChunker Team"

# Экспортируемые классы и функции
__all__ = [
    # Базовый класс
    'BaseChunker',
    
    # Базовые чанкеры
    'CharacterChunker',
    'RecursiveChunker', 
    'TokenChunker',
    'SentenceChunker',
    'ParagraphChunker',
    'FixedSizeChunker',
    
    # Семантические чанкеры
    'SpacySemanticChunker',
    'OpenAISemanticChunker',
    
    # Реестр и утилиты
    'CHUNKERS',
    'get_chunker_by_name',
    'list_available_chunkers',
    'get_chunkers_by_category'
] 