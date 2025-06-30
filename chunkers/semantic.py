"""
Семантические методы чанкования текстов на основе эмбеддингов

Этот модуль импортирует все семантические чанкеры из отдельных файлов
для обратной совместимости и удобства импорта.
"""

# Импортируем все семантические чанкеры из отдельных файлов
from .spacy_semantic import SpacySemanticChunker
from .openai_semantic import OpenAISemanticChunker  
from .advanced_semantic import AdvancedSemanticChunker
from .enhanced_semantic import EnhancedSemanticChunker

# Импортируем общие утилиты
from .semantic_utils import safe_cosine_similarity, SPACY_AVAILABLE, SKLEARN_AVAILABLE

# Экспортируем все классы для импорта
__all__ = [
    'SpacySemanticChunker',
    'OpenAISemanticChunker', 
    'AdvancedSemanticChunker',
    'EnhancedSemanticChunker',
    'safe_cosine_similarity',
    'SPACY_AVAILABLE',
    'SKLEARN_AVAILABLE'
] 