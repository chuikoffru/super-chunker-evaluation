"""
Базовый класс для всех чанкеров
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseChunker(ABC):
    """Базовый класс для всех чанкеров"""
    
    @abstractmethod
    def chunk_text(self, text: str, **kwargs) -> List[str]:
        """
        Разбивает текст на чанки
        
        Args:
            text: Исходный текст для разбиения
            **kwargs: Дополнительные параметры для конкретного чанкера
            
        Returns:
            Список строк-чанков
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Возвращает параметры чанкера
        
        Returns:
            Словарь с параметрами чанкера
        """
        pass 