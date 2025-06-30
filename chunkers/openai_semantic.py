"""
Семантическое чанкование с использованием OpenAI-совместимого API
"""

import nltk
import time
import os
from typing import List, Dict, Any
import numpy as np
import httpx

from .base import BaseChunker
from .basic import CharacterChunker
from .semantic_utils import safe_cosine_similarity


class OpenAISemanticChunker(BaseChunker):
    """Семантическое чанкование с использованием OpenAI-совместимого API"""
    
    def __init__(self, similarity_threshold: float = 0.7, max_chunk_size: int = 1000,
                 api_url: str = "https://api.openai.com/v1/embeddings",
                 api_key: str = "", model_name: str = "text-embedding-ada-002",
                 timeout: int = 30, max_retries: int = 3):
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.api_url = api_url
        self.api_key = api_key or os.getenv('OPENAI_API_KEY', '')
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
    
    def _make_request(self, texts: List[str]) -> List[List[float]]:
        """Делает запрос к API для получения эмбеддингов"""
        if not self.api_key:
            raise ValueError("""
            API ключ не найден. Установите переменную окружения OPENAI_API_KEY 
            или передайте api_key в конструктор.
            
            Для OpenAI: export OPENAI_API_KEY='your-key-here'
            Для других провайдеров (LocalAI, Ollama и т.д.): укажите соответствующий URL
            """)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "input": texts
        }
        
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(self.api_url, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    
                # Извлекаем эмбеддинги из ответа
                if 'data' in data:
                    embeddings = [item['embedding'] for item in data['data']]
                    return embeddings
                else:
                    raise ValueError(f"Неожиданный формат ответа API: {data}")
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise Exception(f"Ошибка API после {self.max_retries} попыток: {e}")
                else:
                    wait_time = 2 ** attempt  # Экспоненциальная задержка
                    print(f"Попытка {attempt + 1} не удалась: {e}. Повтор через {wait_time}с...")
                    time.sleep(wait_time)
        
        # Если мы дошли сюда, то все попытки не удались
        raise Exception("Не удалось получить эмбеддинги после всех попыток")
    
    def chunk_text(self, text: str, **kwargs) -> List[str]:
        """Разбивает текст на семантические чанки с помощью API эмбеддингов"""
        sentences = nltk.sent_tokenize(text)
        if len(sentences) <= 1:
            return [text]
        
        # Ограничиваем количество предложений для экономии API вызовов
        if len(sentences) > 100:
            print(f"Слишком много предложений ({len(sentences)}). Ограничиваем до 100.")
            sentences = sentences[:100]
        
        # Получаем эмбеддинги через API
        embeddings = self._make_request(sentences)
        
        # Вычисляем схожесть между соседними предложениями
        similarities = []
        for i in range(len(embeddings) - 1):
            vec1 = np.array(embeddings[i])
            vec2 = np.array(embeddings[i + 1])
            sim = safe_cosine_similarity(vec1, vec2)
            similarities.append(sim)
        
        # Находим точки разделения
        split_points = [0]
        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold:
                split_points.append(i + 1)
        split_points.append(len(sentences))
        
        # Создаем чанки
        chunks = []
        for i in range(len(split_points) - 1):
            start, end = split_points[i], split_points[i + 1]
            chunk_text = ' '.join(sentences[start:end])
            
            if len(chunk_text) > self.max_chunk_size:
                sub_chunker = CharacterChunker(
                    chunk_size=self.max_chunk_size, 
                    chunk_overlap=100
                )
                chunks.extend(sub_chunker.chunk_text(chunk_text))
            else:
                chunks.append(chunk_text)
        
        return chunks
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'similarity_threshold': self.similarity_threshold,
            'max_chunk_size': self.max_chunk_size,
            'api_url': self.api_url,
            'model_name': self.model_name,
            'timeout': self.timeout,
            'max_retries': self.max_retries
        } 