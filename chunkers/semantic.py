"""
Семантические методы чанкования текстов на основе эмбеддингов
"""

import nltk
import time
import os
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import httpx

# Проверяем доступность spacy
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

from .base import BaseChunker
from .basic import CharacterChunker

class SpacySemanticChunker(BaseChunker):
    """Семантическое чанкование с использованием Spacy эмбеддингов"""
    
    def __init__(self, similarity_threshold: float = 0.7, max_chunk_size: int = 1000, 
                 model_name: str = "ru_core_news_md"):
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.model_name = model_name
        self.nlp = None
        
    def _load_model(self):
        """Загружает Spacy модель с эмбеддингами"""
        if self.nlp is None:
            if not SPACY_AVAILABLE:
                raise ImportError("Spacy не установлен. Используйте: pip install spacy")
            
            try:
                self.nlp = spacy.load(self.model_name)
                
                # Проверяем, что модель поддерживает векторы
                test_doc = self.nlp("тест")
                if not test_doc.vector.any():
                    # Пробуем другие модели с векторами
                    fallback_models = ["ru_core_news_lg", "ru_core_news_sm", "en_core_web_md"]
                    for model in fallback_models:
                        try:
                            self.nlp = spacy.load(model)
                            test_doc = self.nlp("test")
                            if test_doc.vector.any():
                                print(f"Загружена fallback модель: {model}")
                                break
                        except OSError:
                            continue
                    else:
                        raise ValueError("Не найдена Spacy модель с поддержкой векторов")
                        
            except OSError:
                # Подсказка для пользователя
                error_msg = f"""
                Модель {self.model_name} не найдена. Установите её:
                
                Для русского языка:
                python -m spacy download ru_core_news_md
                
                Для английского (fallback):
                python -m spacy download en_core_web_md
                """
                raise OSError(error_msg)
    
    def _get_sentence_embedding(self, sentence: str):
        """Получает эмбеддинг предложения через Spacy"""
        if self.nlp is None:
            self._load_model()
            
        if self.nlp is None:
            raise RuntimeError("Не удалось загрузить модель Spacy")
            
        doc = self.nlp(sentence)
        
        if doc.vector.any():
            return doc.vector
        else:
            # Fallback: усредняем векторы слов
            word_vectors = [token.vector for token in doc if token.has_vector and not token.is_stop]
            if word_vectors:
                return np.array(word_vectors).mean(axis=0)
            else:
                # Возвращаем нулевой вектор соответствующей размерности
                vector_size = getattr(self.nlp.vocab, 'vectors_length', None)
                if vector_size is None:
                    vector_size = 300  # Fallback размер
                return np.zeros(vector_size)
    
    def chunk_text(self, text: str, **kwargs) -> List[str]:
        """Разбивает текст на семантические чанки"""
        self._load_model()
        
        sentences = nltk.sent_tokenize(text)
        if len(sentences) <= 1:
            return [text]
        
        # Получаем эмбеддинги предложений
        embeddings = []
        for sentence in sentences:
            embedding = self._get_sentence_embedding(sentence)
            embeddings.append(embedding)
        
        # Вычисляем схожесть между соседними предложениями
        similarities = []
        for i in range(len(embeddings) - 1):
            vec1 = embeddings[i].reshape(1, -1)
            vec2 = embeddings[i + 1].reshape(1, -1)
            
            # Избегаем деления на ноль
            if np.allclose(vec1, 0) or np.allclose(vec2, 0):
                sim = 0.0
            else:
                sim = cosine_similarity(vec1, vec2)[0][0]
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
                # Разбиваем большие чанки
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
            'model_name': self.model_name
        }

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
            vec1 = np.array(embeddings[i]).reshape(1, -1)
            vec2 = np.array(embeddings[i + 1]).reshape(1, -1)
            sim = cosine_similarity(vec1, vec2)[0][0]
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