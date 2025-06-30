"""
Продвинутое семантическое чанкование с различными стратегиями
"""

import nltk
import time
import os
from typing import List, Dict, Any
import numpy as np
import httpx

from .base import BaseChunker
from .basic import CharacterChunker
from .semantic_utils import safe_cosine_similarity, SPACY_AVAILABLE

# Проверяем доступность spacy
if SPACY_AVAILABLE:
    import spacy


class AdvancedSemanticChunker(BaseChunker):
    """Продвинутое семантическое чанкование с различными стратегиями"""
    
    def __init__(self, method: str = 'cumulative', similarity_threshold: float = 0.7, 
                 max_chunk_size: int = 1000, api_url: str = "https://api.openai.com/v1/embeddings",
                 api_key: str = "", model_name: str = "text-embedding-ada-002",
                 timeout: int = 30, max_retries: int = 3, window_size: int = 3):
        """
        Args:
            method: Метод чанкования ('cumulative', 'hierarchical', 'adaptive')
            similarity_threshold: Порог схожести (0.0-1.0)
            max_chunk_size: Максимальный размер чанка в символах
            api_url: URL API для эмбеддингов
            api_key: API ключ
            model_name: Название модели для эмбеддингов
            timeout: Таймаут запроса в секундах
            max_retries: Количество повторных попыток
            window_size: Размер окна для hierarchical метода
        """
        self.method = method
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.api_url = api_url
        self.api_key = api_key or os.getenv('OPENAI_API_KEY', '')
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.window_size = window_size
        
        # Для spacy fallback
        self.nlp = None
        
    def _load_spacy_model(self):
        """Загружает Spacy модель как fallback"""
        if self.nlp is None and SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("ru_core_news_md")
            except OSError:
                try:
                    self.nlp = spacy.load("en_core_web_md")
                except OSError:
                    pass
    
    def _get_embeddings_api(self, texts: List[str]) -> List[List[float]]:
        """Получает эмбеддинги через API"""
        if not self.api_key:
            # Fallback на Spacy
            self._load_spacy_model()
            if self.nlp:
                return [self._get_spacy_embedding(text) for text in texts]
            else:
                raise ValueError("API ключ не найден и Spacy недоступен")
        
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
                    
                if 'data' in data:
                    return [item['embedding'] for item in data['data']]
                else:
                    raise ValueError(f"Неожиданный формат ответа API: {data}")
                    
            except Exception as e:
                if attempt == self.max_retries - 1:
                    # Fallback на Spacy при ошибке API
                    self._load_spacy_model()
                    if self.nlp:
                        return [self._get_spacy_embedding(text) for text in texts]
                    else:
                        raise Exception(f"Ошибка API и Spacy недоступен: {e}")
                else:
                    time.sleep(2 ** attempt)
        
        raise Exception("Не удалось получить эмбеддинги")
    
    def _get_spacy_embedding(self, text: str) -> List[float]:
        """Получает эмбеддинг через Spacy"""
        if self.nlp is None:
            return [0.0] * 300  # Fallback вектор
            
        doc = self.nlp(text)
        if doc.vector.any():
            return doc.vector.tolist()
        else:
            return [0.0] * len(doc.vector)
    
    def _cumulative_chunking(self, sentences: List[str]) -> List[str]:
        """Кумулятивное чанкование с накоплением семантического контекста"""
        if len(sentences) <= 1:
            return sentences
            
        # Получаем эмбеддинги для всех предложений
        embeddings = self._get_embeddings_api(sentences)
        
        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = np.array(embeddings[0])
        
        for i in range(1, len(sentences)):
            next_embedding = np.array(embeddings[i])
            
            # Сравниваем усредненный вектор чанка с новым предложением
            similarity = safe_cosine_similarity(current_embedding, next_embedding)
            
            chunk_text = ' '.join(current_chunk + [sentences[i]])
            
            if similarity >= self.similarity_threshold and len(chunk_text) <= self.max_chunk_size:
                # Добавляем в текущий чанк и обновляем усредненный вектор
                current_chunk.append(sentences[i])
                current_embedding = np.mean([current_embedding, next_embedding], axis=0)
            else:
                # Сохраняем текущий чанк и начинаем новый
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentences[i]]
                current_embedding = next_embedding
        
        # Добавляем последний чанк
        chunks.append(' '.join(current_chunk))
        return chunks
    
    def _hierarchical_chunking(self, text: str) -> List[str]:
        """Иерархическое чанкование по уровням"""
        # Уровень 1: Разбивка по абзацам
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if len(paragraphs) <= 1:
            # Если нет абзацев, работаем как обычно
            sentences = nltk.sent_tokenize(text)
            return self._cumulative_chunking(sentences)
        
        # Уровень 2: Семантическое чанкование внутри каждого абзаца
        all_chunks = []
        for paragraph in paragraphs:
            sentences = nltk.sent_tokenize(paragraph)
            if len(sentences) > 1:
                para_chunks = self._cumulative_chunking(sentences)
                all_chunks.extend(para_chunks)
            else:
                all_chunks.append(paragraph)
        
        # Уровень 3: Объединение соседних чанков если они семантически близки
        return self._merge_similar_chunks(all_chunks)
    
    def _merge_similar_chunks(self, chunks: List[str]) -> List[str]:
        """Объединяет семантически близкие соседние чанки"""
        if len(chunks) <= 1:
            return chunks
        
        # Получаем эмбеддинги чанков
        embeddings = self._get_embeddings_api(chunks)
        
        merged_chunks = [chunks[0]]
        merged_embeddings = [np.array(embeddings[0])]
        
        for i in range(1, len(chunks)):
            current_embedding = np.array(embeddings[i])
            last_merged_embedding = merged_embeddings[-1]
            
            similarity = safe_cosine_similarity(last_merged_embedding, current_embedding)
            
            combined_text = merged_chunks[-1] + ' ' + chunks[i]
            
            if similarity >= self.similarity_threshold and len(combined_text) <= self.max_chunk_size:
                # Объединяем с предыдущим чанком
                merged_chunks[-1] = combined_text
                merged_embeddings[-1] = np.mean([last_merged_embedding, current_embedding], axis=0)
            else:
                # Добавляем как новый чанк
                merged_chunks.append(chunks[i])
                merged_embeddings.append(current_embedding)
        
        return merged_chunks
    
    def _adaptive_chunking(self, sentences: List[str]) -> List[str]:
        """Адаптивное чанкование с динамическим порогом"""
        if len(sentences) <= 1:
            return sentences
            
        # Получаем эмбеддинги
        embeddings = self._get_embeddings_api(sentences)
        
        # Вычисляем все попарные схожести между соседними предложениями
        similarities = []
        for i in range(len(embeddings) - 1):
            vec1 = np.array(embeddings[i])
            vec2 = np.array(embeddings[i + 1])
            sim = safe_cosine_similarity(vec1, vec2)
            similarities.append(sim)
        
        if not similarities:
            return sentences
        
        # Адаптивный порог на основе статистики
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        
        # Используем более консервативный порог для лучшей группировки
        adaptive_threshold = max(
            float(mean_sim - 0.5 * std_sim),  # Не слишком низкий порог
            self.similarity_threshold * 0.7  # Не меньше 70% от заданного порога
        )
        
        # Применяем кумулятивное чанкование с адаптивным порогом
        original_threshold = self.similarity_threshold
        self.similarity_threshold = adaptive_threshold
        
        try:
            result = self._cumulative_chunking(sentences)
        finally:
            self.similarity_threshold = original_threshold
        
        return result
    
    def chunk_text(self, text: str, **kwargs) -> List[str]:
        """Основной метод чанкования"""
        sentences = nltk.sent_tokenize(text)
        
        if len(sentences) <= 1:
            return [text]
        
        if self.method == 'cumulative':
            return self._cumulative_chunking(sentences)
        elif self.method == 'hierarchical':
            return self._hierarchical_chunking(text)
        elif self.method == 'adaptive':
            return self._adaptive_chunking(sentences)
        else:
            raise ValueError(f"Неизвестный метод: {self.method}")
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'method': self.method,
            'similarity_threshold': self.similarity_threshold,
            'max_chunk_size': self.max_chunk_size,
            'api_url': self.api_url,
            'model_name': self.model_name,
            'timeout': self.timeout,
            'max_retries': self.max_retries,
            'window_size': self.window_size
        } 