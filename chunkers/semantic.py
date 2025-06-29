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
            similarity = cosine_similarity(
                current_embedding.reshape(1, -1), 
                next_embedding.reshape(1, -1)
            )[0][0]
            
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
            
            similarity = cosine_similarity(
                last_merged_embedding.reshape(1, -1),
                current_embedding.reshape(1, -1)
            )[0][0]
            
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
            vec1 = np.array(embeddings[i]).reshape(1, -1)
            vec2 = np.array(embeddings[i + 1]).reshape(1, -1)
            sim = cosine_similarity(vec1, vec2)[0][0]
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