"""
Семантические методы чанкования текстов на основе эмбеддингов
"""

import nltk
import time
import os
import hashlib
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

# Проверяем доступность sklearn для кластеризации
try:
    from sklearn.cluster import AgglomerativeClustering
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .base import BaseChunker
from .basic import CharacterChunker

def safe_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Безопасное вычисление косинусной схожести с проверкой нулевых векторов"""
    try:
        # Проверяем что векторы не нулевые и имеют валидную норму
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
        
        # Убеждаемся что векторы имеют правильную форму
        v1 = vec1.reshape(1, -1) if vec1.ndim == 1 else vec1
        v2 = vec2.reshape(1, -1) if vec2.ndim == 1 else vec2
        
        # Проверяем размерности
        if v1.shape[1] != v2.shape[1]:
            return 0.0
        
        sim = cosine_similarity(v1, v2)[0][0]
        
        # Проверяем что результат валидный
        if np.isnan(sim) or np.isinf(sim):
            return 0.0
            
        return float(sim)
        
    except Exception:
        return 0.0

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
            vec1 = embeddings[i]
            vec2 = embeddings[i + 1]
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

class EnhancedSemanticChunker(BaseChunker):
    """Семантический chunker мирового класса с революционными алгоритмами"""
    
    def __init__(self, method: str = 'weighted_cumulative', similarity_threshold: float = 0.7, 
                 max_chunk_size: int = 1000, api_url: str = "https://api.openai.com/v1/embeddings",
                 api_key: str = "", model_name: str = "text-embedding-ada-002",
                 timeout: int = 30, max_retries: int = 3, window_size: int = 3,
                 min_coherence: float = 0.3, enable_clustering: bool = True):
        """
        Args:
            method: Метод чанкования ('weighted_cumulative', 'optimal_dp', 'multi_scale', 'topic_aware')
            similarity_threshold: Порог схожести (0.0-1.0)
            max_chunk_size: Максимальный размер чанка в символах
            api_url: URL API для эмбеддингов
            api_key: API ключ
            model_name: Название модели для эмбеддингов
            timeout: Таймаут запроса в секундах
            max_retries: Количество повторных попыток
            window_size: Размер окна для анализа контекста
            min_coherence: Минимальная связность чанка
            enable_clustering: Включить тематическую кластеризацию
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
        self.min_coherence = min_coherence
        self.enable_clustering = enable_clustering
        
        # Для spacy fallback
        self.nlp = None
        
        # Кэш для эмбеддингов
        self._embeddings_cache = {}
        self._sentences_cache = {}
        
    def _get_text_hash(self, text: str) -> str:
        """Создает хэш текста для кэширования"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _get_embeddings_from_cache_or_api(self, text: str, sentences: List[str]) -> List[List[float]]:
        """Получает эмбеддинги из кэша или через API"""
        text_hash = self._get_text_hash(text)
        
        # Проверяем кэш
        if text_hash in self._embeddings_cache:
            cached_sentences = self._sentences_cache.get(text_hash, [])
            if cached_sentences == sentences:
                print(f"🎯 Используем кэшированные эмбеддинги для текста (хэш: {text_hash[:8]})")
                return self._embeddings_cache[text_hash]
        
        # Получаем новые эмбеддинги
        print(f"🔄 Создаем новые эмбеддинги для {len(sentences)} предложений...")
        embeddings = self._get_embeddings_api(sentences)
        
        # Сохраняем в кэш
        self._embeddings_cache[text_hash] = embeddings
        self._sentences_cache[text_hash] = sentences.copy()
        
        print(f"💾 Эмбеддинги сохранены в кэш (хэш: {text_hash[:8]})")
        return embeddings
    
    def clear_embeddings_cache(self):
        """Очищает кэш эмбеддингов"""
        self._embeddings_cache.clear()
        self._sentences_cache.clear()
        print("🗑️ Кэш эмбеддингов очищен")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Возвращает информацию о кэше"""
        total_sentences = sum(len(sentences) for sentences in self._sentences_cache.values())
        return {
            'cached_texts': len(self._embeddings_cache),
            'total_sentences': total_sentences,
            'cache_size_mb': sum(
                len(str(emb)) for emb_list in self._embeddings_cache.values() 
                for emb in emb_list
            ) / (1024 * 1024)
        }
    
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
        """Получает эмбеддинги через API с fallback на Spacy"""
        if not self.api_key:
            return self._get_spacy_embeddings(texts)
        
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
                    print(f"API недоступен, переключаемся на Spacy: {e}")
                    return self._get_spacy_embeddings(texts)
                else:
                    time.sleep(2 ** attempt)
        
        return self._get_spacy_embeddings(texts)
    
    def _get_spacy_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Получает эмбеддинги через Spacy"""
        self._load_spacy_model()
        if self.nlp is None:
            # Fallback: случайные векторы
            return [[0.0] * 300 for _ in texts]
            
        embeddings = []
        for text in texts:
            doc = self.nlp(text)
            if doc.vector.any():
                embeddings.append(doc.vector.tolist())
            else:
                embeddings.append([0.0] * len(doc.vector))
        return embeddings
    
    def _calculate_coherence(self, embeddings: List[np.ndarray]) -> float:
        """Вычисляет внутреннюю связность чанка"""
        if len(embeddings) < 2:
            return 1.0
        
        # Вычисляем все попарные схожести внутри чанка
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = safe_cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)
        
        return float(np.mean(similarities)) if similarities else 0.0
    
    def _weighted_cumulative_chunking(self, sentences: List[str], text: str = "") -> List[str]:
        """Улучшенное кумулятивное чанкование с weighted averaging и sliding window"""
        if len(sentences) <= 1:
            return sentences
            
        # Получаем эмбеддинги из кэша или API
        if text:
            embeddings = self._get_embeddings_from_cache_or_api(text, sentences)
        else:
            # Fallback для случаев когда текст не передан
            embeddings = self._get_embeddings_api(sentences)
            
        embeddings = [np.array(emb) for emb in embeddings]
        
        chunks = []
        current_chunk = [sentences[0]]
        chunk_embeddings = [embeddings[0]]
        
        for i in range(1, len(sentences)):
            next_embedding = embeddings[i]
            
            # Взвешенное усреднение: более свежие предложения имеют больший вес
            if len(chunk_embeddings) > 0:
                weights = np.exp(np.linspace(-1, 0, len(chunk_embeddings)))
                weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(chunk_embeddings)) / len(chunk_embeddings)
                
                current_weighted_embedding = np.average(chunk_embeddings, axis=0, weights=weights)
            else:
                current_weighted_embedding = embeddings[0]
            
            # Sliding window analysis: проверяем схожесть с последними предложениями чанка
            window_size = min(self.window_size, len(chunk_embeddings))
            recent_embeddings = chunk_embeddings[-window_size:] if window_size > 0 else []
            
            # Вычисляем множественные метрики схожести
            if recent_embeddings:
                valid_similarities = []
                for emb in recent_embeddings:
                    sim = safe_cosine_similarity(emb, next_embedding)
                    valid_similarities.append(sim)
                
                avg_similarity = np.mean(valid_similarities) if valid_similarities else 0.0
            else:
                avg_similarity = 0.0
            
            weighted_similarity = safe_cosine_similarity(current_weighted_embedding, next_embedding)
            
            # Комбинированная метрика схожести
            combined_similarity = 0.7 * weighted_similarity + 0.3 * avg_similarity
            
            # Проверяем coherence чанка
            potential_chunk_embeddings = chunk_embeddings + [next_embedding]
            coherence_score = self._calculate_coherence(potential_chunk_embeddings)
            
            # Динамический порог на основе coherence
            dynamic_threshold = self.similarity_threshold * (0.8 + 0.4 * coherence_score)
            
            potential_chunk_text = ' '.join(current_chunk + [sentences[i]])
            
            if (combined_similarity >= dynamic_threshold and 
                len(potential_chunk_text) <= self.max_chunk_size and
                coherence_score >= self.min_coherence):
                
                # Добавляем в текущий чанк
                current_chunk.append(sentences[i])
                chunk_embeddings.append(next_embedding)
            else:
                # Сохраняем текущий чанк и начинаем новый
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentences[i]]
                chunk_embeddings = [next_embedding]
        
        # Добавляем последний чанк
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return self._post_process_chunks(chunks)
    
    def _optimal_dp_chunking(self, sentences: List[str], text: str = "") -> List[str]:
        """Глобально оптимальное чанкование через динамическое программирование"""
        if len(sentences) <= 1:
            return sentences
            
        # Получаем эмбеддинги из кэша или API
        if text:
            embeddings = self._get_embeddings_from_cache_or_api(text, sentences)
        else:
            embeddings = self._get_embeddings_api(sentences)
            
        embeddings = [np.array(emb) for emb in embeddings]
        
        n = len(sentences)
        
        # dp[i] = минимальная стоимость разбиения первых i предложений
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        parent = [-1] * (n + 1)
        
        for i in range(1, n + 1):
            for j in range(i):
                chunk_text = ' '.join(sentences[j:i])
                if len(chunk_text) <= self.max_chunk_size:
                    chunk_embeddings = embeddings[j:i]
                    
                    # Стоимость = обратная coherence (чем больше coherence, тем меньше стоимость)
                    coherence = self._calculate_coherence(chunk_embeddings)
                    
                    # Дополнительная стоимость за маленькие чанки
                    size_penalty = 0.1 if (i - j) < 2 else 0.0
                    
                    cost = (1.0 - coherence) + size_penalty
                    
                    if dp[j] + cost < dp[i]:
                        dp[i] = dp[j] + cost
                        parent[i] = j
        
        # Восстанавливаем оптимальное разбиение
        chunks = []
        i = n
        while i > 0:
            j = parent[i]
            if j >= 0:
                chunks.append(' '.join(sentences[j:i]))
            i = j
        
        chunks.reverse()
        return chunks if chunks else sentences
    
    def _multi_scale_chunking(self, text: str) -> List[str]:
        """Multi-scale анализ с иерархическим разбиением"""
        # Уровень 1: Документ → Секции (по пустым строкам)
        sections = [s.strip() for s in text.split('\n\n') if s.strip()]
        
        if len(sections) <= 1:
            # Нет секций, работаем с предложениями
            sentences = nltk.sent_tokenize(text)
            return self._weighted_cumulative_chunking(sentences, text)
        
        all_chunks = []
        for section in sections:
            sentences = nltk.sent_tokenize(section)
            
            if len(sentences) <= 2:
                all_chunks.append(section)
                continue
            
            if len(section) <= self.max_chunk_size:
                # Секция помещается в один чанк, проверяем coherence
                embeddings = self._get_embeddings_from_cache_or_api(section, sentences)
                embeddings = [np.array(emb) for emb in embeddings]
                coherence = self._calculate_coherence(embeddings)
                
                if coherence >= self.min_coherence:
                    all_chunks.append(section)
                    continue
            
            # Разбиваем секцию на чанки
            section_chunks = self._weighted_cumulative_chunking(sentences, section)
            all_chunks.extend(section_chunks)
        
        return all_chunks
    
    def _topic_aware_chunking(self, sentences: List[str], text: str = "") -> List[str]:
        """Тематическое чанкование с кластеризацией"""
        if len(sentences) <= 3 or not SKLEARN_AVAILABLE:
            return self._weighted_cumulative_chunking(sentences, text)
        
        # Получаем эмбеддинги из кэша или API
        if text:
            embeddings = self._get_embeddings_from_cache_or_api(text, sentences)
        else:
            embeddings = self._get_embeddings_api(sentences)
            
        embeddings_array = np.array(embeddings)
        
        # Проверяем что embeddings не пустые и не содержат только нули
        if embeddings_array.shape[0] == 0 or np.allclose(embeddings_array, 0):
            return self._weighted_cumulative_chunking(sentences, text)
        
        # Проверяем наличие нулевых векторов (проблема с cosine метрикой)
        zero_vectors = np.all(embeddings_array == 0, axis=1)
        if np.any(zero_vectors):
            print(f"Найдено {np.sum(zero_vectors)} нулевых векторов, заменяем их на малые случайные значения")
            # Заменяем нулевые векторы на очень маленькие случайные значения
            for i in np.where(zero_vectors)[0]:
                embeddings_array[i] = np.random.normal(0, 0.001, embeddings_array.shape[1])
        
        # Дополнительная проверка: нормы векторов должны быть больше нуля
        norms = np.linalg.norm(embeddings_array, axis=1)
        zero_norm_mask = norms < 1e-10
        if np.any(zero_norm_mask):
            print(f"Найдено {np.sum(zero_norm_mask)} векторов с нулевой нормой, добавляем шум")
            for i in np.where(zero_norm_mask)[0]:
                embeddings_array[i] = np.random.normal(0, 0.001, embeddings_array.shape[1])
        
        # Определяем оптимальное количество кластеров
        max_clusters = min(len(sentences) // 2, 8)
        n_clusters = max(2, max_clusters)
        
        try:
            # Сначала пробуем cosine метрику с исправленными данными
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters, 
                metric='cosine', 
                linkage='average'
            )
            
            sentence_clusters = clustering.fit_predict(embeddings_array)
            
            # Группируем предложения по кластерам в правильном порядке
            cluster_groups = {}
            for i, cluster_id in enumerate(sentence_clusters):
                if cluster_id not in cluster_groups:
                    cluster_groups[cluster_id] = []
                cluster_groups[cluster_id].append((i, sentences[i]))
            
            # Сортируем кластеры по позиции первого предложения
            sorted_clusters = []
            for cluster_id in cluster_groups:
                cluster_sentences = cluster_groups[cluster_id]
                cluster_sentences.sort(key=lambda x: x[0])  # Сортируем по индексу
                sorted_clusters.append((min(s[0] for s in cluster_sentences), cluster_sentences))
            
            sorted_clusters.sort(key=lambda x: x[0])  # Сортируем кластеры
            
            chunks = []
            for _, cluster_sentences in sorted_clusters:
                cluster_text_parts = [s[1] for s in cluster_sentences]  # Извлекаем текст
                cluster_text = ' '.join(cluster_text_parts)
                
                if len(cluster_text) <= self.max_chunk_size:
                    chunks.append(cluster_text)
                else:
                    # Рекурсивно разбиваем большие кластеры
                    sub_chunks = self._weighted_cumulative_chunking(cluster_text_parts, cluster_text)
                    chunks.extend(sub_chunks)
            
            return chunks
            
        except Exception as e:
            print(f"Ошибка cosine кластеризации: {e}. Пробуем euclidean метрику")
            try:
                # Fallback на euclidean метрику
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters, 
                    metric='euclidean', 
                    linkage='ward'
                )
                sentence_clusters = clustering.fit_predict(embeddings_array)
                
                # Продолжаем с обычным кодом группировки
                cluster_groups = {}
                for i, cluster_id in enumerate(sentence_clusters):
                    if cluster_id not in cluster_groups:
                        cluster_groups[cluster_id] = []
                    cluster_groups[cluster_id].append((i, sentences[i]))
                
                # Сортируем кластеры по позиции первого предложения
                sorted_clusters = []
                for cluster_id in cluster_groups:
                    cluster_sentences = cluster_groups[cluster_id]
                    cluster_sentences.sort(key=lambda x: x[0])  # Сортируем по индексу
                    sorted_clusters.append((min(s[0] for s in cluster_sentences), cluster_sentences))
                
                sorted_clusters.sort(key=lambda x: x[0])  # Сортируем кластеры
                
                chunks = []
                for _, cluster_sentences in sorted_clusters:
                    cluster_text_parts = [s[1] for s in cluster_sentences]  # Извлекаем текст
                    cluster_text = ' '.join(cluster_text_parts)
                    
                    if len(cluster_text) <= self.max_chunk_size:
                        chunks.append(cluster_text)
                    else:
                        # Рекурсивно разбиваем большие кластеры
                        sub_chunks = self._weighted_cumulative_chunking(cluster_text_parts, cluster_text)
                        chunks.extend(sub_chunks)
                
                return chunks
                
            except Exception as e2:
                print(f"Ошибка euclidean кластеризации: {e2}. Переключаемся на weighted_cumulative")
                return self._weighted_cumulative_chunking(sentences, text)
    
    def _smart_split_oversized_chunk(self, chunk_text: str) -> List[str]:
        """Умное разделение больших чанков с сохранением семантики"""
        sentences = nltk.sent_tokenize(chunk_text)
        if len(sentences) <= 1:
            # Если один sentence слишком большой, используем character splitting
            if len(chunk_text) > self.max_chunk_size:
                mid = len(chunk_text) // 2
                # Ищем ближайший пробел для разделения
                while mid < len(chunk_text) and chunk_text[mid] != ' ':
                    mid += 1
                if mid < len(chunk_text):
                    return [chunk_text[:mid].strip(), chunk_text[mid:].strip()]
            return [chunk_text]
        
        embeddings = self._get_embeddings_api(sentences)
        embeddings = [np.array(emb) for emb in embeddings]
        
        # Находим оптимальную точку разделения на основе минимальной схожести
        split_scores = []
        for i in range(1, len(sentences)):
            left_embs = embeddings[:i]
            right_embs = embeddings[i:]
            
            if left_embs and right_embs:
                # Проверяем что embeddings не пустые
                valid_left = [emb for emb in left_embs if not np.allclose(emb, 0)]
                valid_right = [emb for emb in right_embs if not np.allclose(emb, 0)]
                
                if valid_left and valid_right:
                    left_centroid = np.mean(valid_left, axis=0)
                    right_centroid = np.mean(valid_right, axis=0)
                    
                    # Минимизируем схожесть между частями
                    inter_similarity = safe_cosine_similarity(left_centroid, right_centroid)
                else:
                    inter_similarity = 0.0
            else:
                inter_similarity = 0.0
            
            split_scores.append(inter_similarity)
        
        if not split_scores:
            return [chunk_text]
        
        # Выбираем точку с минимальной схожестью
        best_split = np.argmin(split_scores) + 1
        
        left_chunk = ' '.join(sentences[:best_split])
        right_chunk = ' '.join(sentences[best_split:])
        
        chunks = []
        
        # Рекурсивно обрабатываем части если они все еще слишком большие
        if len(left_chunk) <= self.max_chunk_size:
            chunks.append(left_chunk)
        else:
            chunks.extend(self._smart_split_oversized_chunk(left_chunk))
        
        if len(right_chunk) <= self.max_chunk_size:
            chunks.append(right_chunk)
        else:
            chunks.extend(self._smart_split_oversized_chunk(right_chunk))
        
        return chunks
    
    def _post_process_chunks(self, chunks: List[str]) -> List[str]:
        """Пост-обработка чанков: объединение маленьких и разделение больших"""
        if not chunks:
            return chunks
        
        processed_chunks = []
        
        for chunk in chunks:
            if len(chunk) > self.max_chunk_size:
                # Разделяем большие чанки
                split_chunks = self._smart_split_oversized_chunk(chunk)
                processed_chunks.extend(split_chunks)
            else:
                processed_chunks.append(chunk)
        
        # Объединяем очень маленькие соседние чанки
        final_chunks = []
        i = 0
        while i < len(processed_chunks):
            current_chunk = processed_chunks[i]
            
            # Если чанк очень маленький, пытаемся объединить со следующим
            if (len(current_chunk) < self.max_chunk_size * 0.3 and 
                i + 1 < len(processed_chunks)):
                
                next_chunk = processed_chunks[i + 1]
                combined = current_chunk + ' ' + next_chunk
                
                if len(combined) <= self.max_chunk_size:
                    final_chunks.append(combined)
                    i += 2  # Пропускаем следующий чанк
                    continue
            
            final_chunks.append(current_chunk)
            i += 1
        
        return [chunk.strip() for chunk in final_chunks if chunk.strip()]
    
    def chunk_text(self, text: str, **kwargs) -> List[str]:
        """Основной метод чанкования"""
        if not text.strip():
            return []
        
        sentences = nltk.sent_tokenize(text)
        
        if len(sentences) <= 1:
            if len(text) <= self.max_chunk_size:
                return [text]
            else:
                return self._smart_split_oversized_chunk(text)
        
        try:
            if self.method == 'weighted_cumulative':
                return self._weighted_cumulative_chunking(sentences, text)
            elif self.method == 'optimal_dp':
                return self._optimal_dp_chunking(sentences, text)
            elif self.method == 'multi_scale':
                return self._multi_scale_chunking(text)
            elif self.method == 'topic_aware':
                return self._topic_aware_chunking(sentences, text)
            else:
                # Fallback на weighted_cumulative
                return self._weighted_cumulative_chunking(sentences, text)
                
        except Exception as e:
            print(f"Ошибка в семантическом чанковании: {e}")
            # Fallback на простое разбиение по предложениям
            return [' '.join(sentences[i:i+3]) for i in range(0, len(sentences), 3)]
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'method': self.method,
            'similarity_threshold': self.similarity_threshold,
            'max_chunk_size': self.max_chunk_size,
            'api_url': self.api_url,
            'model_name': self.model_name,
            'timeout': self.timeout,
            'max_retries': self.max_retries,
            'window_size': self.window_size,
            'min_coherence': self.min_coherence,
            'enable_clustering': self.enable_clustering
        } 