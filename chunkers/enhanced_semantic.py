"""
Семантический chunker мирового класса с революционными алгоритмами
"""

import nltk
import time
import os
import hashlib
from typing import List, Dict, Any
import numpy as np
import httpx

from .base import BaseChunker
from .semantic_utils import safe_cosine_similarity, SPACY_AVAILABLE, SKLEARN_AVAILABLE

# Проверяем доступность spacy
if SPACY_AVAILABLE:
    import spacy

# Проверяем доступность sklearn для кластеризации  
if SKLEARN_AVAILABLE:
    from sklearn.cluster import AgglomerativeClustering


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