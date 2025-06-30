"""
Семантическое чанкование с использованием Spacy эмбеддингов
"""

import nltk
from typing import List, Dict, Any
import numpy as np

from .base import BaseChunker
from .basic import CharacterChunker
from .semantic_utils import safe_cosine_similarity, SPACY_AVAILABLE

# Проверяем доступность spacy
if SPACY_AVAILABLE:
    import spacy


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