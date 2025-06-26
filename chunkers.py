"""
Модуль с различными методами чанкования текстов
"""

import re
import nltk
import tiktoken
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    SpacyTextSplitter
)
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Загружаем необходимые данные NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class BaseChunker(ABC):
    """Базовый класс для всех чанкеров"""
    
    @abstractmethod
    def chunk_text(self, text: str, **kwargs) -> List[str]:
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        pass

class CharacterChunker(BaseChunker):
    """Чанкование по символам"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def chunk_text(self, text: str, **kwargs) -> List[str]:
        splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator=kwargs.get('separator', '\n\n')
        )
        return splitter.split_text(text)
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap
        }

class RecursiveChunker(BaseChunker):
    """Рекурсивное чанкование"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def chunk_text(self, text: str, **kwargs) -> List[str]:
        separators = kwargs.get('separators', ["\n\n", "\n", " ", ""])
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=separators
        )
        return splitter.split_text(text)
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap
        }

class TokenChunker(BaseChunker):
    """Чанкование по токенам"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50, model_name: str = "gpt-3.5-turbo"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        
    def chunk_text(self, text: str, **kwargs) -> List[str]:
        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
            
        splitter = TokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            encoding_name=encoding.name
        )
        return splitter.split_text(text)
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'model_name': self.model_name
        }

class SentenceChunker(BaseChunker):
    """Чанкование по предложениям"""
    
    def __init__(self, sentences_per_chunk: int = 5, overlap_sentences: int = 1):
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap_sentences = overlap_sentences
        
    def chunk_text(self, text: str, **kwargs) -> List[str]:
        sentences = nltk.sent_tokenize(text)
        chunks = []
        
        for i in range(0, len(sentences), self.sentences_per_chunk - self.overlap_sentences):
            chunk_sentences = sentences[i:i + self.sentences_per_chunk]
            if chunk_sentences:
                chunks.append(' '.join(chunk_sentences))
                
        return chunks
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'sentences_per_chunk': self.sentences_per_chunk,
            'overlap_sentences': self.overlap_sentences
        }

class ParagraphChunker(BaseChunker):
    """Чанкование по абзацам"""
    
    def __init__(self, paragraphs_per_chunk: int = 3, overlap_paragraphs: int = 1):
        self.paragraphs_per_chunk = paragraphs_per_chunk
        self.overlap_paragraphs = overlap_paragraphs
        
    def chunk_text(self, text: str, **kwargs) -> List[str]:
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        
        for i in range(0, len(paragraphs), self.paragraphs_per_chunk - self.overlap_paragraphs):
            chunk_paragraphs = paragraphs[i:i + self.paragraphs_per_chunk]
            if chunk_paragraphs:
                chunks.append('\n\n'.join(chunk_paragraphs))
                
        return chunks
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'paragraphs_per_chunk': self.paragraphs_per_chunk,
            'overlap_paragraphs': self.overlap_paragraphs
        }

class SemanticChunker(BaseChunker):
    """Семантическое чанкование на основе схожести предложений"""
    
    def __init__(self, similarity_threshold: float = 0.7, max_chunk_size: int = 1000):
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.model: Optional[SentenceTransformer] = None
        
    def _load_model(self):
        if self.model is None:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def chunk_text(self, text: str, **kwargs) -> List[str]:
        self._load_model()
        
        sentences = nltk.sent_tokenize(text)
        if len(sentences) <= 1:
            return [text]
            
        # Проверяем, что модель загружена
        if self.model is None:
            raise RuntimeError("Модель не загружена")
            
        # Получаем эмбеддинги предложений
        embeddings = self.model.encode(sentences)
        
        # Вычисляем схожесть между соседними предложениями
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            similarities.append(sim)
        
        # Находим точки разделения (где схожесть ниже порога)
        split_points = [0]
        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold:
                split_points.append(i + 1)
        split_points.append(len(sentences))
        
        # Создаем чанки
        chunks = []
        for i in range(len(split_points) - 1):
            start = split_points[i]
            end = split_points[i + 1]
            chunk_text = ' '.join(sentences[start:end])
            
            # Проверяем размер чанка
            if len(chunk_text) > self.max_chunk_size:
                # Если чанк слишком большой, разбиваем его дальше
                sub_chunker = CharacterChunker(chunk_size=self.max_chunk_size, chunk_overlap=100)
                sub_chunks = sub_chunker.chunk_text(chunk_text)
                chunks.extend(sub_chunks)
            else:
                chunks.append(chunk_text)
                
        return chunks
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'similarity_threshold': self.similarity_threshold,
            'max_chunk_size': self.max_chunk_size
        }

class FixedSizeChunker(BaseChunker):
    """Простое чанкование фиксированного размера"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def chunk_text(self, text: str, **kwargs) -> List[str]:
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.overlap
            
        return [chunk for chunk in chunks if chunk.strip()]
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'chunk_size': self.chunk_size,
            'overlap': self.overlap
        }

# Реестр доступных чанкеров
CHUNKERS = {
    'Символы': CharacterChunker,
    'Рекурсивный': RecursiveChunker,
    'Токены': TokenChunker,
    'Предложения': SentenceChunker,
    'Абзацы': ParagraphChunker,
    'Семантический': SemanticChunker,
    'Фиксированный размер': FixedSizeChunker
} 