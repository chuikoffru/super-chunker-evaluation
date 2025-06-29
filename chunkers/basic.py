"""
Базовые методы чанкования текстов
"""

import nltk
import tiktoken
from typing import List, Dict, Any
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    SpacyTextSplitter
)

from .base import BaseChunker

# Загружаем необходимые данные NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

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
        # Пытаемся получить кодировку для указанной модели
        encoding = None
        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            # Используем fallback если модель не найдена
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