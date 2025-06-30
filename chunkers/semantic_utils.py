"""
Общие утилиты и функции для семантических чанкеров
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