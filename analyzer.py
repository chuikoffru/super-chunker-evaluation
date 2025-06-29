"""
Модуль для анализа и визуализации результатов чанкования
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any
import numpy as np

# Проверяем доступность nltk
try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

class ChunkingAnalyzer:
    """Класс для анализа результатов чанкования"""
    
    def __init__(self):
        self.results = {}
    
    def analyze_chunks(self, chunks: List[str], method_name: str) -> Dict[str, Any]:
        """Анализирует результаты чанкования"""
        if not chunks:
            return {
                'method': method_name,
                'total_chunks': 0,
                'avg_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0,
                'std_chunk_size': 0,
                'chunk_sizes': []
            }
        
        chunk_sizes = [len(chunk) for chunk in chunks]
        
        analysis = {
            'method': method_name,
            'total_chunks': len(chunks),
            'avg_chunk_size': np.mean(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'std_chunk_size': np.std(chunk_sizes),
            'chunk_sizes': chunk_sizes,
            'chunks': chunks
        }
        
        self.results[method_name] = analysis
        return analysis
    

    def create_comparison_chart(self) -> go.Figure:
        """Создает сравнительный график метрик"""
        if not self.results:
            return go.Figure()
        
        methods = list(self.results.keys())
        total_chunks = [self.results[method]['total_chunks'] for method in methods]
        avg_sizes = [self.results[method]['avg_chunk_size'] for method in methods]
        std_sizes = [self.results[method]['std_chunk_size'] for method in methods]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Общее количество чанков",
                "Средний размер чанка",
                "Стандартное отклонение размера",
                "Диапазон размеров"
            ]
        )
        
        # Количество чанков
        fig.add_trace(
            go.Bar(x=methods, y=total_chunks, name="Количество чанков"),
            row=1, col=1
        )
        
        # Средний размер
        fig.add_trace(
            go.Bar(x=methods, y=avg_sizes, name="Средний размер"),
            row=1, col=2
        )
        
        # Стандартное отклонение
        fig.add_trace(
            go.Bar(x=methods, y=std_sizes, name="Стд. отклонение"),
            row=2, col=1
        )
        
        # Диапазон размеров (box plot)
        for method in methods:
            chunk_sizes = self.results[method]['chunk_sizes']
            if chunk_sizes:
                fig.add_trace(
                    go.Box(y=chunk_sizes, name=method),
                    row=2, col=2
                )
        
        fig.update_layout(
            title="Сравнение методов чанкования",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_chunk_visualization(self, method_name: str, max_chunks: int = 10) -> go.Figure:
        """Создает визуализацию отдельных чанков"""
        if method_name not in self.results:
            return go.Figure()
        
        chunks = self.results[method_name]['chunks'][:max_chunks]
        chunk_sizes = [len(chunk) for chunk in chunks]
        chunk_indices = list(range(1, len(chunks) + 1))
        
        # Создаем превью чанков (первые 100 символов)
        chunk_previews = [
            chunk[:100] + "..." if len(chunk) > 100 else chunk 
            for chunk in chunks
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=chunk_indices,
            y=chunk_sizes,
            text=chunk_previews,
            textposition='none',  # Убираем текст с баров
            hovertemplate='<b>Чанк %{x}</b><br>' +
                         'Размер: %{y} символов<br>' +
                         'Превью: %{text}<br>' +
                         '<extra></extra>',
            marker_color=px.colors.sequential.Viridis
        ))
        
        fig.update_layout(
            title=f"Размеры чанков - {method_name} (первые {len(chunks)} чанков)",
            xaxis_title="Номер чанка",
            yaxis_title="Размер чанка (символы)",
            height=400
        )
        
        return fig
    
    def get_summary_table(self) -> pd.DataFrame:
        """Создает сводную таблицу результатов"""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for method, analysis in self.results.items():
            data.append({
                'Метод': method,
                'Количество чанков': analysis['total_chunks'],
                'Средний размер': round(analysis['avg_chunk_size'], 1),
                'Мин. размер': analysis['min_chunk_size'],
                'Макс. размер': analysis['max_chunk_size'],
                'Стандартное отклонение': round(analysis['std_chunk_size'], 1)
            })
        
        return pd.DataFrame(data)
    
    def clear_results(self):
        """Очищает сохраненные результаты"""
        self.results = {}

class TextStatistics:
    """Класс для анализа статистики исходного текста"""
    
    @staticmethod
    def get_text_stats(text: str) -> Dict[str, Any]:
        """Получает статистику текста"""
        if not text:
            return {}
        
        lines = text.split('\n')
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        words = text.split()
        
        if NLTK_AVAILABLE:
            sentences = nltk.sent_tokenize(text)
        else:
            # Простой способ подсчета предложений
            sentences = [s for s in text.split('.') if s.strip()]
        
        return {
            'Общее количество символов': len(text),
            'Количество символов без пробелов': len(text.replace(' ', '')),
            'Количество слов': len(words),
            'Количество предложений': len(sentences),
            'Количество абзацев': len(paragraphs),
            'Количество строк': len(lines),
            'Средняя длина слова': round(np.mean([len(word) for word in words]) if words else 0, 1),
            'Средняя длина предложения': round(np.mean([len(sent) for sent in sentences]) if sentences else 0, 1),
            'Средняя длина абзаца': round(np.mean([len(para) for para in paragraphs]) if paragraphs else 0, 1)
        } 