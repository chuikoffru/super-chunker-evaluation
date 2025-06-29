"""
Продвинутая система анализа качества чанкования по принципу PSR
(Performance, Scalability, Reliability)
"""

import time
import numpy as np
from typing import List, Dict, Any
import nltk
from chunkers.base import BaseChunker
import psutil
import gc

class PSRAnalyzer:
    """
    Анализатор качества чанкования по принципу PSR:
    - Performance: Производительность и скорость
    - Scalability: Масштабируемость для больших текстов  
    - Reliability: Надежность и качество результатов
    """
    
    def __init__(self):
        self.results = {}
        
    def analyze_chunker(self, chunker: BaseChunker, test_texts: List[str], 
                       method_name: str) -> Dict[str, Any]:
        """
        Полный анализ чанкера по принципу PSR
        
        Args:
            chunker: Экземпляр чанкера для тестирования
            test_texts: Список тестовых текстов разного размера
            method_name: Название метода для отчета
            
        Returns:
            Словарь с результатами анализа по PSR
        """
        print(f"🔍 Анализируем {method_name}...")
        
        performance_metrics = self._analyze_performance(chunker, test_texts)
        scalability_metrics = self._analyze_scalability(chunker, test_texts)
        reliability_metrics = self._analyze_reliability(chunker, test_texts)
        
        # Вычисляем общий PSR Score
        psr_score = self._calculate_psr_score(
            performance_metrics, scalability_metrics, reliability_metrics
        )
        
        analysis_result = {
            'method_name': method_name,
            'performance': performance_metrics,
            'scalability': scalability_metrics,
            'reliability': reliability_metrics,
            'psr_score': psr_score,
            'timestamp': time.time()
        }
        
        self.results[method_name] = analysis_result
        return analysis_result
    
    def _analyze_performance(self, chunker: BaseChunker, test_texts: List[str]) -> Dict[str, float]:
        """Анализ производительности (Performance)"""
        print("  📊 Анализ производительности...")
        
        execution_times = []
        memory_usage = []
        
        for text in test_texts:
            # Измеряем время выполнения
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            try:
                chunks = chunker.chunk_text(text)
                
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                execution_times.append(end_time - start_time)
                memory_usage.append(end_memory - start_memory)
                
            except Exception as e:
                print(f"    ⚠️ Ошибка при обработке текста: {e}")
                execution_times.append(float('inf'))
                memory_usage.append(float('inf'))
            
            # Очистка памяти
            gc.collect()
        
        # Рассчитываем метрики производительности
        avg_time = np.mean([t for t in execution_times if t != float('inf')])
        max_time = max([t for t in execution_times if t != float('inf')]) if execution_times else 0
        avg_memory = np.mean([m for m in memory_usage if m != float('inf')])
        
        # Скорость обработки (символов в секунду)
        total_chars = sum(len(text) for text in test_texts)
        total_time = sum([t for t in execution_times if t != float('inf')])
        chars_per_second = total_chars / total_time if total_time > 0 else 0
        
        return {
            'avg_execution_time': float(avg_time),
            'max_execution_time': float(max_time),
            'avg_memory_usage': float(avg_memory),
            'chars_per_second': float(chars_per_second),
            'error_rate': float(sum(1 for t in execution_times if t == float('inf')) / len(execution_times))
        }
    
    def _analyze_scalability(self, chunker: BaseChunker, test_texts: List[str]) -> Dict[str, float]:
        """Анализ масштабируемости (Scalability)"""
        print("  📈 Анализ масштабируемости...")
        
        # Тексты разного размера для анализа масштабируемости
        text_sizes = []
        processing_times = []
        chunk_counts = []
        
        for text in test_texts:
            size = len(text)
            
            start_time = time.time()
            try:
                chunks = chunker.chunk_text(text)
                processing_time = time.time() - start_time
                
                text_sizes.append(size)
                processing_times.append(processing_time)
                chunk_counts.append(len(chunks))
                
            except Exception:
                continue
        
        if not text_sizes:
            return {'scalability_factor': 0, 'efficiency_ratio': 0, 'size_correlation': 0}
        
        # Коэффициент масштабируемости (как растет время с размером)
        if len(text_sizes) > 1:
            size_time_correlation = np.corrcoef(text_sizes, processing_times)[0, 1]
        else:
            size_time_correlation = 0
            
        # Эффективность (чанков на единицу времени)
        efficiency_ratios = [
            chunk_count / processing_time if processing_time > 0 else 0 
            for chunk_count, processing_time in zip(chunk_counts, processing_times)
        ]
        avg_efficiency = np.mean(efficiency_ratios) if efficiency_ratios else 0
        
        # Фактор масштабируемости (обратно пропорционален корреляции размера и времени)
        scalability_factor = max(0, 1 - abs(size_time_correlation))
        
        return {
            'scalability_factor': float(scalability_factor),
            'efficiency_ratio': float(avg_efficiency),
            'size_correlation': float(size_time_correlation),
            'max_text_size_handled': float(max(text_sizes) if text_sizes else 0)
        }
    
    def _analyze_reliability(self, chunker: BaseChunker, test_texts: List[str]) -> Dict[str, float]:
        """Анализ надежности (Reliability)"""
        print("  🎯 Анализ надежности...")
        
        chunk_size_consistency = []
        semantic_coherence_scores = []
        boundary_accuracy_scores = []
        error_count = 0
        
        for text in test_texts:
            try:
                chunks = chunker.chunk_text(text)
                
                if not chunks:
                    error_count += 1
                    continue
                
                # 1. Консистентность размеров чанков
                chunk_sizes = [len(chunk) for chunk in chunks]
                size_std = np.std(chunk_sizes) if len(chunk_sizes) > 1 else 0
                size_mean = np.mean(chunk_sizes) if chunk_sizes else 0
                size_consistency = 1 - (size_std / size_mean) if size_mean > 0 else 0
                chunk_size_consistency.append(max(0, min(1, size_consistency)))
                
                # 2. Семантическая когерентность (для семантических чанкеров)
                coherence_score = self._calculate_semantic_coherence(chunks)
                semantic_coherence_scores.append(coherence_score)
                
                # 3. Точность границ (нет разрывов предложений)
                boundary_score = self._calculate_boundary_accuracy(text, chunks)
                boundary_accuracy_scores.append(boundary_score)
                
            except Exception as e:
                error_count += 1
                print(f"    ⚠️ Ошибка при анализе надежности: {e}")
        
        return {
            'chunk_size_consistency': float(np.mean(chunk_size_consistency) if chunk_size_consistency else 0),
            'semantic_coherence': float(np.mean(semantic_coherence_scores) if semantic_coherence_scores else 0),
            'boundary_accuracy': float(np.mean(boundary_accuracy_scores) if boundary_accuracy_scores else 0),
            'error_rate': float(error_count / len(test_texts)),
            'success_rate': float(1 - (error_count / len(test_texts)))
        }
    
    def _calculate_semantic_coherence(self, chunks: List[str]) -> float:
        """Вычисляет семантическую когерентность чанков"""
        if len(chunks) < 2:
            return 1.0
        
        try:
            # Простая оценка на основе длины предложений в чанках
            coherence_scores = []
            
            for chunk in chunks:
                sentences = nltk.sent_tokenize(chunk)
                if len(sentences) > 1:
                    # Чанк более когерентен, если предложения примерно одинаковой длины
                    sent_lengths = [len(sent.split()) for sent in sentences]
                    if sent_lengths:
                        length_std = np.std(sent_lengths)
                        length_mean = np.mean(sent_lengths)
                        coherence = 1 - (length_std / length_mean) if length_mean > 0 else 0
                        coherence_scores.append(max(0, min(1, coherence)))
                    else:
                        coherence_scores.append(0.5)
                else:
                    coherence_scores.append(1.0)  # Одно предложение = максимальная когерентность
            
            return np.mean(coherence_scores) if coherence_scores else 0.5
            
        except Exception:
            return 0.5  # Средняя оценка при ошибке
    
    def _calculate_boundary_accuracy(self, original_text: str, chunks: List[str]) -> float:
        """Вычисляет точность границ чанков"""
        try:
            # Проверяем, что все чанки вместе дают исходный текст (или близко к нему)
            reconstructed_text = ' '.join(chunks)
            
            # Простая метрика: соотношение длин
            length_ratio = min(len(reconstructed_text), len(original_text)) / max(len(reconstructed_text), len(original_text))
            
            # Проверяем, что нет обрыва предложений в середине
            sentence_break_penalty = 0
            for chunk in chunks:
                # Чанк должен заканчиваться на знак препинания или начало нового предложения
                if chunk.strip() and not chunk.strip()[-1] in '.!?':
                    # Проверяем, что следующий чанк начинается с заглавной буквы
                    chunk_index = chunks.index(chunk)
                    if chunk_index < len(chunks) - 1:
                        next_chunk = chunks[chunk_index + 1].strip()
                        if next_chunk and not next_chunk[0].isupper():
                            sentence_break_penalty += 0.1
            
            boundary_score = length_ratio - sentence_break_penalty
            return max(0, min(1, boundary_score))
            
        except Exception:
            return 0.5
    
    def _calculate_psr_score(self, performance: Dict, scalability: Dict, 
                           reliability: Dict) -> Dict[str, float]:
        """Вычисляет общий PSR Score"""
        
        # Нормализуем метрики к шкале 0-1 (больше = лучше)
        
        # Performance Score (0-1)
        # Скорость обработки и низкое использование памяти
        perf_score = 0
        if performance['chars_per_second'] > 0:
            # Логарифмическая нормализация скорости (1000 символов/сек = 0.5)
            speed_score = min(1, np.log10(performance['chars_per_second'] + 1) / 4)
            perf_score += speed_score * 0.4
        
        # Низкое использование памяти (< 100MB = хорошо)
        memory_score = max(0, 1 - performance['avg_memory_usage'] / 100)
        perf_score += memory_score * 0.3
        
        # Низкий процент ошибок
        error_score = 1 - performance['error_rate']
        perf_score += error_score * 0.3
        
        # Scalability Score (0-1)
        scale_score = (
            scalability['scalability_factor'] * 0.5 +
            min(1, scalability['efficiency_ratio'] / 10) * 0.3 +  # 10 чанков/сек = хорошо
            (1 - abs(scalability['size_correlation'])) * 0.2  # Слабая корреляция = лучше
        )
        
        # Reliability Score (0-1)
        rel_score = (
            reliability['chunk_size_consistency'] * 0.3 +
            reliability['semantic_coherence'] * 0.3 +
            reliability['boundary_accuracy'] * 0.2 +
            reliability['success_rate'] * 0.2
        )
        
        # Общий PSR Score (взвешенная сумма)
        total_psr_score = (
            perf_score * 0.35 +    # Performance - 35%
            scale_score * 0.30 +   # Scalability - 30%
            rel_score * 0.35       # Reliability - 35%
        )
        
        return {
            'performance_score': perf_score,
            'scalability_score': scale_score,
            'reliability_score': rel_score,
            'total_psr_score': total_psr_score,
            'grade': self._get_grade(total_psr_score)
        }
    
    def _get_grade(self, score: float) -> str:
        """Преобразует PSR Score в буквенную оценку"""
        if score >= 0.9:
            return "A+"
        elif score >= 0.8:
            return "A"
        elif score >= 0.7:
            return "B+"
        elif score >= 0.6:
            return "B"
        elif score >= 0.5:
            return "C+"
        elif score >= 0.4:
            return "C"
        elif score >= 0.3:
            return "D"
        else:
            return "F"
    
    def get_comparison_report(self) -> Dict[str, Any]:
        """Создает сравнительный отчет по всем проанализированным методам"""
        if not self.results:
            return {}
        
        comparison = {
            'methods': list(self.results.keys()),
            'rankings': {
                'performance': [],
                'scalability': [],
                'reliability': [],
                'overall': []
            },
            'best_method': None,
            'recommendations': []
        }
        
        # Сортируем методы по каждой метрике
        for metric in ['performance_score', 'scalability_score', 'reliability_score', 'total_psr_score']:
            sorted_methods = sorted(
                self.results.items(),
                key=lambda x: x[1]['psr_score'][metric],
                reverse=True
            )
            
            metric_name = metric.replace('_score', '').replace('total_psr', 'overall')
            comparison['rankings'][metric_name] = [
                {
                    'method': method,
                    'score': data['psr_score'][metric],
                    'grade': data['psr_score']['grade']
                }
                for method, data in sorted_methods
            ]
        
        # Определяем лучший метод
        best_method = max(
            self.results.items(),
            key=lambda x: x[1]['psr_score']['total_psr_score']
        )
        comparison['best_method'] = best_method[0]
        
        # Генерируем рекомендации
        comparison['recommendations'] = self._generate_recommendations()
        
        return comparison
    
    def _generate_recommendations(self) -> List[str]:
        """Генерирует рекомендации на основе анализа"""
        recommendations = []
        
        if not self.results:
            return ["Нет данных для анализа"]
        
        # Анализируем результаты и даем рекомендации
        best_perf = max(self.results.items(), key=lambda x: x[1]['psr_score']['performance_score'])
        best_scale = max(self.results.items(), key=lambda x: x[1]['psr_score']['scalability_score'])
        best_rel = max(self.results.items(), key=lambda x: x[1]['psr_score']['reliability_score'])
        
        recommendations.append(f"🚀 Лучшая производительность: {best_perf[0]}")
        recommendations.append(f"📈 Лучшая масштабируемость: {best_scale[0]}")
        recommendations.append(f"🎯 Лучшая надежность: {best_rel[0]}")
        
        # Специфические рекомендации
        for method, data in self.results.items():
            psr = data['psr_score']
            if psr['performance_score'] < 0.5:
                recommendations.append(f"⚠️ {method}: Низкая производительность - рассмотрите оптимизацию")
            if psr['scalability_score'] < 0.5:
                recommendations.append(f"⚠️ {method}: Проблемы с масштабируемостью для больших текстов")
            if psr['reliability_score'] < 0.5:
                recommendations.append(f"⚠️ {method}: Низкая надежность - проверьте качество чанкования")
        
        return recommendations 