#!/usr/bin/env python3
"""
Скрипт для запуска приложения SuperChunker
"""

import subprocess
import sys
import os

def check_dependencies():
    """Проверяет установленность зависимостей"""
    required_packages = [
        'streamlit', 'langchain', 'tiktoken', 'nltk', 
        'plotly', 'pandas', 'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Отсутствуют следующие пакеты:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n🔧 Для установки выполните:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ Все зависимости установлены")
    return True

def download_nltk_data():
    """Загружает необходимые данные NLTK"""
    try:
        import nltk
        print("📦 Загружаем данные NLTK...")
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        print("✅ Данные NLTK загружены")
    except Exception as e:
        print(f"⚠️ Ошибка при загрузке данных NLTK: {e}")

def main():
    """Основная функция запуска"""
    print("🚀 Запуск SuperChunker...")
    print("=" * 50)
    
    # Проверяем зависимости
    if not check_dependencies():
        sys.exit(1)
    
    # Загружаем данные NLTK
    download_nltk_data()
    
    # Запускаем Streamlit
    print("\n🌐 Запускаем веб-приложение...")
    print("📍 Приложение будет доступно по адресу: http://localhost:8501")
    print("💡 Для остановки нажмите Ctrl+C")
    print("=" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\n👋 Приложение остановлено")

if __name__ == "__main__":
    main() 