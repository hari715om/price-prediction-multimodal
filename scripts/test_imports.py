# scripts/test_imports.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocess_text import TextPreprocessor
print("✅ Import successful!")
