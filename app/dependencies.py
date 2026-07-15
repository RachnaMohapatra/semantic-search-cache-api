import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print(f"Loading Sentence Transformer model ({settings.MODEL_NAME})...")
model = SentenceTransformer(settings.MODEL_NAME)

print(f"Loading processed documents from {settings.DOCUMENTS_PATH}...")
with open(os.path.join(BASE_DIR, settings.DOCUMENTS_PATH), "r", encoding="utf-8") as f:
    clean_documents = [line.strip() for line in f]

print(f"Loading embeddings from {settings.EMBEDDINGS_PATH}...")
doc_embeddings = np.load(os.path.join(BASE_DIR, settings.EMBEDDINGS_PATH))

print(f"Loading FAISS index from {settings.INDEX_PATH}...")
index = faiss.read_index(os.path.join(BASE_DIR, settings.INDEX_PATH))

print("System Ready!")
