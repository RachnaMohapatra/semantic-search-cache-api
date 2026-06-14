# Semantic Product Search Engine with Intelligent Caching

A semantic product search engine built using **Sentence Transformers, FAISS, FastAPI, and Intelligent Query Caching**.

Unlike traditional keyword-based search systems, this project retrieves products based on their semantic meaning, enabling users to discover relevant products even when exact keywords are not present.

The system combines transformer-based embeddings, vector similarity search, caching, and REST APIs to create a scalable and efficient semantic retrieval pipeline.

---

## Features

### Semantic Search

Uses Sentence Transformer embeddings to understand the intent and meaning behind user queries.

### Vector Similarity Search

Uses FAISS (Facebook AI Similarity Search) for fast nearest-neighbor retrieval over thousands of product embeddings.

### Intelligent Query Caching

Frequently searched queries are stored in cache.

Benefits:

* Faster repeated searches
* Reduced computation
* Improved response time

### Cache Monitoring

Tracks:

* Cache Hits
* Cache Misses
* Hit Rate
* Total Cache Entries

### FastAPI Backend

REST API endpoints for:

* Product Search
* Cache Statistics
* Cache Management

### Interactive API Documentation

FastAPI automatically generates Swagger UI documentation for testing and exploring endpoints.

---

## Dataset

The project uses a Flipkart Product Dataset containing approximately 12,000+ products across multiple categories.

Categories include:

* Electronics
* Mobile Accessories
* Earbuds
* Headphones
* Speakers
* Home Appliances
* Furniture
* Lifestyle Products

The raw dataset is stored as:

```text
data/dataset.csv
```

The processed search corpus is stored as:

```text
clean_documents.txt
```

Each document contains structured product information including:

* Product Title
* Description
* Product Rating
* Seller Rating
* Selling Price
* Category Information

---

## Embedding Model

The project uses:

```text
all-MiniLM-L6-v2
```

from Sentence Transformers.

This model converts product documents and user queries into dense vector embeddings for semantic retrieval.

---

## Search Pipeline

```text
User Query
     │
     ▼
Sentence Transformer
     │
     ▼
Query Embedding
     │
     ▼
Cache Check
     │
 ┌───┴────┐
 │        │
 ▼        ▼
Hit      Miss
 │        │
 │        ▼
 │    FAISS Search
 │        │
 │        ▼
 │   Top Results
 │        │
 └────────┘
      │
      ▼
Return Response
```

---

## Technologies Used

* Python
* FastAPI
* Sentence Transformers
* FAISS
* NumPy
* Pandas
* Scikit-Learn
* Uvicorn
* HTML
* CSS
* JavaScript
* Docker

---

## Project Structure

```text
semantic-search-engine/

├── app.py
├── README.md
├── requirements.txt
├── Dockerfile

├── clean_documents.txt
├── product_embeddings.npy
├── product_index.faiss

├── 01_dataset_and_embeddings.ipynb

├── index.html

├── data/
│   └── dataset.csv

└── .gitignore
```

---

## API Endpoints

### Home

```http
GET /
```

Response:

```json
{
  "message": "Semantic Product Search API Running"
}
```

---

### Search Products

```http
POST /query
```

Request:

```json
{
  "query": "wireless earbuds under 1000"
}
```

Response:

```json
{
  "query": "wireless earbuds under 1000",
  "cache_hit": false,
  "top_match_id": 2019,
  "distance": 0.76,
  "results": [...]
}
```

---

### Cache Statistics

```http
GET /cache/stats
```

Response:

```json
{
  "total_queries": 2,
  "cache_entries": 1,
  "hit_count": 1,
  "miss_count": 1,
  "hit_rate": "50.00%"
}
```

---

### Clear Cache

```http
DELETE /cache
```

Clears all cached queries and resets cache statistics.

---

## Running the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Start FastAPI Server

```bash
uvicorn app:app --reload
```

### Open Swagger Documentation

```text
http://127.0.0.1:8000/docs
```

---

## Sample Queries

```text
wireless earbuds under 1000

bluetooth speaker

soundbar under 1000

gaming headphones

wireless earbuds under 500
```

---

## Experimental Work

The accompanying notebook includes additional experimentation with:

* PCA (Principal Component Analysis)
* Gaussian Mixture Model (GMM) Clustering
* Cluster Analysis
* Semantic Cache Exploration

These experiments were used to analyze product embedding distributions and clustering behavior.

---

## Future Improvements

* Personalized Recommendations
* Hybrid Semantic + Keyword Search
* Real-Time Product Updates
* Cluster-Aware Semantic Caching
* Multi-Language Search
* Retrieval-Augmented Product Assistant

---

## Author

### Rachna Mohapatra

Electronics and Computer Engineering

Areas of Interest:

* Artificial Intelligence
* Semantic Search
* Machine Learning Systems
* Information Retrieval
* AI Infrastructure
