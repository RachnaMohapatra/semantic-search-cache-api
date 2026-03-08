# Semantic Search API with Intelligent Caching

A semantic document search system built using **Sentence Transformers, FAISS, and FastAPI**.  
The system retrieves documents based on **semantic meaning rather than keyword matching** and uses a **semantic cache** to improve performance for repeated or similar queries.

This project demonstrates an end-to-end semantic retrieval pipeline including **embedding generation, vector search, caching, API development, and a frontend UI**.

---

# Key Features

• Semantic document search using **Sentence Transformer embeddings**  
• Fast similarity retrieval using **FAISS vector database**  
• **Semantic caching layer** for faster repeated queries  
• REST API built with **FastAPI**  
• Interactive API testing using **Swagger UI**  
• **Frontend search interface** for querying the API  
• Containerized deployment using **Docker**

---

# Dataset

This project uses the **20 Newsgroups dataset**, containing approximately **18,846 discussion posts** across multiple topics including:

- Space
- Sports
- Politics
- Religion
- Technology
- Medicine
- Automobiles

After preprocessing and cleaning, the documents are stored in:

```
clean_documents.txt
```

Each line represents **one document used for semantic search**.

---

# System Architecture

```
User Query
   ↓
Sentence Transformer
   ↓
Query Embedding
   ↓
Check Semantic Cache
   ↓
Cache HIT → Return Cached Results
   ↓
Cache MISS
   ↓
FAISS Vector Search
   ↓
Retrieve Top Similar Documents
   ↓
Store Results in Cache
   ↓
Return API Response
```

---

# Tech Stack

- Python
- FastAPI
- Sentence Transformers
- FAISS (Facebook AI Similarity Search)
- Scikit-learn
- NumPy
- Docker
- Uvicorn
- HTML / CSS / JavaScript (Frontend)

---

# API Endpoints

## Search Documents

```
POST /query
```

Example Request

```json
{
  "query": "mars exploration missions"
}
```

Example Response

```json
{
  "query": "mars exploration missions",
  "cache_hit": false,
  "similarity_score": 0.32,
  "dominant_cluster": 0,
  "results": [
    {
      "text": "NASA launched a Mars rover mission to study planetary geology.",
      "distance": 0.68
    }
  ]
}
```

---

## Cache Statistics

```
GET /cache/stats
```

Example Response

```json
{
  "total_entries": 5,
  "hit_count": 2,
  "miss_count": 3,
  "hit_rate": "40%"
}
```

---

## Clear Cache

```
DELETE /cache
```

Clears all cached queries.

---

# Frontend UI

A simple search interface is included to interact with the API.

Open:

```
frontend/index.html
```

The interface allows users to:

• Enter semantic queries  
• View retrieved documents  
• Observe **cache hits and misses**

---

# Running the Project

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Start the FastAPI server

```bash
uvicorn app:app --reload --port 8000
```

---

## 3. Open API documentation

```
http://127.0.0.1:8000/docs
```

Swagger UI allows you to **interactively test all endpoints**.

---

## 4. Launch the frontend UI

Open:

```
frontend/index.html
```

Then search queries directly from the browser.

---

# Docker Deployment

Build Docker image

```bash
docker build -t semantic-cache-api .
```

Run container

```bash
docker run -p 8000:8000 semantic-cache-api
```

Open:

```
http://localhost:8000/docs
```

---

# Example Queries

Try queries such as:

```
mars exploration missions
nasa missions to mars
hockey team players
car engine performance
medical disease treatment
```

Running similar queries repeatedly demonstrates the **semantic caching mechanism**.

---

# Author

**Rachna Mohapatra**  
Electronics and Computer Engineering

Interests:

- AI Systems
- Semantic Search
- Machine Learning Infrastructure
