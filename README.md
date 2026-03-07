# Semantic Search API with Intelligent Caching

FastAPI based semantic search system using **Sentence Transformers** and **FAISS** with an intelligent caching layer for faster repeated queries.

This project implements a semantic document search system that retrieves documents based on **semantic meaning instead of keyword matching** and uses a **semantic cache** to improve performance for repeated or similar queries.

---

## Key Features

* Semantic document search using **Sentence Transformer embeddings**
* Fast similarity retrieval using **FAISS vector database**
* Fuzzy clustering of documents using **Gaussian Mixture Models**
* **Semantic caching** to reuse results for similar queries
* REST API built with **FastAPI**
* Interactive API testing using **Swagger UI**
* Containerized deployment using **Docker**

---

## Dataset

This project uses the **20 Newsgroups dataset**, a well-known dataset containing approximately **18,846 discussion posts** across multiple topics such as:

* Space
* Sports
* Politics
* Religion
* Technology
* Medicine
* Automobiles

After preprocessing and cleaning, the documents are stored in:

```
clean_documents.txt
```

Each line represents a single cleaned document used for semantic search.

---

## System Architecture

```
User Query
   ↓
Sentence Transformer → Convert query to embedding
   ↓
Cluster Detection
   ↓
Check Semantic Cache
```

If similar query exists → **Cache Hit** → Return cached results

Else → **Cache Miss**

```
FAISS Vector Search
   ↓
Retrieve Top Similar Documents
   ↓
Store query + results in cache
   ↓
Return API Response
```

---

## Tech Stack

* Python
* FastAPI
* Sentence Transformers
* FAISS (Facebook AI Similarity Search)
* Scikit-learn
* NumPy
* Docker
* Uvicorn

---

## API Endpoints

### Search Documents

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

### Cache Statistics

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

### Clear Cache

```
DELETE /cache
```

Clears all cached queries.

---

## Running the Project

Install dependencies

```
pip install -r requirements.txt
```

Run the API server

```
uvicorn app:app --reload --port 8000
```

Open API documentation

```
http://127.0.0.1:8000/docs
```

Swagger UI allows you to interactively test all API endpoints.

---

## Docker Deployment

Build Docker image

```
docker build -t semantic-cache-api .
```

Run container

```
docker run -p 8000:8000 semantic-cache-api
```

Then open:

```
http://localhost:8000/docs
```

---

## Example Queries

Try queries like:

* mars exploration missions
* nasa missions to mars
* hockey team players
* car engine performance
* medical disease treatment

Similar queries may trigger **cache hits**, demonstrating the semantic caching mechanism.

---

## Author

**Rachna Mohapatra**
Electronics and Computer Engineering

Interests:

* AI Systems
* Semantic Search
* Machine Learning Infrastructure
