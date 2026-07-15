def test_home_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "documents" in data
    assert "cache_entries" in data


def test_query_success(client):
    response = client.post("/query", json={"query": "wireless earbuds"})
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "wireless earbuds"
    assert data["cache_hit"] is False
    assert len(data["results"]) == 5
    assert "text" in data["results"][0]


def test_query_cache_hit(client):
    # First search (miss)
    client.post("/query", json={"query": "gaming mouse"})

    # Second search (hit)
    response = client.post("/query", json={"query": "gaming mouse"})
    assert response.status_code == 200
    data = response.json()
    assert data["cache_hit"] is True


def test_query_validation_too_long(client):
    long_query = "a" * 501
    response = client.post("/query", json={"query": long_query})
    assert response.status_code == 422  # Pydantic validation error


def test_query_validation_empty(client):
    response = client.post("/query", json={"query": ""})
    assert response.status_code == 422  # Pydantic validation error


def test_query_validation_whitespace(client):
    response = client.post("/query", json={"query": "    "})
    assert response.status_code == 400  # Custom HTTP exception for empty after strip
