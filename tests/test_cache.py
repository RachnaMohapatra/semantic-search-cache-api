def test_cache_lifecycle(client):
    # Clear cache first
    client.delete("/cache")

    # Query something to populate cache
    client.post("/query", json={"query": "mechanical keyboard"})

    # Check stats
    response = client.get("/cache/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["cache_entries"] == 1
    assert data["miss_count"] == 1
    assert data["hit_count"] == 0

    # Query again (hit)
    client.post("/query", json={"query": "mechanical keyboard"})

    response = client.get("/cache/stats")
    data = response.json()
    assert data["hit_count"] == 1

    # Clear cache
    response = client.delete("/cache")
    assert response.status_code == 200
    assert response.json() == {"message": "Cache cleared successfully"}

    # Check stats again
    response = client.get("/cache/stats")
    data = response.json()
    assert data["cache_entries"] == 0
    assert data["hit_count"] == 0
    assert data["miss_count"] == 0
