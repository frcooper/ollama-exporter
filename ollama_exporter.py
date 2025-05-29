import os
import time
import httpx
from fastapi import FastAPI, Request, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Configurable Ollama host (via env variable or defaults to localhost)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

app = FastAPI()

# Prometheus Metrics
REQUEST_COUNT = Counter("ollama_requests_total", "Total Ollama requests", ["model"])
RESPONSE_TIME = Histogram("ollama_response_seconds", "Ollama response time", ["model"])
TOKEN_COUNT = Counter("ollama_tokens_generated_total", "Total tokens generated", ["model"])
EVAL_COUNT = Counter("ollama_eval_total", "Total evaluation steps performed", ["model"])
LOAD_TIME = Histogram("ollama_load_time_seconds", "Time taken to load models", ["model"])

@app.get("/metrics")
def metrics():
    """Expose Prometheus metrics."""
    return Response(content=generate_latest(), media_type="text/plain")

@app.post("/api/generate")
async def generate(request: Request):
    """Intercept Ollama generation requests, record metrics, and proxy to Ollama."""
    start_time = time.time()
    body = await request.json()
    model = body.get("model", "unknown")

    async with httpx.AsyncClient() as client:
        response = await client.post(f"{OLLAMA_HOST}/api/generate", json=body)
    
    elapsed_time = time.time() - start_time
    response_data = response.json()

    # Extract relevant metrics
    token_count = response_data.get("eval_count", 0)
    eval_count = response_data.get("num_prompt_tokens", 0)
    load_time = response_data.get("load_time", 0.0)

    # Update Prometheus metrics
    REQUEST_COUNT.labels(model=model).inc()
    RESPONSE_TIME.labels(model=model).observe(elapsed_time)
    TOKEN_COUNT.labels(model=model).inc(token_count)
    EVAL_COUNT.labels(model=model).inc(eval_count)
    LOAD_TIME.labels(model=model).observe(load_time)

    return response_data

@app.get("/api/tags")
async def tags():
    """Proxy Ollama model tags API."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{OLLAMA_HOST}/api/tags")
    return response.json()
