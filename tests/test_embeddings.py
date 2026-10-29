import gzip
import itertools
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

import ollama_exporter as exporter


models = itertools.count()


@pytest.fixture
def proxy(monkeypatch):
    """Exercise the ASGI routes against an in-memory Ollama HTTP transport."""
    real_client = httpx.AsyncClient

    def configure(handler):
        transport = httpx.MockTransport(handler)
        monkeypatch.setattr(exporter.httpx, "AsyncClient", lambda **kwargs: real_client(transport=transport, **kwargs))
        monkeypatch.setattr(exporter, "OLLAMA_HOST", "http://ollama.test/")
        return TestClient(exporter.app)

    return configure


def sample(name, model):
    return REGISTRY.get_sample_value(name, {"model": model})


@pytest.mark.parametrize("path", ["/api/embed", "/api/embeddings"])
def test_embedding_stats_and_wire_response(proxy, path):
    model = f"embedding-{next(models)}"
    body = {"model": model, "input": ["one", "two"]}
    stats = {
        "model": model, "embeddings": [[0.1, 0.2]],
        "total_duration": 2_000_000_000, "load_duration": 500_000_000,
        "prompt_eval_count": 8,
    }
    wire = json.dumps(stats).encode()

    def upstream(request):
        assert request.url.path == path
        assert request.url.params.get_list("tag") == ["a", "b"]
        assert json.loads(request.content) == body
        assert request.headers["authorization"] == "Bearer test-token"
        return httpx.Response(200, content=wire, headers={"content-type": "application/json", "x-ollama": "ok"})

    with proxy(upstream) as client:
        response = client.post(path + "?tag=a&tag=b", json=body, headers={"authorization": "Bearer test-token"})
        assert response.content == wire
        assert response.status_code == 200
        assert response.headers["x-ollama"] == "ok"
        assert response.headers["content-type"] == "application/json"
        assert sample("ollama_requests_total", model) == 1
        assert sample("ollama_tokens_processed_total", model) == 8
        assert sample("ollama_response_seconds_sum", model) == 2
        assert sample("ollama_load_duration_seconds_sum", model) == 0.5
        assert sample("ollama_tokens_generated_total", model) is None
        assert sample("ollama_tokens_per_second_count", model) is None
        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert metrics.headers["content-type"].startswith("text/plain")
        assert f'ollama_tokens_processed_total{{model="{model}"}} 8.0' in metrics.text


def test_legacy_response_without_telemetry_still_counts(proxy):
    model = f"legacy-{next(models)}"
    with proxy(lambda request: httpx.Response(200, json={"embedding": [0.1]})) as client:
        response = client.post("/api/embeddings", json={"model": model, "prompt": "hello"})
    assert response.json() == {"embedding": [0.1]}
    assert sample("ollama_requests_total", model) == 1
    assert sample("ollama_tokens_processed_total", model) is None


@pytest.mark.parametrize("status,content", [(400, b'{"error":"bad input"}'), (404, b'{"error":"model missing"}'), (500, b'failure'), (200, b'not json'), (200, b'[]')])
def test_unusual_upstream_responses_pass_through(proxy, status, content):
    model = f"error-{next(models)}"
    with proxy(lambda request: httpx.Response(status, content=content)) as client:
        response = client.post("/api/embed", json={"model": model, "input": "hello"})
    assert response.status_code == status
    assert response.content == content
    assert sample("ollama_requests_total", model) == 1
    assert sample("ollama_tokens_processed_total", model) is None


@pytest.mark.parametrize("body", [b'not json', b'[]', b'{"model":null}', b'{"model":{}}'])
def test_invalid_requests_are_left_to_ollama(proxy, body):
    def upstream(request):
        assert request.content == body
        return httpx.Response(400, json={"error": "invalid request"})

    with proxy(upstream) as client:
        assert client.post("/api/embed", content=body).status_code == 400


@pytest.mark.parametrize("error,status", [(httpx.ConnectError, 502), (httpx.ReadTimeout, 504)])
def test_upstream_failures_have_gateway_status(proxy, error, status):
    def upstream(request):
        raise error("test failure", request=request)

    with proxy(upstream) as client:
        assert client.post("/api/embed", json={"model": "embedding"}).status_code == status


def test_decoded_response_headers_and_repeated_headers(proxy):
    wire = b'{"embeddings":[[0.1]]}'
    def upstream(request):
        return httpx.Response(200, content=gzip.compress(wire), headers=[
            ("content-encoding", "gzip"), ("transfer-encoding", "chunked"),
            ("connection", "x-private"), ("x-private", "secret"),
            ("set-cookie", "a=1"), ("set-cookie", "b=2"),
        ])

    with proxy(upstream) as client:
        response = client.post("/api/embed", json={"model": "embedding"})
    assert response.content == wire
    # Streaming the decoded body prevents us from knowing its size in advance.
    assert "content-length" not in response.headers
    assert not {"content-encoding", "transfer-encoding", "connection", "x-private"} & response.headers.keys()
    assert response.headers.get_list("set-cookie") == ["a=1", "b=2"]


def test_request_framing_and_connection_headers_are_not_forwarded(proxy):
    def upstream(request):
        assert request.headers["host"] == "ollama.test"
        assert "transfer-encoding" not in request.headers
        assert "x-private" not in request.headers
        assert json.loads(request.content) == {"model": "embedding"}
        return httpx.Response(200, json={"embeddings": [[0.1]]})

    with proxy(upstream) as client:
        assert client.post("/api/embed", json={"model": "embedding"}, headers={
            "connection": "x-private", "x-private": "secret", "transfer-encoding": "chunked",
        }).status_code == 200


@pytest.mark.parametrize("value", [None, "8", {}, [], -1, True, float("nan"), float("inf"), 10**1000])
def test_bad_telemetry_does_not_interrupt_proxy(value):
    model = f"invalid-{next(models)}"
    exporter.extract_and_record_metrics({
        "total_duration": value, "load_duration": value, "prompt_eval_duration": value,
        "prompt_eval_count": value, "eval_duration": value, "eval_count": value,
    }, model)
    assert sample("ollama_tokens_processed_total", model) is None


@pytest.mark.parametrize("path", ["/api/chat", "/api/generate"])
def test_existing_nonstreaming_metrics_remain_compatible(proxy, path):
    model = f"chat-{next(models)}"
    stats = {"done": True, "eval_count": 12, "eval_duration": 2_000_000_000}
    with proxy(lambda request: httpx.Response(200, json=stats)) as client:
        assert client.post(path, json={"model": model, "stream": False}).json() == stats
    assert sample("ollama_tokens_generated_total", model) == 12
    assert sample("ollama_tokens_per_second_sum", model) == 6
