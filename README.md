# Ollama Prometheus Exporter

This is a **Prometheus Exporter** for **Ollama**, designed to monitor request statistics, response times, token usage, and model performance. It runs as a FastAPI service and is **Docker-ready**.

## Features
- **Tracks requests per model** (`ollama_requests_total`)
- **Measures response time** (`ollama_response_seconds`)
- **Records model load times** (`ollama_load_duration_seconds`)
- **Tracks evaluation durations** (`ollama_prompt_eval_duration_seconds` and `ollama_eval_duration_seconds`)
- **Monitors token usage** (`ollama_tokens_processed_total` and `ollama_tokens_generated_total`)
- **Measures token generation rate** (`ollama_tokens_per_second`)
- **Tracks in-flight inference requests** (`ollama_inflight_requests`)
- **Transparent streaming proxy** for every Ollama API endpoint, including the
  OpenAI-compatible `/v1/*` API
- **Cancels abandoned generations**: when a client disconnects, the upstream
  request is closed and Ollama stops generating
- **Total deadline per inference request**, so no generation can run unbounded

## Installation

### Running Locally

#### 1. Install Dependencies
```sh
pip install fastapi uvicorn prometheus_client httpx anyio
```

#### 2. Run the Exporter
```sh
python ollama_exporter.py
```
By default, it connects to `http://localhost:11434` for Ollama and listens on
`[::]:8000`, a dual-stack socket reachable over both IPv6 and IPv4.

#### Configuration
Every option can be set by environment variable or CLI flag (the flag wins).

| Environment variable | Flag | Default | Description |
|----------------------|------|---------|-------------|
| `OLLAMA_HOST` | `--ollama-host` | `http://localhost:11434` | Upstream Ollama URL |
| `EXPORTER_HOST` | `--host` | `::` | Address to bind (dual-stack) |
| `EXPORTER_PORT` | `--port` | `8000` | Port to listen on |
| `EXPORTER_REQUEST_TIMEOUT` | `--request-timeout` | `1800` | Total deadline in seconds for one inference request, `0` disables it |
| `LOG_LEVEL` | `--log-level` | `INFO` | Logging verbosity |

The request deadline covers the whole exchange, from sending the request to
the last streamed byte. It applies to inference only: `POST` on `/api/chat`,
`/api/generate`, `/api/embed`, `/api/embeddings` and any `/v1/*` endpoint.
Model pulls, pushes and listings are not limited. A request that hits the
deadline before Ollama answered gets a `504`; one that is already streaming
has its stream ended. Either way the upstream connection is closed, which is
what makes Ollama stop generating.

#### 3. Run the Tests (optional)
```sh
pip install -r requirements-dev.txt
pytest
```

### Running with Docker

#### 1. Build the Docker Image
```sh
docker build -t ollama-exporter .
```

#### 2. Run the Container
```sh
docker run -d --name ollama-exporter -p 8000:8000 \
  -e OLLAMA_HOST="http://192.168.1.100:11434" ollama-exporter
```

Arguments appended to `docker run` are passed straight to the exporter CLI, so
the flags of the local install work the same way in a container:
```sh
docker run -d --name ollama-exporter -p 9000:9000 ollama-exporter \
  --ollama-host http://192.168.1.100:11434 --port 9000 --log-level DEBUG
```

## Prometheus Integration

### Add to `prometheus.yml`
```yaml
scrape_configs:
  - job_name: 'ollama-metrics'
    static_configs:
      - targets: ['192.168.1.100:8000']
```
Restart Prometheus to apply changes:
```sh
docker restart <prometheus-container-name>
```

## Metrics
| Metric Name | Description |
|------------|-------------|
| `ollama_requests_total` | Total chat and generate requests (`/api/chat`, `/api/generate`, `/v1/chat/completions`, `/v1/completions`) |
| `ollama_inflight_requests` | Inference requests currently being proxied to Ollama, per model |
| `ollama_response_seconds` | Total time spent for the response |
| `ollama_load_duration_seconds` | Time spent loading the model |
| `ollama_prompt_eval_duration_seconds` | Time spent evaluating prompt |
| `ollama_eval_duration_seconds` | Time spent generating the response |
| `ollama_tokens_processed_total` | Number of tokens in the prompt |
| `ollama_tokens_generated_total` | Number of tokens in the response |
| `ollama_tokens_per_second` | Tokens generated per second |

Durations and tokens per second come from Ollama's native stats, so only
`/api/chat` and `/api/generate` feed them. The OpenAI-compatible
`/v1/chat/completions` and `/v1/completions` feed the two token counters from
their `usage` object; for streamed responses that object is only sent when the
client sets `stream_options.include_usage`.

## Grafana Integration
1. Open **Grafana**.
2. Go to **Dashboards → Import**.
3. Click **Upload JSON file** and select `dashboard.json` from the project directory.
4. Select your **Prometheus data source**.
5. Click **Import** to add the dashboard.

## API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/metrics` | GET | Exposes Prometheus metrics |
| `/api/chat` | POST | Proxies requests to Ollama and logs metrics |
| `/api/generate` | POST | Proxies requests to Ollama and logs metrics |
| `/v1/chat/completions`, `/v1/completions` | POST | Proxies requests to Ollama and logs request and token metrics |

All other endpoints are proxied to the Ollama API.

## Usage Scenario

Suppose you want to monitor your local Ollama instance with Prometheus using this exporter:

1. **Start Ollama** locally (default: `http://localhost:11434`).
2. **Run the exporter** on your machine:
   ```sh
   OLLAMA_HOST=http://localhost:11434 python ollama_exporter.py
   # or with Docker:
   # docker run -d -p 8000:8000 -e OLLAMA_HOST="http://localhost:11434" ollama-exporter
   ```
3. **Configure your application (eg Open WebUI)** to use the exporter as the API endpoint:
   - Set `OLLAMA_HOST=http://localhost:8000` (the exporter will proxy and collect metrics).
4. **Prometheus** scrapes metrics from the exporter:
   - Add `localhost:8000/metrics` to your Prometheus scrape config.

This setup allows you to transparently monitor all Ollama API usage and performance via Prometheus and Grafana dashboards.

## License
There is no spoon.
