# Ollama Prometheus Exporter

This is a **Prometheus Exporter** for **Ollama**, designed to monitor request statistics, response times, token usage, and model performance. It runs as a FastAPI service and is **Docker-ready**.

## Features
- **Tracks chat, generation, and embedding requests per model** (`ollama_requests_total`)
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
Use Python 3.11 or newer. A virtual environment is recommended.

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
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
python -m pip install -r requirements-dev.txt
python -m pytest -q
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

If Ollama runs on the Docker host, Docker Desktop supports
`OLLAMA_HOST=http://host.docker.internal:11434`. On Linux, add
`--add-host=host.docker.internal:host-gateway`. `localhost` inside the container
refers to the container itself. Ollama must listen on an address reachable from
the container. If both services share a Docker network, use the Ollama service
name, for example `http://ollama:11434`.

The container runs as an unprivileged user. The exporter forwards Ollama's API,
including model management operations, and adds no authentication of its own.
Keep it on a trusted network or behind an authenticated reverse proxy.

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
| `ollama_requests_total` | Total chat, generate, and embedding requests, counted when received (including failed attempts); includes `/v1/chat/completions` and `/v1/completions` |
| `ollama_inflight_requests` | Inference requests currently being proxied to Ollama, per model |
| `ollama_response_seconds` | Total time spent for the response |
| `ollama_load_duration_seconds` | Time spent loading the model |
| `ollama_prompt_eval_duration_seconds` | Time spent evaluating prompt |
| `ollama_eval_duration_seconds` | Time spent generating the response |
| `ollama_tokens_processed_total` | Number of tokens in the prompt |
| `ollama_tokens_generated_total` | Number of tokens in the response |
| `ollama_tokens_per_second` | Tokens generated per second |

Generation durations and tokens per second come from Ollama's native stats on
`/api/chat` and `/api/generate`. The OpenAI-compatible
`/v1/chat/completions` and `/v1/completions` feed the two token counters from
their `usage` object; for streamed responses that object is only sent when the
client sets `stream_options.include_usage`.

`POST /api/embed` records input tokens, total duration, and load duration when
Ollama includes them in the response. Embeddings produce no generated tokens.
The older `/api/embeddings` endpoint counts requests and records only the stats
its response actually contains; it may return just an embedding vector.
Embedding calls use the same relay, cancellation, and deadline handling as
chat and generation requests. Metric names and the `model` label are unchanged.

Counters reset when the exporter restarts. Prometheus retains the scraped
history; use `rate()` for rates and `increase()` for totals over a time range,
rather than displaying a raw counter as an all-time total. The dashboards use
`increase(ollama_requests_total[$__range])` for selected-range totals. These are
estimates that need at least two scraped samples per series; requests before
the first scrape or during scrape gaps cannot be recovered exactly.

## Grafana Integration
1. Open **Grafana**.
2. Go to **Dashboards → Import**.
3. Click **Upload JSON file** and select `dashboard.json` or `dashboard_custom.json` from the project directory.
4. Click **Import** to add the dashboard.
5. Select your **Prometheus** data source in the dashboard's selector.

## API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/metrics` | GET | Exposes Prometheus metrics |
| `/api/chat` | POST | Proxies requests to Ollama and logs metrics |
| `/api/generate` | POST | Proxies requests to Ollama and logs metrics |
| `/api/embed` | POST | Proxies embedding requests and records input tokens and timings |
| `/api/embeddings` | POST | Proxies legacy embedding requests and records available metrics |
| `/v1/chat/completions`, `/v1/completions` | POST | Proxies requests to Ollama and logs request and token metrics |

All other endpoints are proxied to the Ollama API.

## Usage Scenario

Suppose you want to monitor your local Ollama instance with Prometheus using this exporter:

1. **Start Ollama** locally (default: `http://localhost:11434`).
2. **Run the exporter** on your machine:
   ```sh
   OLLAMA_HOST=http://localhost:11434 python ollama_exporter.py
   # or with Docker:
   # docker run -d -p 8000:8000 -e OLLAMA_HOST="http://host.docker.internal:11434" ollama-exporter
   ```
3. **Configure your application (eg Open WebUI)** to use the exporter as the API endpoint:
   - Set `OLLAMA_HOST=http://localhost:8000` (the exporter will proxy and collect metrics).
4. **Prometheus** scrapes metrics from the exporter:
   - Add `localhost:8000/metrics` to your Prometheus scrape config.

This setup routes Ollama traffic through the exporter for monitoring with
Prometheus and Grafana.

Only requests sent through the exporter can be counted. The native chat,
generate, and embedding endpoints, plus `/v1/chat/completions` and
`/v1/completions`, are instrumented. Other endpoints are forwarded without
collecting request and token metrics.

## Development

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m pip check
```

Tests use a simulated upstream and do not require Ollama, downloaded models, or
a GPU. CI runs the tests on Python 3.11–3.14 and checks that the container builds.
Direct dependencies are pinned in `requirements.txt`; Docker uses that same
file. Dependabot checks Python packages, the Docker base, and Actions monthly.
The CI workflow validates changes and does not publish images or releases.

## License
There is no spoon.
