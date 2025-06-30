# Ollama Prometheus Exporter

This is a **Prometheus Exporter** for **Ollama**, designed to monitor request statistics, response times, token usage, and model performance. It runs as a FastAPI service and is **Docker-ready**.

## Features
- **Tracks requests per model** (`ollama_requests_total`)
- **Measures response time** (`ollama_response_seconds`)
- **Records model load times** (`ollama_load_duration_seconds`)
- **Tracks evaluation durations** (`ollama_prompt_eval_duration_seconds` and `ollama_eval_duration_seconds`)
- **Monitors token usage** (`ollama_tokens_processed_total` and `ollama_tokens_generated_total`)
- **Measures token generation rate** (`ollama_tokens_per_second`)
- **Transparent proxy** for all non-chat Ollama API endpoints

## Installation

### Running Locally

#### 1. Install Dependencies
```sh
pip install fastapi uvicorn prometheus_client httpx
```

#### 2. Run the Exporter
```sh
python ollama_exporter.py
```
By default, it connects to `http://localhost:11434` for Ollama.

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
| `ollama_requests_total` | Total chat and generate requests |
| `ollama_response_seconds` | Total time spent for the response |
| `ollama_load_duration_seconds` | Time spent loading the model |
| `ollama_prompt_eval_duration_seconds` | Time spent evaluating prompt |
| `ollama_eval_duration_seconds` | Time spent generating the response |
| `ollama_tokens_processed_total` | Number of tokens in the prompt |
| `ollama_tokens_generated_total` | Number of tokens in the response |
| `ollama_tokens_per_second` | Tokens generated per second |

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
