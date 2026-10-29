# Ollama Prometheus Exporter

A Prometheus exporter and HTTP proxy for Ollama. Point applications at the
exporter to collect request counts, timings, and token usage while forwarding
their requests to Ollama. Run it locally or in Docker.

```text
Application -> Exporter :8000 -> Ollama :11434
Prometheus  -> Exporter :8000/metrics
```

Requests sent directly to Ollama are not visible to the exporter.

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

### Pinning a Release

Choose a version from [GitHub Releases](https://github.com/frcooper/ollama-exporter/releases),
then download its source archive or check out its tag:

```sh
git clone --branch v0.1.0 --depth 1 https://github.com/frcooper/ollama-exporter.git
cd ollama-exporter
```

Use that checkout for the Python, Docker, or WSLC instructions below. Releases
include both Grafana dashboards. Container images are built locally; this
project does not publish images. Published release tags are immutable.

### Running Locally

#### 1. Install Dependencies
Use Python 3.11 or newer. A virtual environment is recommended.

Linux or macOS:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell (activation is not required):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Replace `-3.12` with your installed Python version, 3.11 or newer.

#### 2. Run the Exporter
```sh
python ollama_exporter.py
```

On Windows, use `.\.venv\Scripts\python.exe ollama_exporter.py`.

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
deadline before Ollama answered gets a `504`. If the response has already
started, the connection closes with an incomplete response so the client can
detect the interruption. Either way the upstream connection is closed, which
is what makes Ollama stop generating. Disabling the total deadline leaves
the upstream connection and read timeouts in place (900 seconds each).

#### 3. Connect Your Application

Set your application's Ollama connection URL to the exporter, for example
`http://localhost:8000`, using an address reachable from that application.
Set the exporter's `OLLAMA_HOST` to the actual Ollama server.

For Open WebUI, use **Admin Panel > Settings > Connections** to set its Ollama
URL. For a new deployment, `OLLAMA_BASE_URL=http://localhost:8000` can provide
the initial setting. Saved connection settings can override environment
variables; see the [Open WebUI connection documentation](https://docs.openwebui.com/troubleshooting/connection-error/).

If Open WebUI runs in Docker, use the exporter service name on a shared Docker
network or a reachable host address. `localhost` in that container refers to
Open WebUI's own container.

For OpenAI-compatible clients, use the exporter URL with `/v1`, for example
`http://localhost:8000/v1`. Request and token metrics are collected on
`/v1/chat/completions` and `/v1/completions`; streamed token counts require
`stream_options.include_usage`.

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

### Running with WSL Containers (`wslc`)

Run these commands from Windows PowerShell in the repository directory.
`wslc` builds the existing Dockerfile and runs it with WSL's container engine.
It requires WSL 2.9.3 or newer; see Microsoft's
[WSL container setup guide](https://learn.microsoft.com/en-us/windows/wsl/tutorials/wsl-containers).

Check that the commands are available:

```powershell
wsl --version
wslc version
```

If your WSL installation does not include `wslc`, update it with `wsl --update`.

Build the image, then start the exporter with Ollama running on Windows:

```powershell
wslc build --tag ollama-exporter .
wslc run --detach --name ollama-exporter --publish 8000:8000 --env OLLAMA_HOST=http://host.wslc.internal:11434 ollama-exporter
```

`host.wslc.internal` is the default name for reaching the Windows host from a
WSL container. If Ollama runs on another machine, replace that URL with its
reachable address. `localhost` inside this container refers to the exporter
container itself.

Verify both the proxy and metrics from Windows:

```powershell
curl.exe --fail http://localhost:8000/api/version
curl.exe --fail http://localhost:8000/metrics
wslc logs --tail 50 ollama-exporter
```

Configure applications to use `http://localhost:8000` when they run on the
Windows host, and use `localhost:8000` as the Prometheus scrape target there.
Published ports bind to Windows loopback (`127.0.0.1`) by default. For access
from other machines, configure the publishing address intentionally using
the [WSL container networking settings](https://learn.microsoft.com/en-us/windows/wsl/wsl-container-settings)
and allow the chosen port through Windows Firewall.

If Ollama is already running in a WSL container named `ollama`, you can connect
both containers to a shared network instead:

```powershell
wslc network create ollama-monitoring
wslc network connect --network-alias ollama ollama-monitoring ollama
wslc run --detach --name ollama-exporter --network ollama-monitoring --publish 8000:8000 --env OLLAMA_HOST=http://ollama:11434 ollama-exporter
```

Use one of the two exporter run commands above. Stop and remove the exporter
before recreating it with the same name or switching upstream configurations:

```powershell
wslc stop ollama-exporter
wslc remove ollama-exporter
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
3. **Configure your application** to use `http://localhost:8000` as its Ollama
   connection URL. For Open WebUI, update the Ollama connection in its admin
   settings. Keep the exporter's own `OLLAMA_HOST` pointed at Ollama.
4. **Prometheus** scrapes metrics from the exporter:
   - Set the scrape target to `localhost:8000`; Prometheus uses `/metrics` by default.

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

The automated suite uses a simulated upstream and does not require Ollama,
downloaded models, or a GPU. CI runs the tests on Python 3.11–3.14 and checks
that the container builds.
Direct dependencies are pinned in `requirements.txt`; Docker uses that same
file. Dependabot checks Python packages, the Docker base, and Actions monthly.
The CI workflow validates changes. A separate, manually started release
workflow reruns those checks before publishing a versioned source release;
neither workflow publishes container images. See the
[release procedure](.github/RELEASING.md). Documentation, tests, and CI changes
alone do not trigger a release.

Changes to `main` go through a pull request. The four Python checks and the
container build must pass, the branch must be current with `main`, and review
conversations must be resolved. These rules also apply to administrators;
force pushes and deletion of `main` are disabled. No second reviewer is
required for this single-maintainer repository.

CI uses a read-only `GITHUB_TOKEN`, does not retain checkout credentials, and
cannot approve pull requests. Only the release publishing job has repository
contents write access. Fork workflows from first-time contributors require
maintainer approval.

## Live Validation

Main commit `090cd9e` was validated on 2026-10-29 against live Ollama and
Grafana/VictoriaMetrics services:

- 57 automated tests and the dependency check passed.
- 16 live application checks passed, including native Windows startup, WSLC
  execution, native and OpenAI-compatible streaming, embeddings, cancellation,
  and inference deadlines.
- All 18 dashboard queries returned data, and both dashboards rendered all
  14 panels with the existing data source selected automatically.

See the [dated validation record and screenshots](docs/validation/2026-10-29.md)
for the tested versions, scope, and cleanup. This live acceptance check is
separate from the simulated-upstream tests run by CI.

## License
There is no spoon.
