import os
import argparse
import asyncio
import httpx
import json
import logging
import socket
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

import uvicorn
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Default values, overridable via environment variables or CLI arguments.
# CLI arguments take precedence over environment variables (see parse_args).
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_LISTEN_HOST = "::"  # dual-stack: binds both IPv6 and IPv4
DEFAULT_LISTEN_PORT = 8000
DEFAULT_LOG_LEVEL = "INFO"

# Fallback when the host has no usable IPv6 stack (see bind_listen_socket).
IPV4_WILDCARD = "0.0.0.0"

# Spellings of the IPv6 wildcard we bind ourselves rather than leaving to
# uvicorn; any other address is unambiguous and asyncio handles it correctly.
IPV6_WILDCARDS = frozenset({"::", "[::]", "::0"})

# Configurable Ollama host. Populated by parse_args(); the env default keeps
# backward compatibility for code paths that import this module directly.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)

logging.basicConfig()
logger = logging.getLogger(__name__)
LOG_LEVEL = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

app = FastAPI()

OLLAMA_CHAT_REQUEST_COUNT = Counter("ollama_requests_total", "Total chat requests", ["model"])

OLLAMA_TOTAL_DURATION =       Histogram("ollama_response_seconds", "Total time spent for the response", ["model"])
OLLAMA_LOAD_DURATION =        Histogram("ollama_load_duration_seconds", "Time spent loading the model", ["model"])
OLLAMA_PROMPT_EVAL_DURATION = Histogram("ollama_prompt_eval_duration_seconds", "Time spent evaluating prompt", ["model"])
OLLAMA_EVAL_DURATION =        Histogram("ollama_eval_duration_seconds", "Time spent generating the response", ["model"])

OLLAMA_PROMPT_EVAL_COUNT = Counter("ollama_tokens_processed_total", "Number of tokens in the prompt", ["model"])
OLLAMA_EVAL_COUNT =        Counter("ollama_tokens_generated_total", "Number of tokens in the response", ["model"])

OLLAMA_TOKENS_PER_SECOND = Histogram(
    "ollama_tokens_per_second",
    "Tokens generated per second",
    ["model"],
    # Use buckets with suitable ranges for tokens/s measurements
    buckets=[5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
)


# Headers that describe the upstream connection/framing and must never be
# forwarded verbatim. Ollama answers with `Transfer-Encoding: chunked`; once we
# buffer the body and hand it to Starlette, Starlette adds its own
# `Content-Length`. A response carrying both is illegal (RFC 9112 s6.1) and
# strict clients reject it outright - aiohttp (Open WebUI) fails the request
# with "Content-Length can't be present with Transfer-Encoding" while lenient
# ones like curl let it slide.
HOP_BY_HOP_HEADERS = frozenset({
    "connection",
    "content-encoding",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
})


def sanitize_response_headers(headers):
    """Strip hop-by-hop headers from an upstream response before forwarding.

    Parameters
    ----------
    headers : Mapping[str, str]
        Headers as returned by the upstream Ollama response.

    Returns
    -------
    dict of str to str
        The same headers minus every entry in :data:`HOP_BY_HOP_HEADERS`, safe
        to hand back to Starlette which recomputes the framing itself.
    """
    return {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP_HEADERS}


def extract_and_record_metrics(response_data, model):
    """Extract and record metrics from Ollama response data."""
    if not isinstance(response_data, dict):
        return

    # https://github.com/ollama/ollama/blob/main/docs/api.md#response
    total_duration = response_data.get("total_duration", 0) # total time spent in nanoseconds generating the response
    load_duration = response_data.get("load_duration", 0) # time spent in nanoseconds loading the model
    prompt_eval_duration = response_data.get("prompt_eval_duration", 0) # time spent in nanoseconds evaluating the prompt
    prompt_eval_count = response_data.get("prompt_eval_count", 0) # number of tokens in the prompt
    eval_duration = response_data.get("eval_duration", 0) # time spent in nanoseconds generating the response
    eval_count = response_data.get("eval_count", 0) # number of tokens in the response

    if total_duration > 0:
        total_duration_seconds = total_duration / 1_000_000_000
        OLLAMA_TOTAL_DURATION.labels(model=model).observe(total_duration_seconds)
        logger.debug(f"Model: {model}, Total Duration: {total_duration_seconds:.2f} seconds")
    if load_duration > 0:
        load_duration_seconds = load_duration / 1_000_000_000
        OLLAMA_LOAD_DURATION.labels(model=model).observe(load_duration_seconds)
        logger.debug(f"Model: {model}, Load Duration: {load_duration_seconds:.2f} seconds")
    if prompt_eval_duration > 0:
        prompt_eval_time_seconds = prompt_eval_duration / 1_000_000_000
        OLLAMA_PROMPT_EVAL_DURATION.labels(model=model).observe(prompt_eval_time_seconds)
        logger.debug(f"Model: {model}, Prompt Eval Duration: {prompt_eval_time_seconds:.2f} seconds")
    if prompt_eval_count > 0:
        OLLAMA_PROMPT_EVAL_COUNT.labels(model=model).inc(prompt_eval_count)
        logger.debug(f"Model: {model}, Prompt Eval Count: {prompt_eval_count}")
    if eval_duration > 0:
        eval_duration_seconds = eval_duration / 1_000_000_000
        OLLAMA_EVAL_DURATION.labels(model=model).observe(eval_duration_seconds)
        logger.debug(f"Model: {model}, Eval Duration: {eval_duration_seconds:.2f} seconds")
    if eval_count > 0:
        OLLAMA_EVAL_COUNT.labels(model=model).inc(eval_count)
        logger.debug(f"Model: {model}, Eval Count: {eval_count}")
    if eval_duration > 0 and eval_count > 0:
        tps = eval_count / eval_duration * 1_000_000_000
        OLLAMA_TOKENS_PER_SECOND.labels(model=model).observe(tps)
        logger.debug(f"Model: {model}, Tokens per Second: {tps:.2f}")

@app.get("/metrics")
def metrics():
    """Expose Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/api/chat")
@app.post("/api/generate")
async def chat_with_metrics(request: Request):
    """Handle chat and generate requests with streaming support and metrics extraction."""
    body = await request.json()
    model = body.get("model", "unknown")
    # logger.debug(f"Chat request body: {json.dumps(body, indent=4)}")
    is_streaming = body.get("stream", False)

    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    headers.pop("content-type", None)

    OLLAMA_CHAT_REQUEST_COUNT.labels(model=model).inc()

    if is_streaming:
        async def generate_stream():
            endpoint = request.url.path  # /api/chat or /api/generate
            async with httpx.AsyncClient(timeout=httpx.Timeout(900.0, read=900.0)) as client:
                async with client.stream("POST", f"{OLLAMA_HOST}{endpoint}", headers=headers, json=body, params=request.query_params) as response:

                    final_chunk_data = None

                    async for chunk in response.aiter_bytes():
                        # Forward the chunk immediately to the client
                        yield chunk

                        # Try to parse the chunk to look for metrics
                        if chunk:
                            try:
                                chunk_text = chunk.decode('utf-8')
                                lines = chunk_text.strip().split('\n')

                                for line in lines:
                                    if line.strip():
                                        try:
                                            chunk_json = json.loads(line)
                                            # Check if this is the final chunk (contains "done": true)
                                            if chunk_json.get("done", False):
                                                final_chunk_data = chunk_json
                                        except json.JSONDecodeError:
                                            continue

                            except UnicodeDecodeError:
                                pass

                    # Extract metrics from the final chunk if available
                    if final_chunk_data:
                        extract_and_record_metrics(final_chunk_data, model)

        return StreamingResponse(generate_stream(), media_type="application/json")
    else:
        endpoint = request.url.path  # /api/chat or /api/generate
        async with httpx.AsyncClient(timeout=httpx.Timeout(900.0, read=900.0)) as client:
            response = await client.post(f"{OLLAMA_HOST}{endpoint}", headers=headers, json=body, params=request.query_params)

            if response.status_code == 200:
                try:
                    response_data = response.json()
                    extract_and_record_metrics(response_data, model)
                except (json.JSONDecodeError, TypeError):
                    pass

            return Response(content=response.content, status_code=response.status_code, headers=sanitize_response_headers(response.headers))

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def simple_proxy(request: Request, path: str):
    """Simple pass-through proxy for all other endpoints."""
    logger.debug(f"Proxying {request.method} request to /{path}")
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)

    async with httpx.AsyncClient(timeout=httpx.Timeout(900.0, read=900.0)) as client:
        response = await client.request(method=request.method, url=f"{OLLAMA_HOST}/{path}", headers=headers, content=await request.body(), params=request.query_params)

    logger.debug(f"Proxy response: {response.status_code} for {request.method} /{path}")
    return Response(content=response.content, status_code=response.status_code, headers=sanitize_response_headers(response.headers))

async def verify_ollama_connection():
    """Verify connection to Ollama server at startup."""
    logger.debug(f"Verifying connection to Ollama server at {OLLAMA_HOST}")

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            response = await client.get(f"{OLLAMA_HOST}/api/version")
            if response.status_code == 200:
                version_data = response.json()
                logger.info(f"Connected to Ollama")
            else:
                logger.error(f"Failed to connect to Ollama server. Status code: {response.status_code}")
    except Exception as e:
        logger.error(f"Failed to connect to Ollama server at {OLLAMA_HOST}: {e}")
        logger.error("Please ensure Ollama is running and accessible at the configured host")

def parse_args(argv=None):
    """Parse command-line arguments.

    Environment variables provide the defaults so that container deployments
    keep working without flags, while explicit CLI arguments always win.

    Parameters
    ----------
    argv : list of str, optional
        Argument vector to parse. Defaults to ``sys.argv[1:]`` when ``None``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with ``host``, ``port``, ``ollama_host`` and
        ``log_level`` attributes.
    """
    parser = argparse.ArgumentParser(
        description="Prometheus exporter and metrics-extracting proxy for Ollama."
    )
    parser.add_argument(
        "--host",
        default=os.getenv("EXPORTER_HOST", DEFAULT_LISTEN_HOST),
        help="Address to bind the exporter to "
             "(default: %(default)s, dual-stack IPv6/IPv4).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("EXPORTER_PORT", DEFAULT_LISTEN_PORT)),
        help="Port to listen on (default: %(default)s).",
    )
    parser.add_argument(
        "--ollama-host",
        default=os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
        help="Base URL of the upstream Ollama server (default: %(default)s).",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper(),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        type=str.upper,
        help="Logging verbosity (default: %(default)s).",
    )
    return parser.parse_args(argv)


def bind_listen_socket(host, port):
    """Bind the listening socket when the requested host needs special care.

    asyncio sets ``IPV6_V6ONLY`` on every AF_INET6 socket it opens, so letting
    uvicorn bind ``::`` on its own produces an IPv6-only listener: IPv4 clients,
    including anything reaching the exporter over ``127.0.0.1``, get a
    connection refused despite the advertised dual-stack default. Binding here
    with the option cleared gives a single socket serving both families.

    Parameters
    ----------
    host : str
        Address the exporter was asked to bind. Only the IPv6 wildcard is
        handled here.
    port : int
        TCP port to listen on.

    Returns
    -------
    socket.socket or None
        A bound socket to hand over to uvicorn, or ``None`` when ``host`` is
        not the IPv6 wildcard and uvicorn can bind it itself.
    """
    if host not in IPV6_WILDCARDS:
        return None

    sock = None
    try:
        sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Explicit 0 also overrides a system-wide net.ipv6.bindv6only=1.
        sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        sock.bind(("::", port))
    except OSError as exc:
        # No usable IPv6 stack (kernel with ipv6.disable=1, IPv4-only sandbox):
        # serve IPv4 rather than refusing to start at all.
        if sock is not None:
            sock.close()
        logger.warning(
            f"Dual-stack bind on [::]:{port} failed ({exc}), "
            f"falling back to {IPV4_WILDCARD}:{port}"
        )
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((IPV4_WILDCARD, port))

    # uvicorn passes the socket to loop.create_server(), which calls listen().
    return sock


async def main():
    """Configure runtime from CLI/env, then start the exporter server."""
    global OLLAMA_HOST
    args = parse_args()

    # CLI/env arguments override the module-level defaults set at import time.
    OLLAMA_HOST = args.ollama_host
    logger.setLevel(getattr(logging, args.log_level, logging.INFO))

    await verify_ollama_connection()
    config = uvicorn.Config(
        app, host=args.host, port=args.port, log_level=args.log_level.lower()
    )
    server = uvicorn.Server(config)

    sock = bind_listen_socket(args.host, args.port)
    if sock is None:
        await server.serve()
    else:
        # serve(sockets=...) skips uvicorn's own bind, so our socket options
        # survive.
        families = "IPv6+IPv4" if sock.family == socket.AF_INET6 else "IPv4"
        logger.info(f"Listening on {sock.getsockname()[0]}:{args.port} ({families})")
        await server.serve(sockets=[sock])

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # SIGINT (Ctrl-C, or `docker stop` via STOPSIGNAL) is a normal shutdown:
        # uvicorn already closed the server, so exit quietly instead of letting
        # the traceback surface and the process report a failure status.
        logger.info("Shutting down")
