import os
import argparse
import asyncio
import httpx
import json
import logging
import socket
from urllib.parse import urlsplit

import anyio
from fastapi import FastAPI, Request, Response

import uvicorn
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Default values, overridable via environment variables or CLI arguments.
# CLI arguments take precedence over environment variables (see parse_args).
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_LISTEN_HOST = "::"  # dual-stack: binds both IPv6 and IPv4
DEFAULT_LISTEN_PORT = 8000
DEFAULT_LOG_LEVEL = "INFO"
# Total deadline, in seconds, for one inference request (0 disables it).
DEFAULT_REQUEST_TIMEOUT = 1800.0

# Fallback when the host has no usable IPv6 stack (see bind_listen_socket).
IPV4_WILDCARD = "0.0.0.0"

# Spellings of the IPv6 wildcard we bind ourselves rather than leaving to
# uvicorn; any other address is unambiguous and asyncio handles it correctly.
IPV6_WILDCARDS = frozenset({"::", "[::]", "::0"})

# Configurable Ollama host. Populated by parse_args(); the env default keeps
# backward compatibility for code paths that import this module directly.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)

# Total deadline applied to inference requests, populated the same way.
REQUEST_TIMEOUT = float(os.getenv("EXPORTER_REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT))

# Native endpoints that run a model. Any POST under /v1/ (Ollama's OpenAI
# compatible API) counts as well, see is_inference().
INFERENCE_PATHS = frozenset({"/api/chat", "/api/generate", "/api/embed", "/api/embeddings"})

# Endpoints whose final record carries Ollama's native stats (NDJSON).
NATIVE_METRICS_PATHS = frozenset({"/api/chat", "/api/generate"})
# OpenAI-compatible endpoints whose responses carry a `usage` object.
OPENAI_METRICS_PATHS = frozenset({"/v1/chat/completions", "/v1/completions"})
# Endpoints counted in ollama_requests_total.
COUNTED_PATHS = NATIVE_METRICS_PATHS | OPENAI_METRICS_PATHS

logging.basicConfig()
logger = logging.getLogger(__name__)
LOG_LEVEL = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

app = FastAPI()

OLLAMA_CHAT_REQUEST_COUNT = Counter("ollama_requests_total", "Total chat requests", ["model"])
OLLAMA_INFLIGHT = Gauge("ollama_inflight_requests", "Inference requests currently proxied to Ollama", ["model"])

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

# Headers uvicorn always adds itself; forwarding Ollama's too would send them
# twice (`date: X, X`, `server: uvicorn, uvicorn`).
SERVER_OWNED_HEADERS = frozenset({"date", "server"})


def sanitize_response_headers(headers):
    """Strip hop-by-hop headers from an upstream response before forwarding.

    ``Content-Length`` is kept when the upstream body has no
    ``Content-Encoding``: httpx then hands the bytes over untouched, so the
    length still matches and clients keep download sizes, HEAD answers and a
    way to detect a truncated body. Uvicorn frames the response with it
    instead of switching to chunked encoding.

    Parameters
    ----------
    headers : httpx.Headers
        Headers as returned by the upstream Ollama response.

    Returns
    -------
    list of tuple of (bytes, bytes)
        Raw ASGI header pairs, lower-cased names, repeated headers preserved,
        minus every entry in :data:`HOP_BY_HOP_HEADERS` and
        :data:`SERVER_OWNED_HEADERS`.
    """
    dropped = HOP_BY_HOP_HEADERS | SERVER_OWNED_HEADERS
    if "content-encoding" not in headers:
        dropped = dropped - {"content-length"}
    # .raw keeps the bytes as received, so no header can fail to re-encode.
    return [(k.lower(), v) for k, v in headers.raw if k.lower().decode("latin-1") not in dropped]


def extract_and_record_metrics(response_data, model):
    """Extract and record metrics from Ollama response data.

    Parameters
    ----------
    response_data : dict
        Final native response record (the one with ``"done": true``), holding
        Ollama's durations in nanoseconds and its token counts.
    model : str
        Model label, taken from the request body.
    """
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
    """Expose Prometheus metrics.

    Returns
    -------
    fastapi.Response
        Every registered metric in the Prometheus text format.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

def record_openai_usage(response_data, model):
    """Record token counts from an OpenAI-compatible (``/v1``) response.

    Ollama's OpenAI layer reports no durations, only a ``usage`` object, so
    only the token counters can be fed from it.

    Parameters
    ----------
    response_data : dict
        Decoded completion object, or the last streamed chunk carrying
        ``usage`` (sent when the client asks for ``stream_options.include_usage``).
    model : str
        Model label, taken from the request body.
    """
    usage = response_data.get("usage") if isinstance(response_data, dict) else None
    if not isinstance(usage, dict):
        return
    prompt_tokens = usage.get("prompt_tokens") or 0
    completion_tokens = usage.get("completion_tokens") or 0
    if prompt_tokens > 0:
        OLLAMA_PROMPT_EVAL_COUNT.labels(model=model).inc(prompt_tokens)
    if completion_tokens > 0:
        OLLAMA_EVAL_COUNT.labels(model=model).inc(completion_tokens)
    logger.debug(f"Model: {model}, OpenAI usage: {prompt_tokens} prompt / {completion_tokens} completion tokens")


class MetricsTap:
    """Watch a relayed response body for the record that carries the stats.

    The body is fed chunk by chunk as it flows to the client. Lines are
    reassembled across chunk boundaries, so a JSON record split by the network
    is still parsed; only the last matching record is kept and recorded once
    the response completes.

    Parameters
    ----------
    path : str
        Request path, which decides the body format: native NDJSON for
        :data:`NATIVE_METRICS_PATHS`, OpenAI JSON or SSE for
        :data:`OPENAI_METRICS_PATHS`, nothing at all for anything else.
    model : str
        Model label to record the metrics under.
    """

    def __init__(self, path, model):
        """Initialise an empty tap for one response (see class docstring)."""
        if path in NATIVE_METRICS_PATHS:
            self.kind = "native"
        elif path in OPENAI_METRICS_PATHS:
            self.kind = "openai"
        else:
            self.kind = None
        self.model = model
        self._buffer = b""
        self._final = None

    def feed(self, chunk):
        """Consume one body chunk.

        Parameters
        ----------
        chunk : bytes
            Decoded body bytes, exactly as forwarded to the client.
        """
        if self.kind is None:
            return
        self._buffer += chunk
        *lines, self._buffer = self._buffer.split(b"\n")
        for line in lines:
            self._parse(line)

    def finish(self):
        """Parse what is left of the body and record the metrics found."""
        if self.kind is None:
            return
        # A non-streaming body is usually a single line with no trailing newline.
        self._parse(self._buffer)
        self._buffer = b""
        if self._final is None:
            return
        if self.kind == "native":
            extract_and_record_metrics(self._final, self.model)
        else:
            record_openai_usage(self._final, self.model)

    def _parse(self, line):
        """Remember ``line`` if it is the record holding the final stats.

        Parameters
        ----------
        line : bytes
            One body line, an NDJSON record or an SSE ``data:`` field.
        """
        line = line.strip()
        if self.kind == "openai" and line.startswith(b"data:"):
            line = line[len(b"data:"):].strip()
        if not line or line == b"[DONE]":
            return
        try:
            data = json.loads(line)
        except ValueError:  # JSONDecodeError and UnicodeDecodeError alike
            return
        if not isinstance(data, dict):
            return
        if self.kind == "native" and data.get("done"):
            self._final = data
        elif self.kind == "openai" and isinstance(data.get("usage"), dict):
            self._final = data


def is_inference(method, path):
    """Tell whether a request makes Ollama run a model.

    Parameters
    ----------
    method : str
        HTTP method of the request.
    path : str
        Request path, with its leading slash.

    Returns
    -------
    bool
        ``True`` for POSTs to the native inference endpoints and to any
        OpenAI-compatible ``/v1`` endpoint. Those get the in-flight gauge and
        the total deadline; model pulls, pushes and listings do not.
    """
    return method == "POST" and (path in INFERENCE_PATHS or path.startswith("/v1/"))


def model_from_body(body):
    """Pull the ``model`` field out of a raw JSON request body.

    Parameters
    ----------
    body : bytes
        Request body as received from the client.

    Returns
    -------
    str
        The model name, or ``"unknown"`` when the body is not a JSON object
        naming one (Ollama will reject such a request itself).
    """
    try:
        data = json.loads(body) if body else None
    except ValueError:
        return "unknown"
    model = data.get("model") if isinstance(data, dict) else None
    return model if isinstance(model, str) and model else "unknown"


class UpstreamProxyResponse(Response):
    """ASGI response that relays one request to Ollama as a live stream.

    Every proxied request goes through this class, streaming or not. It owns
    the whole upstream exchange so that its lifetime is bound to the client's:

    - the body is forwarded chunk by chunk as Ollama produces it, nothing is
      buffered to the end;
    - a task listens for ``http.disconnect`` the whole time, including while
      Ollama has not answered yet (a non-streaming generation sends nothing
      until it is over, so a failed write would never reveal the hang-up);
      on disconnect the upstream request is cancelled and its connection
      closed, which is what makes Ollama stop generating;
    - an optional total deadline bounds the exchange end to end, unlike
      httpx's read timeout which a steadily streaming generation never hits.

    Parameters
    ----------
    method : str
        HTTP method to use upstream.
    url : str
        Full upstream URL.
    headers : dict of str to str
        Request headers to forward.
    params : starlette.datastructures.QueryParams
        Query string to forward.
    body : bytes
        Request body to forward.
    model : str
        Model label for metrics.
    deadline : float or None
        Seconds after which the exchange is aborted, ``None`` for no limit.
    track_inflight : bool
        Whether this request counts in ``ollama_inflight_requests``.
    """

    def __init__(self, method, url, headers, params, body, model, deadline, track_inflight):
        """Store the upstream request to perform (see class docstring)."""
        super().__init__()
        self.method = method
        self.url = url
        self.forward_headers = headers
        self.params = params
        self.forward_body = body
        self.model = model
        self.deadline = deadline
        self.track_inflight = track_inflight
        self.tap = MetricsTap(urlsplit(url).path, model)
        # Set right before the last body message: from then on, uvicorn's
        # receive() answers http.disconnect for a completed response, which
        # is not the client hanging up.
        self.finished = False

    async def __call__(self, scope, receive, send):
        """Run the upstream exchange until it ends or the client leaves.

        Parameters
        ----------
        scope : dict
            ASGI connection scope.
        receive : callable
            ASGI receive channel; the request body was already consumed, so
            the only message left to come is ``http.disconnect``.
        send : callable
            ASGI send channel.
        """
        if self.track_inflight:
            OLLAMA_INFLIGHT.labels(model=self.model).inc()
        try:
            async with anyio.create_task_group() as task_group:

                async def watch_disconnect():
                    """Cancel the exchange as soon as the client hangs up."""
                    while True:
                        message = await receive()
                        if message["type"] == "http.disconnect":
                            if self.finished:
                                return
                            logger.info(
                                f"Client disconnected, cancelling {self.method} "
                                f"{urlsplit(self.url).path} (model {self.model})"
                            )
                            task_group.cancel_scope.cancel()
                            return

                task_group.start_soon(watch_disconnect)
                await self._relay(send)
                # The exchange is over: stop listening for a disconnect.
                task_group.cancel_scope.cancel()
        finally:
            if self.track_inflight:
                OLLAMA_INFLIGHT.labels(model=self.model).dec()

    async def _relay(self, send):
        """Perform the upstream request and forward its response.

        Parameters
        ----------
        send : callable
            ASGI send channel to the client.
        """
        path = urlsplit(self.url).path
        client = httpx.AsyncClient(timeout=httpx.Timeout(900.0, read=900.0))
        upstream = None
        started = False
        try:
            # fail_after(None) imposes no limit.
            with anyio.fail_after(self.deadline):
                request = client.build_request(
                    self.method, self.url, headers=self.forward_headers,
                    params=self.params, content=self.forward_body,
                )
                upstream = await client.send(request, stream=True)
                await send({
                    "type": "http.response.start",
                    "status": upstream.status_code,
                    "headers": sanitize_response_headers(upstream.headers),
                })
                started = True
                # aiter_bytes() undoes any content-encoding, which is why that
                # header is stripped by sanitize_response_headers().
                async for chunk in upstream.aiter_bytes():
                    self.tap.feed(chunk)
                    await send({"type": "http.response.body", "body": chunk, "more_body": True})
            # Record before the final send: once the response is complete the
            # watcher may wake up and cancel this task.
            if upstream.status_code == 200:
                self.tap.finish()
            self.finished = True
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            logger.debug(f"Proxy response: {upstream.status_code} for {self.method} {path}")
        except TimeoutError:
            logger.warning(
                f"{self.method} {path} (model {self.model}) exceeded the "
                f"{self.deadline:g}s request deadline, aborting it upstream"
            )
            await self._fail(send, started, 504, "request deadline exceeded")
        except httpx.HTTPError as exc:
            logger.error(f"Upstream error on {self.method} {path}: {exc!r}")
            await self._fail(send, started, 502, f"upstream error: {exc}")
        except Exception:
            # Anything else would escape the task group as an ExceptionGroup
            # and reach the client as a bare 500 or a stream ended cleanly.
            logger.exception(f"Proxy failure on {self.method} {path}")
            await self._fail(send, started, 502, "proxy error")
        finally:
            # Closing the connection is what tells Ollama to stop generating.
            # Shielded, because on a client disconnect we run under a
            # cancelled scope where any unshielded await would bail out first.
            with anyio.CancelScope(shield=True):
                if upstream is not None:
                    await upstream.aclose()
                await client.aclose()

    @staticmethod
    async def _fail(send, started, status, message):
        """Report an aborted exchange to the client.

        Parameters
        ----------
        send : callable
            ASGI send channel to the client.
        started : bool
            Whether the response status line was already sent. If so, the
            status cannot change any more, so the body is deliberately left
            unterminated: returning without the final message makes uvicorn
            drop the connection, and the client sees a truncated response
            instead of a 200 that looks complete.
        status : int
            HTTP status to answer with when nothing was sent yet.
        message : str
            Error text, returned in Ollama's ``{"error": ...}`` shape.
        """
        if not started:
            payload = json.dumps({"error": message}).encode()
            await send({
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(payload)).encode()),
                ],
            })
            await send({"type": "http.response.body", "body": payload})


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def proxy(request: Request, path: str):
    """Relay any request to Ollama, recording metrics on the way.

    Parameters
    ----------
    request : fastapi.Request
        Incoming client request.
    path : str
        Request path without its leading slash.

    Returns
    -------
    UpstreamProxyResponse
        Response that performs the upstream exchange when served.
    """
    endpoint = f"/{path}"
    body = await request.body()
    inference = is_inference(request.method, endpoint)
    model = model_from_body(body) if request.method == "POST" else "unknown"
    logger.debug(f"Proxying {request.method} request to {endpoint}")

    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)

    if request.method == "POST" and endpoint in COUNTED_PATHS:
        OLLAMA_CHAT_REQUEST_COUNT.labels(model=model).inc()

    return UpstreamProxyResponse(
        method=request.method,
        url=f"{OLLAMA_HOST}{endpoint}",
        headers=headers,
        params=request.query_params,
        body=body,
        model=model,
        deadline=REQUEST_TIMEOUT if inference and REQUEST_TIMEOUT > 0 else None,
        track_inflight=inference,
    )

async def verify_ollama_connection():
    """Verify connection to Ollama server at startup.

    Only logs the outcome: the exporter starts either way, so a late Ollama
    does not keep it in a crash loop.
    """
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
        Parsed arguments with ``host``, ``port``, ``ollama_host``,
        ``request_timeout`` and ``log_level`` attributes.
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
        "--request-timeout",
        type=float,
        default=float(os.getenv("EXPORTER_REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT)),
        help="Total deadline in seconds for one inference request, after "
             "which it is aborted upstream; 0 disables it (default: %(default)s).",
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
    """Configure runtime from CLI/env, then start the exporter server.

    Sets the module-level ``OLLAMA_HOST`` and ``REQUEST_TIMEOUT`` from the
    parsed arguments before serving, since request handlers read them there.
    """
    global OLLAMA_HOST, REQUEST_TIMEOUT
    args = parse_args()

    # CLI/env arguments override the module-level defaults set at import time.
    OLLAMA_HOST = args.ollama_host
    REQUEST_TIMEOUT = args.request_timeout
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
