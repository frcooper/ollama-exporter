"""Integration tests for the proxy's lifecycle against a fake Ollama.

The 2026-10-04 incident: a client gave up on ``/v1/chat/completions`` but the
exporter kept its upstream request open, so Ollama, which only stops
generating when its HTTP request context ends, kept the GPUs busy for two
hours. These tests run the real exporter under uvicorn on a loopback port,
in front of a raw TCP server that plays Ollama and records the moment its
connection is closed, which is exactly what Ollama reacts to.
"""

import asyncio
import json
import socket

import pytest
import uvicorn
from prometheus_client import REGISTRY

import ollama_exporter


# How long a disconnect may take to propagate upstream before we call it lost.
PROPAGATION_TIMEOUT = 3.0


class FakeOllama:
    """Minimal HTTP/1.1 server standing in for Ollama.

    Parameters
    ----------
    mode : {"stream", "hang", "reply", "die"}
        ``"stream"`` answers 200 and then sends a chunk every 50 ms forever,
        like a model that never stops generating. ``"hang"`` never answers,
        like a non-streaming request still being generated. ``"reply"`` sends
        ``reply_body`` once with a ``Content-Length`` and keeps the connection.
        ``"die"`` starts a chunked 200, sends one chunk and drops the
        connection, like Ollama crashing mid-generation.
    reply_body : bytes, optional
        Body sent in ``"reply"`` mode.
    chunk : bytes, optional
        Payload repeated in ``"stream"`` mode.
    """

    def __init__(self, mode, reply_body=b"", chunk=b'{"done":false}\n'):
        self.mode = mode
        self.reply_body = reply_body
        self.chunk = chunk
        self.requests = []
        self.connected = asyncio.Event()
        self.closed = asyncio.Event()
        self._server = None
        self._writers = []

    async def start(self):
        """Listen on an ephemeral loopback port.

        Returns
        -------
        str
            Base URL to use as ``OLLAMA_HOST``.
        """
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{port}"

    async def stop(self):
        """Drop every open connection and stop listening."""
        for writer in self._writers:
            writer.close()
        self._server.close()

    async def _handle(self, reader, writer):
        """Serve one connection and flag when the proxy closes it.

        Parameters
        ----------
        reader : asyncio.StreamReader
            Incoming side of the proxy's connection.
        writer : asyncio.StreamWriter
            Outgoing side of the proxy's connection.
        """
        self._writers.append(writer)
        head = await reader.readuntil(b"\r\n\r\n")
        request_line, *header_lines = head.decode("latin-1").split("\r\n")
        headers = {}
        for line in header_lines:
            if ":" in line:
                name, value = line.split(":", 1)
                headers[name.strip().lower()] = value.strip()
        body = await reader.readexactly(int(headers.get("content-length", 0)))
        self.requests.append((request_line, headers, body))
        self.connected.set()

        pump = None
        if self.mode == "stream":
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\n"
                b"Transfer-Encoding: chunked\r\n\r\n"
            )
            pump = asyncio.create_task(self._pump(writer))
        elif self.mode == "reply":
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                b"Date: upstream\r\nContent-Length: %d\r\n\r\n%s"
                % (len(self.reply_body), self.reply_body)
            )
        elif self.mode == "die":
            writer.write(
                b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
                b"%x\r\n%s\r\n" % (len(self.chunk), self.chunk)
            )
            await writer.drain()
            writer.close()
            self.closed.set()
            return

        # The proxy sends nothing after its request, so the next read only
        # returns once it closes the connection: that is Ollama's cancel signal.
        try:
            await reader.read(1)
        except ConnectionError:
            pass
        self.closed.set()
        if pump is not None:
            pump.cancel()

    async def _pump(self, writer):
        """Write one chunk every 50 ms until the connection breaks.

        Parameters
        ----------
        writer : asyncio.StreamWriter
            Connection to stream the chunks on.
        """
        try:
            while True:
                writer.write(b"%x\r\n%s\r\n" % (len(self.chunk), self.chunk))
                await writer.drain()
                await asyncio.sleep(0.05)
        except (ConnectionError, RuntimeError):
            pass


async def start_exporter():
    """Serve the exporter app on an ephemeral loopback port.

    Returns
    -------
    tuple of (uvicorn.Server, asyncio.Task, int)
        The server, the task running it and the port it listens on.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    config = uvicorn.Config(
        ollama_exporter.app, log_level="warning", timeout_graceful_shutdown=1
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:
        await asyncio.sleep(0.01)
    return server, task, port


async def stop_exporter(server, task):
    """Shut the exporter down without waiting forever on stuck requests.

    Parameters
    ----------
    server : uvicorn.Server
        Server returned by :func:`start_exporter`.
    task : asyncio.Task
        Task running that server.
    """
    server.should_exit = True
    try:
        await asyncio.wait_for(task, 5)
    except asyncio.TimeoutError:
        task.cancel()


def raw_post(path, payload):
    """Build a raw HTTP/1.1 POST request.

    Parameters
    ----------
    path : str
        Request target, e.g. ``/v1/chat/completions``.
    payload : dict
        JSON body.

    Returns
    -------
    bytes
        The request, ready to write on a socket.
    """
    body = json.dumps(payload).encode()
    return (
        f"POST {path} HTTP/1.1\r\nHost: test\r\nContent-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n\r\n"
    ).encode() + body


def run_scenario(monkeypatch, upstream, scenario, request_timeout=None):
    """Run ``scenario`` with the exporter wired in front of ``upstream``.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to point the exporter at the fake upstream.
    upstream : callable
        Zero-argument factory returning a :class:`FakeOllama`; it is built
        inside the event loop because asyncio primitives bind to it.
    scenario : callable
        ``async (fake, port) -> None`` doing the actual checks.
    request_timeout : float, optional
        Total deadline to configure on the exporter (``None`` keeps the
        default).
    """

    async def runner():
        fake = upstream()
        monkeypatch.setattr(ollama_exporter, "OLLAMA_HOST", await fake.start())
        if request_timeout is not None:
            monkeypatch.setattr(ollama_exporter, "REQUEST_TIMEOUT", request_timeout)
        server, task, port = await start_exporter()
        try:
            await asyncio.wait_for(scenario(fake, port), 15)
        finally:
            await fake.stop()
            await stop_exporter(server, task)

    asyncio.run(runner())


@pytest.mark.parametrize(
    "path, payload, mode",
    [
        ("/v1/chat/completions", {"model": "m", "stream": True}, "stream"),
        ("/v1/chat/completions", {"model": "m", "stream": False}, "hang"),
        ("/api/chat", {"model": "m", "stream": True}, "stream"),
        ("/api/chat", {"model": "m", "stream": False}, "hang"),
        ("/api/generate", {"model": "m"}, "hang"),
    ],
)
def test_client_disconnect_closes_upstream(monkeypatch, path, payload, mode):
    """A client hanging up mid-generation must close the request to Ollama."""

    async def scenario(fake, port):
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(raw_post(path, payload))
        await writer.drain()
        await asyncio.wait_for(fake.connected.wait(), PROPAGATION_TIMEOUT)
        await asyncio.sleep(0.3)  # let generation run a little
        writer.close()
        await asyncio.wait_for(fake.closed.wait(), PROPAGATION_TIMEOUT)

    run_scenario(monkeypatch, lambda: FakeOllama(mode), scenario)


def test_streamed_bytes_reach_client_before_upstream_finishes(monkeypatch):
    """``/v1`` responses are relayed as they arrive, not buffered to the end."""

    async def scenario(fake, port):
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(raw_post("/v1/chat/completions", {"model": "m", "stream": True}))
        await writer.drain()
        data = b""
        while b"data: x" not in data:
            data += await asyncio.wait_for(reader.read(4096), PROPAGATION_TIMEOUT)
        assert data.startswith(b"HTTP/1.1 200")
        writer.close()

    run_scenario(monkeypatch, lambda: FakeOllama("stream", chunk=b"data: x\n\n"), scenario)


def test_deadline_ends_a_stream_that_never_stops(monkeypatch):
    """The total deadline cuts a generation that keeps sending bytes."""

    async def scenario(fake, port):
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(raw_post("/api/chat", {"model": "m", "stream": True}))
        await writer.drain()
        # The upstream never ends on its own, so EOF here means the deadline
        # hit; the missing final chunk tells the client it was cut short.
        data = await asyncio.wait_for(reader.read(), PROPAGATION_TIMEOUT)
        assert data.startswith(b"HTTP/1.1 200")
        assert not data.endswith(b"\r\n0\r\n\r\n")
        await asyncio.wait_for(fake.closed.wait(), PROPAGATION_TIMEOUT)
        writer.close()

    run_scenario(monkeypatch, lambda: FakeOllama("stream"), scenario, request_timeout=0.5)


def test_deadline_before_headers_returns_504(monkeypatch):
    """A request still waiting for Ollama at the deadline gets a 504."""

    async def scenario(fake, port):
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(raw_post("/v1/chat/completions", {"model": "m"}))
        await writer.drain()
        data = await asyncio.wait_for(reader.read(4096), PROPAGATION_TIMEOUT)
        assert data.startswith(b"HTTP/1.1 504")
        await asyncio.wait_for(fake.closed.wait(), PROPAGATION_TIMEOUT)
        writer.close()

    run_scenario(monkeypatch, lambda: FakeOllama("hang"), scenario, request_timeout=0.5)


def test_deadline_does_not_apply_to_model_pulls(monkeypatch):
    """Long downloads like ``/api/pull`` are not inference and keep running."""

    async def scenario(fake, port):
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(raw_post("/api/pull", {"model": "m"}))
        await writer.drain()
        await asyncio.sleep(1.0)  # twice the deadline
        assert not fake.closed.is_set()
        writer.close()

    run_scenario(monkeypatch, lambda: FakeOllama("stream"), scenario, request_timeout=0.5)


def inflight(model):
    """Read the current ``ollama_inflight_requests`` value for ``model``.

    Parameters
    ----------
    model : str
        Model label to read.

    Returns
    -------
    float
        Gauge value, ``0.0`` when the series does not exist yet.
    """
    value = REGISTRY.get_sample_value("ollama_inflight_requests", {"model": model})
    return value or 0.0


def test_inflight_gauge_tracks_open_requests(monkeypatch):
    """The gauge rises while a ``/v1`` request runs and drops on disconnect."""

    async def scenario(fake, port):
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(raw_post("/v1/chat/completions", {"model": "gauge-m", "stream": True}))
        await writer.drain()
        await asyncio.wait_for(fake.connected.wait(), PROPAGATION_TIMEOUT)
        assert inflight("gauge-m") == 1
        writer.close()
        await asyncio.wait_for(fake.closed.wait(), PROPAGATION_TIMEOUT)
        for _ in range(100):
            if inflight("gauge-m") == 0:
                break
            await asyncio.sleep(0.02)
        assert inflight("gauge-m") == 0

    run_scenario(monkeypatch, lambda: FakeOllama("stream"), scenario)


def test_openai_usage_is_counted(monkeypatch):
    """``/v1/chat/completions`` feeds the request and token counters."""
    body = json.dumps(
        {"model": "oa-m", "usage": {"prompt_tokens": 7, "completion_tokens": 11}}
    ).encode()

    def sample(name):
        return REGISTRY.get_sample_value(name, {"model": "oa-m"}) or 0.0

    before = {n: sample(n) for n in (
        "ollama_requests_total", "ollama_tokens_processed_total", "ollama_tokens_generated_total")}

    async def scenario(fake, port):
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(raw_post("/v1/chat/completions", {"model": "oa-m"}))
        await writer.drain()
        data = b""
        while body not in data:
            data += await asyncio.wait_for(reader.read(4096), PROPAGATION_TIMEOUT)
        writer.close()
        for _ in range(100):
            if sample("ollama_tokens_generated_total") > before["ollama_tokens_generated_total"]:
                break
            await asyncio.sleep(0.02)

    run_scenario(monkeypatch, lambda: FakeOllama("reply", reply_body=body), scenario)
    assert sample("ollama_requests_total") - before["ollama_requests_total"] == 1
    assert sample("ollama_tokens_processed_total") - before["ollama_tokens_processed_total"] == 7
    assert sample("ollama_tokens_generated_total") - before["ollama_tokens_generated_total"] == 11


def test_native_streaming_metrics_survive_split_lines(monkeypatch):
    """The final ``done`` record is parsed even when split across chunks."""
    final = json.dumps({"done": True, "eval_count": 5, "eval_duration": 10**9}).encode()
    # Two chunks cutting the JSON line in half, then the stream ends.
    reply = b"%x\r\n%s\r\n%x\r\n%s\r\n0\r\n\r\n" % (8, final[:8], len(final) - 8 + 1, final[8:] + b"\n")

    def sample():
        return REGISTRY.get_sample_value("ollama_tokens_generated_total", {"model": "split-m"}) or 0.0

    before = sample()

    class ChunkedReply(FakeOllama):
        """Fake Ollama sending one pre-chunked streaming reply."""

        async def _handle(self, reader, writer):
            """Answer with ``reply`` then wait for the proxy to hang up.

            Parameters
            ----------
            reader : asyncio.StreamReader
                Incoming side of the proxy's connection.
            writer : asyncio.StreamWriter
                Outgoing side of the proxy's connection.
            """
            self._writers.append(writer)
            await reader.readuntil(b"\r\n\r\n")
            writer.write(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")
            writer.write(reply[:20])
            await writer.drain()
            await asyncio.sleep(0.1)
            writer.write(reply[20:])
            await writer.drain()
            await reader.read(1)

    async def scenario(fake, port):
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(raw_post("/api/chat", {"model": "split-m", "stream": True}))
        await writer.drain()
        data = b""
        while not data.endswith(b"0\r\n\r\n"):
            data += await asyncio.wait_for(reader.read(4096), PROPAGATION_TIMEOUT)
        writer.close()

    run_scenario(monkeypatch, lambda: ChunkedReply("reply"), scenario)
    assert sample() - before == 5


def test_completed_request_is_not_logged_as_disconnect(monkeypatch, caplog):
    """A response that ran to completion is not reported as a client hang-up."""
    body = b'{"done":true}'

    async def scenario(fake, port):
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(raw_post("/api/chat", {"model": "m"}))
        await writer.drain()
        data = b""
        while body not in data:
            data += await asyncio.wait_for(reader.read(4096), PROPAGATION_TIMEOUT)
        await asyncio.sleep(0.2)  # let the proxy notice the completed response
        writer.close()

    caplog.set_level("INFO", logger="ollama_exporter")
    run_scenario(monkeypatch, lambda: FakeOllama("reply", reply_body=body), scenario)
    assert "Client disconnected" not in caplog.text


def test_upstream_failure_mid_stream_is_not_a_clean_end(monkeypatch):
    """Ollama dying mid-stream must not reach the client as a complete 200."""

    async def scenario(fake, port):
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(raw_post("/api/chat", {"model": "m", "stream": True}))
        await writer.drain()
        data = await asyncio.wait_for(reader.read(), PROPAGATION_TIMEOUT)
        assert data.startswith(b"HTTP/1.1 200")
        assert b'{"done":false}' in data
        assert not data.endswith(b"\r\n0\r\n\r\n")
        writer.close()

    run_scenario(monkeypatch, lambda: FakeOllama("die"), scenario)


def test_content_length_forwarded_and_server_headers_not_doubled(monkeypatch):
    """An identity-encoded body keeps its length; Date is not sent twice."""
    body = b'{"models":[]}'

    async def scenario(fake, port):
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"GET /api/tags HTTP/1.1\r\nHost: test\r\n\r\n")
        await writer.drain()
        data = b""
        while body not in data:
            data += await asyncio.wait_for(reader.read(4096), PROPAGATION_TIMEOUT)
        head = data.split(b"\r\n\r\n", 1)[0].lower()
        assert b"content-length: %d" % len(body) in head
        assert b"transfer-encoding" not in head
        assert b"date: upstream" not in head
        assert head.count(b"\r\ndate:") == 1
        writer.close()

    run_scenario(monkeypatch, lambda: FakeOllama("reply", reply_body=body), scenario)
