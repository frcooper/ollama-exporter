"""Regression tests for the exporter's listening socket.

The default bind address is the IPv6 wildcard, and asyncio sets ``IPV6_V6ONLY``
on the sockets it creates, which used to leave IPv4 clients with a connection
refused. These tests pin the dual-stack behaviour and its IPv4 fallback.
"""

import socket

import pytest

import ollama_exporter


def _ipv6_loopback_available():
    """Report whether the machine can actually serve over IPv6.

    Returns
    -------
    bool
        ``True`` when an IPv6 socket binds on the loopback, ``False`` on hosts
        with IPv6 disabled, where the dual-stack expectations cannot hold.
    """
    if not socket.has_ipv6:
        return False
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
            probe.bind(("::1", 0))
    except OSError:
        return False
    return True


needs_ipv6 = pytest.mark.skipif(
    not _ipv6_loopback_available(), reason="host has no usable IPv6 stack"
)


@needs_ipv6
def test_ipv6_wildcard_serves_both_families():
    """A wildcard bind must accept IPv4 and IPv6 clients on the same socket."""
    sock = ollama_exporter.bind_listen_socket("::", 0)
    try:
        assert sock.family == socket.AF_INET6
        # uvicorn normally does this; the test drives the socket itself.
        sock.listen(8)
        port = sock.getsockname()[1]

        for address in ("127.0.0.1", "::1"):
            with socket.create_connection((address, port), timeout=2):
                pass
    finally:
        sock.close()


@pytest.mark.parametrize("host", ["::", "[::]", "::0"])
@needs_ipv6
def test_wildcard_spellings_are_all_bound_here(host):
    """Every accepted spelling of the IPv6 wildcard takes the dual-stack path."""
    sock = ollama_exporter.bind_listen_socket(host, 0)
    try:
        assert sock is not None
        assert sock.getsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY) == 0
    finally:
        sock.close()


@pytest.mark.parametrize("host", ["127.0.0.1", "0.0.0.0", "::1", "192.0.2.10"])
def test_specific_hosts_are_left_to_uvicorn(host):
    """Unambiguous addresses need no special handling, so no socket is built."""
    assert ollama_exporter.bind_listen_socket(host, 0) is None


def test_falls_back_to_ipv4_without_ipv6(monkeypatch):
    """A host without IPv6 still gets a listener, on the IPv4 wildcard."""
    real_socket = socket.socket

    def fake_socket(family=socket.AF_INET, *args, **kwargs):
        """Stand in for socket.socket, rejecting AF_INET6 like a v6-less host."""
        if family == socket.AF_INET6:
            raise OSError(97, "Address family not supported by protocol")
        return real_socket(family, *args, **kwargs)

    monkeypatch.setattr(socket, "socket", fake_socket)

    sock = ollama_exporter.bind_listen_socket("::", 0)
    try:
        assert sock.family == socket.AF_INET
        assert sock.getsockname()[0] == ollama_exporter.IPV4_WILDCARD
    finally:
        sock.close()
