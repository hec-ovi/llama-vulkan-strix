"""The websearch MCP HTTP wrapper is valid and flips the bundled stdio server
to network-reachable HTTP transport with sane, env-overridable defaults.

This is source-level (CI-safe): it does not import the websearch package, which
is installed only inside the optional sidecar container at runtime.
"""
import ast
from pathlib import Path

WRAPPER = Path(__file__).resolve().parent.parent / "tools" / "websearch_http.py"
SRC = WRAPPER.read_text()


def test_wrapper_is_valid_python():
    ast.parse(SRC)  # raises SyntaxError on a broken wrapper


def test_imports_the_bundled_fastmcp_server_object():
    assert "from websearch.layer3_agentio.mcp_server import mcp" in SRC


def test_runs_over_http_transport_not_stdio():
    assert 'transport="http"' in SRC, "must serve HTTP (the bundled CLI is stdio-only)"


def test_binds_all_interfaces_and_default_port_from_env():
    assert 'WEBSEARCH_MCP_HOST", "0.0.0.0"' in SRC
    assert 'WEBSEARCH_MCP_PORT", "8000"' in SRC
