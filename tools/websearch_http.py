#!/usr/bin/env python3
"""Serve the websearch-skill MCP server over HTTP (streamable-http).

The bundled `websearch mcp` command speaks stdio only, which suits an MCP client
that spawns a child process. To make web_search/web_fetch reachable over the
network (so the model under test, or any MCP client on the compose network, can
call it), this thin wrapper flips the very same FastMCP server object to HTTP
transport. It is the entry point of the optional `websearch` service in
docker-compose.yml (profile: tools); endpoint: http://<host>:<port>/mcp

Env:
  WEBSEARCH_MCP_HOST   bind address (default 0.0.0.0)
  WEBSEARCH_MCP_PORT   bind port    (default 8000)
  WEBSEARCH_PERSIST_PATH  optional SQLite store so web_open resolves handles
  WEBSEARCH_SEARXNG_URL   optional SearXNG base URL to fuse a second engine
(the last two are read by the websearch package itself).
"""
import os

from websearch.layer3_agentio.mcp_server import mcp


def main() -> None:
    mcp.run(
        transport="http",
        host=os.environ.get("WEBSEARCH_MCP_HOST", "0.0.0.0"),
        port=int(os.environ.get("WEBSEARCH_MCP_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
