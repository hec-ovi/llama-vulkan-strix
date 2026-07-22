"""End-to-end tests for scripts/bench_server.py against a local fake llama-server."""
import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "bench_server.py"
SPEC = importlib.util.spec_from_file_location("bench_server", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeLlamaServer(BaseHTTPRequestHandler):
    """Answers /completion with canned timings; records every request body."""

    requests: list[dict] = []
    # per_second values cycle so best-of-N has a real max to find
    rates = [(400.0, 30.0), (500.0, 35.0), (450.0, 32.0)]

    def do_POST(self):
        assert self.path == "/completion"
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        FakeLlamaServer.requests.append(body)
        prefill, decode = self.rates[(len(self.requests) - 1) % len(self.rates)]
        payload = json.dumps(
            {
                "timings": {
                    "prompt_n": len(body["prompt"].split()),
                    "prompt_per_second": prefill,
                    "predicted_n": body["n_predict"],
                    "predicted_per_second": decode,
                }
            }
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture()
def server():
    FakeLlamaServer.requests = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeLlamaServer)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def run_main(argv, capsys):
    import sys

    old = sys.argv
    sys.argv = ["bench_server.py", *argv]
    try:
        code = MODULE.main()
    finally:
        sys.argv = old
    out, err = capsys.readouterr()
    return code, out, err


def test_best_of_n_markdown_table_through_main(server, capsys):
    code, out, _ = run_main(
        ["--url", server, "--depths", "2048,8192", "--runs", "3"], capsys
    )
    assert code == 0
    # 3 runs per depth, 2 depths
    assert len(FakeLlamaServer.requests) == 6
    # best-of-3 picks the maximum rates the fake cycles through
    assert "| 2k " in out and "| 8k " in out
    assert "| 500 | 35.0 |" in out


def test_requests_are_fresh_forced_length_and_unique_per_run(server, capsys):
    code, _, _ = run_main(
        ["--url", server, "--depths", "2048", "--runs", "3", "--decode-tokens", "64"],
        capsys,
    )
    assert code == 0
    prompts = [request["prompt"] for request in FakeLlamaServer.requests]
    assert len(set(prompts)) == 3  # no shared prefix cache hits across runs
    for request in FakeLlamaServer.requests:
        assert request["cache_prompt"] is False
        assert request["ignore_eos"] is True
        assert request["n_predict"] == 64


def test_json_output_mode(server, capsys):
    code, out, _ = run_main(["--url", server, "--depths", "4096", "--json"], capsys)
    assert code == 0
    rows = json.loads(out)
    assert rows[0]["depth"] == 4096
    assert rows[0]["prefill"] == 500.0


def test_unreachable_server_fails_with_error(capsys):
    code, _, err = run_main(
        ["--url", "http://127.0.0.1:1", "--depths", "2048", "--timeout", "2"], capsys
    )
    assert code == 1
    assert "bench error" in err


def test_prompt_scales_with_depth_and_differs_by_run():
    short = MODULE.build_prompt(2048, 0)
    long = MODULE.build_prompt(32768, 0)
    assert len(long.split()) > 10 * len(short.split())
    assert MODULE.build_prompt(2048, 1) != short
