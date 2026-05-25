"""Minimal HTTP server serving the training monitor UI and recorded videos."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import unquote

_HERE = Path(__file__).parent


class _Handler(BaseHTTPRequestHandler):
    video_dir: Path = None
    state: dict = {}

    def log_message(self, *_):
        pass

    def do_GET(self):
        path = unquote(self.path.split("?")[0])

        if path in ("/", "/index.html"):
            self._send_file(_HERE / "index.html", "text/html")

        elif path == "/api/videos":
            files = sorted(f.name for f in self.video_dir.glob("*.mp4"))
            body = json.dumps({
                "videos": files,
                "iteration": self.state.get("iteration"),
                "alive": self.state.get("alive", False),
            }).encode()
            self._respond(200, "application/json", body)

        elif path.startswith("/videos/"):
            filename = path[len("/videos/"):]
            filepath = self.video_dir / filename
            if filepath.exists() and filepath.suffix == ".mp4":
                self._send_file(filepath, "video/mp4")
            else:
                self._respond(404, "text/plain", b"not found")

        else:
            self._respond(404, "text/plain", b"not found")

    def _send_file(self, filepath: Path, content_type: str):
        data = filepath.read_bytes()
        self._respond(200, content_type, data)

    def _respond(self, code: int, content_type: str, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


def start(video_dir: Path, port: int = 8080) -> dict:
    """Start HTTP server in a background daemon thread.

    Returns a shared state dict — update state['iteration'] and state['alive']
    from the training loop to reflect current progress in the UI.
    """
    video_dir.mkdir(parents=True, exist_ok=True)

    state = {"iteration": None, "alive": True}
    _Handler.video_dir = video_dir
    _Handler.state = state

    server = HTTPServer(("0.0.0.0", port), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    return state
