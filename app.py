from __future__ import annotations

import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from wine_atelier.model import metadata, predict, train_and_save


ROOT_DIR = Path(__file__).resolve().parent
STATIC_DIR = ROOT_DIR / "static"


class WineAtelierHandler(BaseHTTPRequestHandler):
    server_version = "WineQualityAtelier/1.0"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._serve_file(STATIC_DIR / "index.html")
        elif path == "/api/health":
            self._json({"status": "ok", "modelReady": True})
        elif path == "/api/metadata":
            self._json(metadata())
        elif path.startswith("/static/"):
            self._serve_static(path.removeprefix("/static/"))
        else:
            self._json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/predict":
                payload = self._read_json()
                self._json(predict(payload).as_dict())
            elif path == "/api/retrain":
                self._json(train_and_save())
            else:
                self._json({"error": "Not found"}, status=404)
        except ValueError as exc:
            self._json({"error": str(exc)}, status=400)
        except Exception as exc:
            self._json({"error": "Server error", "detail": str(exc)}, status=500)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("Request body must be valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object")
        return payload

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, relative_path: str) -> None:
        requested = (STATIC_DIR / unquote(relative_path)).resolve()
        try:
            requested.relative_to(STATIC_DIR.resolve())
        except ValueError:
            self._json({"error": "Not found"}, status=404)
            return
        self._serve_file(requested)

    def _serve_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self._json({"error": "Not found"}, status=404)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    metadata()
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8765"))
    server = ThreadingHTTPServer((host, port), WineAtelierHandler)
    print(f"Wine Quality Atelier running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
