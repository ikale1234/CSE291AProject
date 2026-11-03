"""A stand-in for vLLM's OpenAI-compatible server, for testing the evaluation pipeline without a GPU.

It validates each chat request the way vLLM + Qwen's chat template need it (known roles, images only in user
messages as decodable PNG data URLs, tool schemas present, tool results as strings) and answers 400 if not.
Replies cycle through the shapes the real model can produce: a native tool call, a `<tool_call>` block left in
the text, a malformed block, and plain text. Every request's image count is logged to `--log`.

Usage:
    python vpi_harness/mock_vllm_server.py --port 8011 --log /tmp/mock_requests.jsonl
"""

import argparse
import base64
import itertools
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "qwen3-vl-8b"
NO_AUTO_TOOLS = False
"""Like a vLLM server started without --enable-auto-tool-choice: rejects tool_choice="auto"."""
_counter = itertools.count()
_lock = threading.Lock()


def validate(body: dict) -> tuple[str | None, int]:
    images = 0
    if not body.get("tools"):
        return "no tools", 0
    for message in body["messages"]:
        role = message["role"]
        if role not in ("system", "user", "assistant", "tool"):
            return f"unknown role {role}", images
        content = message.get("content")
        if role in ("system", "tool") and not isinstance(content, str):
            return f"{role} content must be a string", images
        if isinstance(content, list):
            for part in content:
                if part["type"] == "image_url":
                    if role != "user":
                        return f"image in a {role} message", images
                    header, data = part["image_url"]["url"].split(",", 1)
                    if header != "data:image/png;base64" or not base64.b64decode(data).startswith(b"\x89PNG"):
                        return "bad image data URL", images
                    images += 1
        if role == "tool" and not message.get("tool_call_id"):
            return "tool message without tool_call_id", images
    return None, images


def reply(kind: int) -> dict:
    if kind == 0 and NO_AUTO_TOOLS:
        return {
            "role": "assistant",
            "content": '<tool_call>\n{"name": "answer_user", "arguments": {"text": "Done."}}\n</tool_call>',
            "tool_calls": [],
        }
    if kind == 0:
        call = {
            "id": "call_mock",
            "type": "function",
            "function": {"name": "answer_user", "arguments": json.dumps({"text": "Done."})},
        }
        return {"role": "assistant", "content": None, "tool_calls": [call]}
    if kind == 1:
        return {
            "role": "assistant",
            "content": '<tool_call>\n{"name": "screenshot", "arguments": {}}\n</tool_call>',
            "tool_calls": [],
        }
    if kind == 2:
        return {
            "role": "assistant",
            "content": '<tool_call>\n{"name": "screenshot", "arguments": {\n</tool_call>',
            "tool_calls": [],
        }
    return {"role": "assistant", "content": "I will look at the screen first.", "tool_calls": []}


class Handler(BaseHTTPRequestHandler):
    log_path: str | None = None

    def _send(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, {"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "mock"}]})
        else:
            self._send(200, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        error, images = validate(body)
        if NO_AUTO_TOOLS and body.get("tool_choice") == "auto":
            error = '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set'
        if self.log_path:
            with _lock, open(self.log_path, "a") as f:
                f.write(json.dumps({"images": images, "error": error, "seed": body.get("seed")}) + "\n")
        if error:
            self._send(400, {"error": {"message": error, "type": "BadRequestError"}})
            return
        with _lock:
            kind = next(_counter) % 4
        message = reply(kind)
        self._send(
            200,
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": 0,
                "model": MODEL,
                "choices": [
                    {"index": 0, "message": message, "finish_reason": "tool_calls" if message["tool_calls"] else "stop"}
                ],
                "usage": {
                    "prompt_tokens": 1000 * images + 500,
                    "completion_tokens": 20,
                    "total_tokens": 1000 * images + 520,
                },
            },
        )

    def log_message(self, format, *args):
        pass


def main() -> None:
    global MODEL, NO_AUTO_TOOLS
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--log")
    parser.add_argument("--model-name", default=MODEL)
    parser.add_argument("--no-auto-tools", action="store_true")
    args = parser.parse_args()
    MODEL, NO_AUTO_TOOLS = args.model_name, args.no_auto_tools
    Handler.log_path = args.log
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
