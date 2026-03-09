import base64
import io
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

try:
    import numpy as np
    NUMPY_FOUND = True
except ImportError:
    NUMPY_FOUND = False

try:
    from PIL import Image
    PIL_FOUND = True
except ImportError:
    PIL_FOUND = False


class _FrameHub:
    def __init__(self, initial_camera_index=0, camera_count=1):
        self._lock = threading.Lock()
        self._frame_cond = threading.Condition(self._lock)
        self._frame_payload = None
        self._frame_version = 0
        self._camera_index = initial_camera_index
        self._camera_count = max(1, camera_count)

    def set_camera_count(self, camera_count):
        with self._lock:
            self._camera_count = max(1, camera_count)
            self._camera_index = max(0, min(self._camera_index, self._camera_count - 1))

    def set_camera_index(self, camera_index):
        with self._lock:
            self._camera_index = max(0, min(camera_index, self._camera_count - 1))

    def get_camera_index(self):
        with self._lock:
            return self._camera_index

    def publish(self, payload):
        with self._frame_cond:
            self._frame_payload = payload
            self._frame_version += 1
            self._frame_cond.notify_all()

    def get_latest(self):
        with self._lock:
            return self._frame_payload

    def wait_next(self, previous_version, timeout_sec=15.0):
        with self._frame_cond:
            if self._frame_version == previous_version:
                self._frame_cond.wait(timeout=timeout_sec)
            return self._frame_payload, self._frame_version


class WebViewerServer:
    def __init__(self, host, port, initial_camera_index=0, camera_count=1, debug=False):
        self.host = host
        self.port = port
        self.debug = debug
        self._hub = _FrameHub(initial_camera_index=initial_camera_index, camera_count=camera_count)
        self._httpd = None
        self._thread = None
        self._published_frames = 0

    def _debug(self, message):
        if self.debug:
            print(f"[WebViewer] {message}")

    def start(self):
        hub = self._hub

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):
                return

            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path == "/":
                    html = _index_html().encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(html)))
                    self.end_headers()
                    self.wfile.write(html)
                    return

                if parsed.path == "/camera":
                    query = parse_qs(parsed.query)
                    try:
                        camera_index = int(query.get("index", ["0"])[0])
                    except ValueError:
                        camera_index = 0
                    hub.set_camera_index(camera_index)
                    self.server.parent._debug(f"Camera index set to {hub.get_camera_index()} by {self.client_address}")
                    payload = json.dumps({"ok": True, "camera_index": hub.get_camera_index()}).encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return

                if parsed.path == "/latest":
                    payload = hub.get_latest()
                    if payload is None:
                        self.server.parent._debug(f"/latest requested by {self.client_address}, but no frame yet")
                        self.send_response(HTTPStatus.NO_CONTENT)
                        self.end_headers()
                        return
                    body = json.dumps(payload).encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    self.server.parent._debug(f"/latest served to {self.client_address}: iter={payload.get('iteration')}, cam={payload.get('camera_index')}")
                    return

                if parsed.path == "/stream":
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.send_header("X-Accel-Buffering", "no")
                    self.end_headers()
                    self.server.parent._debug(f"SSE connected: {self.client_address}")
                    version = -1
                    try:
                        while True:
                            payload, version = hub.wait_next(version)
                            if payload is None:
                                self.wfile.write(b": keepalive\n\n")
                            else:
                                self.wfile.write(f"event: frame\ndata: {json.dumps(payload)}\n\n".encode("utf-8"))
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        self.server.parent._debug(f"SSE disconnected: {self.client_address}")
                        return

                self.send_response(HTTPStatus.NOT_FOUND)
                self.end_headers()

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self._httpd.parent = self
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        self._debug(f"Server started at http://{self.host}:{self.port}")

    def stop(self):
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()

    def set_camera_count(self, camera_count):
        self._hub.set_camera_count(camera_count)

    def get_camera_index(self):
        return self._hub.get_camera_index()

    def publish_frame(self, image_data_url, iteration, camera_index, camera_name):
        self._published_frames += 1
        self._hub.publish({
            "iteration": iteration,
            "camera_index": camera_index,
            "camera_name": camera_name,
            "image": image_data_url,
        })
        if self.debug:
            self._debug(
                f"Published frame#{self._published_frames}: iter={iteration}, cam={camera_index}, name={camera_name}"
            )


def tensor_to_jpeg_data_url(render_tensor, quality=80):
    if not PIL_FOUND:
        raise RuntimeError("Pillow is required for --web_viewer. Please install pillow.")

    rgb = (render_tensor.detach().clamp(0.0, 1.0) * 255.0).byte().permute(1, 2, 0).contiguous().cpu()
    height, width = rgb.shape[0], rgb.shape[1]
    if NUMPY_FOUND:
        image = Image.fromarray(np.asarray(rgb.numpy()), mode="RGB")
    else:
        image = Image.frombytes("RGB", (width, height), bytes(rgb.view(-1).tolist()))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _index_html():
    return """<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>3DGS Training Web Viewer</title>
  <style>
    body { font-family: sans-serif; margin: 16px; }
    img { max-width: 100%; border: 1px solid #ccc; }
    .row { margin-bottom: 12px; }
  </style>
</head>
<body>
  <h2>3D Gaussian Splatting - Training Web Viewer</h2>
  <div class=\"row\">
    <label for=\"cameraIndex\">Camera index:</label>
    <input id=\"cameraIndex\" type=\"number\" min=\"0\" value=\"0\" />
    <button id=\"applyCamera\">Apply</button>
  </div>
  <div class=\"row\" id=\"meta\">Waiting for frames...</div>
  <img id=\"frame\" alt=\"training frame\"/>

  <script>
    const img = document.getElementById('frame');
    const meta = document.getElementById('meta');
    const input = document.getElementById('cameraIndex');

    function updateFrame(data) {
      if (!data || !data.image) return;
      img.src = data.image;
      meta.textContent = `Iteration: ${data.iteration} | Camera[${data.camera_index}]: ${data.camera_name}`;
    }

    async function pollLatest() {
      try {
        const response = await fetch('/latest', { cache: 'no-store' });
        if (response.status === 200) {
          const data = await response.json();
          updateFrame(data);
        }
      } catch (err) {
      }
    }

    document.getElementById('applyCamera').onclick = async () => {
      await fetch(`/camera?index=${encodeURIComponent(input.value)}`);
      await pollLatest();
    };

    const source = new EventSource('/stream');
    source.addEventListener('frame', (event) => {
      const data = JSON.parse(event.data);
      updateFrame(data);
    });
    source.onerror = () => {
      source.close();
    };

    setInterval(pollLatest, 1000);
    pollLatest();
  </script>
</body>
</html>
"""
