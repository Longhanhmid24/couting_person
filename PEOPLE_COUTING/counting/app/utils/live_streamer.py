import threading
import time
import cv2
import numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_latest_frame_jpeg = None
_lock = threading.Lock()
_server = None


class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            html = """<!DOCTYPE html>
<html>
<head>
    <title>AI People Counting — Live View</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { margin: 0; background: #0f172a; color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; }
        .card { background: #1e293b; padding: 20px; border-radius: 12px; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5); text-align: center; max-width: 95vw; }
        h1 { margin: 0 0 8px 0; font-size: 20px; color: #38bdf8; }
        .desc { color: #94a3b8; font-size: 13px; margin-bottom: 14px; }
        .stream-container { position: relative; border-radius: 8px; overflow: hidden; border: 2px solid #334155; }
        img { display: block; max-width: 100%; height: auto; }
        .badge { display: inline-block; background: #22c55e; color: #052e16; font-size: 12px; font-weight: bold; padding: 4px 10px; border-radius: 20px; margin-bottom: 12px; }
        .legend { display: flex; justify-content: center; gap: 15px; margin-top: 12px; font-size: 13px; }
        .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; }
    </style>
</head>
<body>
    <div class="card">
        <span class="badge">● LIVE AI REALTIME</span>
        <h1>Camera Đếm Người (ByteTrack + Vạch Đếm)</h1>
        <div class="desc">Chấm đỏ: Điểm chân tiếp đất | Hộp vàng: Đang bám vết | Hộp xanh lá: Đã đếm | Đường vàng: Vạch đếm</div>
        <div class="stream-container">
            <img src="/video_feed" alt="AI Live Stream">
        </div>
    </div>
</body>
</html>"""
            self.wfile.write(html.encode('utf-8'))
        elif self.path == '/video_feed':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            while True:
                with _lock:
                    frame = _latest_frame_jpeg
                if frame is not None:
                    try:
                        self.wfile.write(b'--frame\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
                    except Exception:
                        break
                time.sleep(0.04)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        return  # Tắt log access http để không làm rối terminal


def update_stream_frame(frame, tracked_objects, line_points, direction_vector=None, cam_name="cam"):
    global _latest_frame_jpeg
    if frame is None:
        return
    vis = frame.copy()
    h, w = vis.shape[:2]

    # 1. Vẽ vạch đếm
    if line_points and len(line_points) >= 2:
        pt1 = tuple(int(v) for v in line_points[0])
        pt2 = tuple(int(v) for v in line_points[1])
        cv2.line(vis, pt1, pt2, (0, 215, 255), 3)  # Vàng đậm
        cv2.putText(vis, "COUNTING LINE", (pt1[0], max(25, pt1[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)

        # Vẽ hướng di chuyển nếu có
        if direction_vector:
            mx = (pt1[0] + pt2[0]) // 2
            my = (pt1[1] + pt2[1]) // 2
            vx, vy = direction_vector
            end_x = int(mx + vx * 120)
            end_y = int(my + vy * 120)
            cv2.arrowedLine(vis, (mx, my), (end_x, end_y), (255, 128, 0), 2, tipLength=0.25)
            cv2.putText(vis, "IN DIRECTION", (end_x + 5, end_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 2)

    # 2. Vẽ các đối tượng đang track
    for obj in tracked_objects or []:
        bx, by, bw, bh = obj.bbox
        x1, y1, x2, y2 = int(bx), int(by), int(bx + bw), int(by + bh)
        color = (0, 255, 0) if getattr(obj, 'counted', False) else (0, 215, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Chấm đỏ chân người
        if hasattr(obj, 'curr_bottom') and obj.curr_bottom:
            cx, cy = int(obj.curr_bottom[0]), int(obj.curr_bottom[1])
            cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)

        conf_str = f" {obj.conf:.2f}" if getattr(obj, 'conf', None) is not None else ""
        label = f"ID#{obj.id}{conf_str}"
        cv2.putText(vis, label, (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Nén JPEG chất lượng 70
    ret, jpeg = cv2.imencode('.jpg', vis, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if ret:
        with _lock:
            _latest_frame_jpeg = jpeg.tobytes()


def start_streamer(port=8899):
    global _server
    if _server is not None:
        return
    try:
        _server = ThreadingHTTPServer(('0.0.0.0', port), MJPEGHandler)
        t = threading.Thread(target=_server.serve_forever, daemon=True)
        t.start()
        print(f"[LiveStreamer] Server started on port {port}")
    except Exception as e:
        print(f"[LiveStreamer] Error starting server: {e}")
