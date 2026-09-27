import grpc
import time
import threading
import logging
import cv2
from typing import List, Tuple

from core.settings import settings
from grpc_clients.grpc_det_person import detector_pb2, detector_pb2_grpc

logger = logging.getLogger(__name__)

try:
    from turbojpeg import TurboJPEG
    _turbo_jpeg = TurboJPEG()
except Exception:
    _turbo_jpeg = None


def _encode_jpeg(frame, quality=80):
    """Encode JPEG bằng libjpeg-turbo khi có, fallback OpenCV."""
    if _turbo_jpeg is not None:
        try:
            return _turbo_jpeg.encode(frame, quality=quality)
        except Exception:
            pass
    ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


class GRPCClient:
    """Optimized gRPC client with connection pooling, retries and backoff for Person Detection."""

    def __init__(
        self,
        yolo_person_addr: str = None,
        max_msg_mb: int = 50,
        timeout_s: float = None,
        connect_timeout_s: float = 10.0,
        max_retries: int = None,
        retry_delay: float = None,
    ):
        self.opts = [
            ("grpc.max_send_message_length",    max_msg_mb << 20),
            ("grpc.max_receive_message_length", max_msg_mb << 20),
            ("grpc.keepalive_time_ms",          120000),
            ("grpc.keepalive_timeout_ms",        5000),
            ("grpc.keepalive_permit_without_calls", True),
        ]
        self.timeout = timeout_s if timeout_s is not None else getattr(settings, 'GRPC_TIMEOUT', 3.0)
        self.connect_timeout = connect_timeout_s
        self.max_retries = max_retries if max_retries is not None else getattr(settings, 'GRPC_MAX_RETRIES', 2)
        self.retry_delay = retry_delay if retry_delay is not None else getattr(settings, 'GRPC_RETRY_DELAY', 0.2)

        self.yolo_person_addr = yolo_person_addr or getattr(settings, 'YOLO_PERSON_ADDR', 'localhost:50060')

        self._lock = threading.Lock()
        self.yolo_person: detector_pb2_grpc.YoloServiceStub = None

    def _make_channel(self, addr: str) -> grpc.Channel:
        """Tạo và kiểm tra READY channel gRPC."""
        ch = grpc.insecure_channel(addr, options=self.opts)
        try:
            grpc.channel_ready_future(ch).result(timeout=self.connect_timeout)
            logger.info(f"[gRPC] Connected to {addr}")
        except grpc.FutureTimeoutError:
            logger.error(f"[gRPC] Could not connect to {addr} within {self.connect_timeout}s")
            raise
        return ch

    def _get_person_stub(self):
        with self._lock:
            if self.yolo_person is None:
                ch = self._make_channel(self.yolo_person_addr)
                self.yolo_person = detector_pb2_grpc.YoloServiceStub(ch)
            return self.yolo_person

    def detect_person_yolo(
        self, frame: cv2.Mat, allowed_classes: List[int] = None
    ) -> List[Tuple[int, int, int, int, float, str]]:
        """
        Detect person using YOLO over gRPC, with retries.
        Returns list of (x1, y1, x2, y2, confidence, class_name).
        """
        size = settings.TARGET_SIZE
        rs = cv2.resize(frame, (size, size))
        image_bytes = _encode_jpeg(rs, quality=80)
        req = detector_pb2.DetectRequest(
            image=image_bytes,
            resized_width=size,
            resized_height=size,
            orig_width=frame.shape[1],
            orig_height=frame.shape[0],
            allowed_classes=allowed_classes or [],
        )

        for attempt in range(self.max_retries):
            try:
                stub = self._get_person_stub()
                resp = stub.DetectFrame(req, timeout=self.timeout)
                return [
                    (
                        b.x1,
                        b.y1,
                        b.x2,
                        b.y2,
                        b.confidence,
                        getattr(b, "class_name", "person") or "person",
                    )
                    for b in resp.bboxes
                ]
            except grpc.RpcError as e:
                code = e.code()
                logger.warning(
                    f"[gRPC DET-PERSON] Attempt {attempt+1}/{self.max_retries} failed: {code} - {e.details()}"
                )
                if code in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED):
                    with self._lock:
                        self.yolo_person = None  # Force reconnect
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    break
            except Exception as e:
                logger.error(f"[gRPC DET-PERSON] Unexpected error: {e}")
                break

        return []
