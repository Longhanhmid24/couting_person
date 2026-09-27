"""
server.py — Unified gRPC YOLO detector server with dynamic batching.
(identical bytes in every YOLO detector; behaviour driven by core/settings.py)

settings.py must define:
    GRPC_PORT     listen port
    LOG_TAG       log prefix, e.g. "DET-VEHICLE"
and the generated proto modules must be services/detector_pb2{,_grpc}.py.

The BoundingBox message MAY or MAY NOT contain class_name — handled
automatically via descriptor introspection (plate proto has no class_name).
"""
import os
import queue
import threading
import time
from collections import Counter
from concurrent import futures

import cv2
import grpc
import numpy as np
import torch

from core.settings import settings
from services import detector_pb2
from services import detector_pb2_grpc
from services.yolo_model import init_yolo, detect_batch

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 8))
BATCH_TIMEOUT_SEC = float(os.getenv("BATCH_TIMEOUT_MS", 10)) / 1000.0
# Mỗi RPC chiếm 1 worker thread trong lúc CHỜ event (không tính toán gì cả), nên
# max_workers chính là số request tối đa được phép nằm trong hàng đợi batch.
# Cũ là 16: từ camera thứ 17 trở đi request phải xếp hàng NGOÀI server, batch
# không bao giờ đầy được. Thread chờ event rất rẻ nên nâng thoải mái.
MAX_WORKERS = int(os.getenv("GRPC_MAX_WORKERS", 64))
# Log gộp thay vì in mỗi batch (in mỗi batch = tranh lock stdout trong vòng lặp nóng).
LOG_EVERY_SEC = float(os.getenv("LOG_EVERY_SEC", 5.0))
# Chờ tối đa trong DetectFrame khi client KHÔNG đặt deadline.
FALLBACK_WAIT_SEC = float(os.getenv("BATCH_WAIT_SEC", 10.0))
_HAS_CLASS_NAME = "class_name" in detector_pb2.BoundingBox.DESCRIPTOR.fields_by_name


class YoloServiceServicer(detector_pb2_grpc.YoloServiceServicer):
    def __init__(self):
        self.model, self.device, self.model_is_pt = init_yolo()
        self.batch_queue = queue.Queue()
        self.shutdown_flag = threading.Event()

        # Thống kê gộp cho log định kỳ
        self._stat_lock = threading.Lock()
        self._stat_t0 = time.time()
        self._stat_batches = 0
        self._stat_frames = 0
        self._stat_gpu_ms = 0.0
        self._stat_detected = 0
        self._stat_dropped = 0        # request client đã bỏ trước khi tới lượt
        self._stat_classes = Counter()

        # Background Dynamic Batch Worker
        self.worker_thread = threading.Thread(target=self._batch_worker_loop, daemon=True)
        self.worker_thread.start()
        print(f"[YoloService] Dynamic Batching enabled "
              f"(BATCH_SIZE={BATCH_SIZE}, TIMEOUT={BATCH_TIMEOUT_SEC*1000:.0f}ms, "
              f"MAX_WORKERS={MAX_WORKERS}). Model ready.")

    def _still_wanted(self, req_tuple):
        """
        Bỏ request mà client đã bỏ chờ (huỷ RPC / hết deadline). Không có bước
        này thì GPU vẫn chạy cho những frame không ai nhận kết quả — vừa mất
        slot trong batch vừa mất thời gian GPU, đúng lúc hệ đang quá tải.
        """
        ctx = req_tuple.get("ctx")
        if ctx is None:
            return True
        try:
            if not ctx.is_active():
                req_tuple["event"].set()
                with self._stat_lock:
                    self._stat_dropped += 1
                return False
            remaining = ctx.time_remaining()
            if remaining is not None and remaining <= 0:
                req_tuple["event"].set()
                with self._stat_lock:
                    self._stat_dropped += 1
                return False
        except Exception:
            return True
        return True

    def _batch_worker_loop(self):
        while not self.shutdown_flag.is_set():
            # Chờ request ĐẦU TIÊN không giới hạn thời gian. Bản cũ tính
            # remaining_time từ BATCH_TIMEOUT (10ms) ngay từ item đầu, nên khi
            # hàng đợi rỗng nó quay vòng get(timeout=0.001) — tức ~1000 vòng/giây
            # đốt CPU vô ích trên đúng cái máy đang phải chạy GPU.
            try:
                first = self.batch_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            batch_requests = []
            if self._still_wanted(first):
                batch_requests.append(first)

            # Đã có item đầu -> mở cửa sổ BATCH_TIMEOUT để gom thêm.
            deadline = time.monotonic() + BATCH_TIMEOUT_SEC
            while len(batch_requests) < BATCH_SIZE:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    req_tuple = self.batch_queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if self._still_wanted(req_tuple):
                    batch_requests.append(req_tuple)

            if not batch_requests:
                continue

            t_gpu_start = time.perf_counter()

            batch_items = [r["item"] for r in batch_requests]
            try:
                batch_results = detect_batch(
                    model=self.model,
                    device=self.device,
                    model_is_pt=self.model_is_pt,
                    batch_items=batch_items,
                )
            except Exception as e:
                print(f"[YOLO ERROR] Batch detect error: {e}")
                if self.device != "cpu":
                    torch.cuda.empty_cache()
                batch_results = [[] for _ in batch_requests]

            t_gpu_dur_ms = (time.perf_counter() - t_gpu_start) * 1000.0

            # detect_batch phải trả đúng 1 kết quả / request; lệch thì bù rỗng
            # thay vì để IndexError giết luôn worker (mất batching vĩnh viễn).
            if len(batch_results) != len(batch_requests):
                print(f"[YOLO WARN] batch_results={len(batch_results)} != "
                      f"requests={len(batch_requests)}, bù danh sách rỗng")
                batch_results = list(batch_results) + \
                    [[] for _ in range(len(batch_requests) - len(batch_results))]

            # Distribute results to waiting gRPC handlers
            total_detected = 0
            class_counts = Counter()
            for idx, req_tuple in enumerate(batch_requests):
                bboxes = batch_results[idx]
                req_tuple["result"] = bboxes
                for box in bboxes:
                    total_detected += 1
                    class_counts[box["class_name"]] += 1
                req_tuple["event"].set()

            self._accumulate(len(batch_requests), t_gpu_dur_ms,
                             total_detected, class_counts)

    def _accumulate(self, n_frames, gpu_ms, detected, class_counts):
        """Gộp số liệu, in một dòng mỗi LOG_EVERY_SEC."""
        now = time.time()
        with self._stat_lock:
            self._stat_batches += 1
            self._stat_frames += n_frames
            self._stat_gpu_ms += gpu_ms
            self._stat_detected += detected
            self._stat_classes.update(class_counts)
            elapsed = now - self._stat_t0
            if elapsed < LOG_EVERY_SEC:
                return
            batches, frames = self._stat_batches, self._stat_frames
            gpu_ms_tot, detected_tot = self._stat_gpu_ms, self._stat_detected
            dropped, classes = self._stat_dropped, self._stat_classes
            self._stat_t0 = now
            self._stat_batches = self._stat_frames = self._stat_detected = 0
            self._stat_dropped = 0
            self._stat_gpu_ms = 0.0
            self._stat_classes = Counter()

        avg_batch = frames / batches if batches else 0.0
        fill = avg_batch / BATCH_SIZE * 100.0 if BATCH_SIZE else 0.0
        class_summary = ", ".join(f"{k}: {v}" for k, v in classes.items()) or "none"
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] [{settings.LOG_TAG}] {frames} frame / {elapsed:.1f}s "
              f"= {frames / elapsed:.1f} fps | batch tb {avg_batch:.1f}/{BATCH_SIZE} "
              f"({fill:.0f}% đầy) | GPU tb {gpu_ms_tot / batches:.1f}ms"
              f"{f' | bỏ {dropped}' if dropped else ''} | "
              f"detected {detected_tot} [{class_summary}]")

    def DetectFrame(self, request, context):
        # Decode JPEG → BGR
        buf = np.frombuffer(request.image, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Cannot decode JPEG")
            return detector_pb2.DetectResponse()

        # Ensure resized dims match
        if img.shape[0] != request.resized_height or img.shape[1] != request.resized_width:
            img = cv2.resize(
                img,
                (request.resized_width, request.resized_height),
                interpolation=cv2.INTER_AREA,
            )

        item = {
            "img": img,
            "orig_w": request.orig_width,
            "orig_h": request.orig_height,
            "allowed_classes": list(request.allowed_classes) or None,
        }
        event = threading.Event()
        req_tuple = {"item": item, "event": event, "result": [], "ctx": context}

        # Push to batch queue and wait for Dynamic Batch Worker.
        # Chờ theo ĐÚNG deadline client đặt, thay vì hardcode 2.0s như bản cũ:
        # client cho 4s (hoặc 10s ở bản cũ) mà server tự bỏ ở 2s thì frame bị
        # trả DEADLINE_EXCEEDED trong khi client vẫn đang sẵn sàng đợi — mất
        # frame một cách vô cớ đúng lúc GPU tải cao và batch đang xếp hàng.
        self.batch_queue.put(req_tuple)
        remaining = context.time_remaining()
        if remaining is None:
            remaining = FALLBACK_WAIT_SEC
        # chừa một chút để còn kịp build response trước khi gRPC cắt
        wait_s = max(0.05, remaining - 0.05)
        if not event.wait(timeout=wait_s):
            context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
            context.set_details("Batch processing timeout")
            return detector_pb2.DetectResponse()

        bboxes = req_tuple["result"]

        # Build gRPC response
        resp = detector_pb2.DetectResponse()
        for box in bboxes:
            bb = resp.bboxes.add()
            bb.x1 = box["x1"]
            bb.y1 = box["y1"]
            bb.x2 = box["x2"]
            bb.y2 = box["y2"]
            bb.confidence = box["confidence"]
            if _HAS_CLASS_NAME:
                bb.class_name = box["class_name"]

        return resp


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_WORKERS),
                         options=[
                             # Cho phép nhiều stream trên MỘT channel: client dùng
                             # channel dùng chung cho tất cả camera, mà default
                             # MAX_CONCURRENT_STREAMS có thể chặn ở 100.
                             ("grpc.max_concurrent_streams", MAX_WORKERS * 4),
                             ("grpc.keepalive_permit_without_calls", True),
                             ("grpc.http2.min_ping_interval_without_data_ms", 30000),
                         ])
    detector_pb2_grpc.add_YoloServiceServicer_to_server(YoloServiceServicer(), server)
    server.add_insecure_port(f"[::]:{settings.GRPC_PORT}")
    server.start()
    print(f"[Server] {settings.LOG_TAG} gRPC on port {settings.GRPC_PORT} "
          f"(Dynamic Batching Enabled)")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
