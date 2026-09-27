"""
central_pipeline.py — Centralized People Counting Pipeline.
Pipeline điều phối đếm người:
  - Tiếp nhận frame từ shared reader queue của CentralEngine
  - Phân luồng về từng CameraPipelineInstance
  - Gọi PersonProcessor (YOLO + tracking + đếm vượt vạch)
  - Gọi StatisticsPoster (tổng hợp và POST /api/camera-statistics định kỳ)
"""
import threading
import queue
import logging
import time
import numpy as np
from typing import Dict, Optional, List

from camera_reader import FramePacket
from services.person_process import PersonProcessor
from services.statistics_poster import PersonCounter, StatisticsPoster
from core.settings import settings
from utils.helper import extract_rule_config

LOGGER = logging.getLogger(__name__)


def parse_camera_params(cam: dict, rules: list = None):
    """
    Hook cho CentralEngine: trả về {"line_config": ..., "direction_config": ...} cho camera này.
    BắT BUỘC phải có cả line và direction trên CMS.
    Trả về None (SKIP camera) nếu thiếu 1 trong 2.
    """
    cam_id = cam.get('id')
    cam_name = cam.get('name') or (cam_id or '')[:8]
    param = cam.get('parameter') or {}

    rule_cfg = extract_rule_config(
        cam_id=cam_id,
        rules=rules,
        rule_types=("people_couting", "people_counting"),
        param=param
    )

    line_config = rule_cfg.get("line")
    if not line_config:
        LOGGER.warning(
            f"[{cam_name}] ⚠️ Chưa cấu hình counting line — bỏ qua camera"
        )
        return None

    # Đọc trực tiếp danh sách filter từ rules API để kiểm tra có filter type="direction" riêng biệt
    # (không dùng rule_cfg.get("direction") vì helper.py fallback copy line → direction)
    direction_config = _find_direction_filter(cam_id, rules)
    if not direction_config:
        LOGGER.warning(
            f"[{cam_name}] ⚠️ Chưa kẻ hướng đếm (direction) — bỏ qua camera. "
            f"Vui lòng vẽ mũi tên hướng trên CMS UI."
        )
        return None

    return {"line_config": line_config, "direction_config": direction_config}


def _find_direction_filter(cam_id: str, rules: list) -> list:
    """
    Tìm filter type='direction' riêng biệt từ danh sách rules gốc (API /api/rules).
    Tránh dùng extract_rule_config vì nó fallback copy line → direction.
    """
    if not rules or not isinstance(rules, list) or not cam_id:
        return None
    for r in rules:
        if not isinstance(r, dict) or r.get('status') != 1:
            continue
        r_type = str(r.get('type', '')).lower().strip()
        if r_type not in ('people_couting', 'people_counting'):
            continue
        # Kiểm tra camera có thuộc rule này không
        stream_ids = r.get('stream_ids') or []
        filters = r.get('filters') or []
        has_stream = cam_id in stream_ids
        if not has_stream:
            for f in filters:
                if isinstance(f, dict) and f.get('stream_id') == cam_id:
                    has_stream = True
                    break
        if not has_stream:
            continue
        # Tìm filter type='direction' cho camera này
        for f in filters:
            if not isinstance(f, dict) or f.get('status') != 1:
                continue
            fstream = f.get('stream_id')
            if fstream is not None and fstream != '' and fstream != cam_id:
                continue
            ftype = str(f.get('type', '')).lower().strip()
            if ftype == 'direction':
                cond = f.get('condition')
                if isinstance(cond, str):
                    import json
                    try:
                        cond = json.loads(cond)
                    except Exception:
                        pass
                if isinstance(cond, list) and len(cond) >= 2:
                    return cond
    return None


class CameraPipelineInstance:
    """Quản lý các luồng xử lý đếm người cho 1 camera cụ thể."""

    def __init__(
        self,
        cam_id: str,
        name: Optional[str] = None,
        line_config=None,
        direction_config=None,
        queue_size: int = 32,
    ):
        self.cam_id = cam_id
        self.cam_name = name or cam_id
        self.line_config = line_config
        self.direction_config = direction_config
        self.shutdown_flag = threading.Event()

        # Frame queue cho processor
        self.frame_queue = queue.Queue(maxsize=queue_size)

        # Bộ đếm người thread-safe
        self.counter = PersonCounter(cam_id, cam_name=self.cam_name)

        # Bộ xử lý bám vết và cắt vạch
        self.person_processor = PersonProcessor(
            cam_id,
            self.frame_queue,
            counting_callback=self.counter.increment,
            shutdown_flag=self.shutdown_flag,
            line_config=line_config,
            direction_config=direction_config,
            cam_name=self.cam_name,
        )

        # Thread POST thống kê lên CMS
        self.stats_poster = StatisticsPoster(
            cam_id,
            self.counter,
            self.shutdown_flag,
            cam_name=self.cam_name,
        )

        # Quản lý Threads
        self.threads: List[threading.Thread] = []
        self.t_person: Optional[threading.Thread] = None
        self.t_stats: Optional[threading.Thread] = None

    def start(self):
        """Khởi chạy các worker thread cho camera."""
        LOGGER.info(f"Bắt đầu pipeline đếm người cho camera [{self.cam_name}]")

        t_person = threading.Thread(
            target=self.person_processor.process_person,
            name=f"PersonProc-{self.cam_name}",
            daemon=True,
        )
        t_stats = threading.Thread(
            target=self.stats_poster.run,
            name=f"StatsPost-{self.cam_name}",
            daemon=True,
        )

        self.t_person = t_person
        self.t_stats = t_stats
        self.threads = [t_person, t_stats]
        for t in self.threads:
            t.start()

    def update_params(self, line_config=None, direction_config=None):
        """Cập nhật line_config và direction_config động cho camera."""
        self.line_config = line_config
        self.direction_config = direction_config
        if hasattr(self, 'person_processor') and self.person_processor is not None:
            self.person_processor.line_config = line_config
            self.person_processor._line_resolved = False
            self.person_processor.direction_config = direction_config
            self.person_processor._direction_resolved = False
            self.person_processor._direction_vector = None
            LOGGER.info(f"[{self.cam_name}] Đã cập nhật counting line + direction mới")

    def stop(self):
        """Drain processor trước, rồi mới cho StatisticsPoster flush lần cuối."""
        LOGGER.info(f"Dừng pipeline workers cho camera [{self.cam_name}]")
        try:
            self.frame_queue.put(None, timeout=1.0)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(None)
            except (queue.Empty, queue.Full):
                pass

        if self.t_person and self.t_person.is_alive():
            self.t_person.join(timeout=10.0)

        self.shutdown_flag.set()
        if self.t_stats and self.t_stats.is_alive():
            self.t_stats.join(timeout=15.0)

        LOGGER.info(f"[{self.cam_name}] Tổng số người đã đếm từ khi khởi động: {self.counter.get_total()}")
        LOGGER.info(f"Đã dừng pipeline workers cho camera [{self.cam_name}]")

    def enqueue_frame(
        self, timestamp: float, frame_uuid: str, frame: np.ndarray, frame_count: int
    ):
        """Đẩy frame vào hàng đợi xử lý."""
        try:
            self.frame_queue.put(
                (timestamp, frame_uuid, frame, frame_count), block=False
            )
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put(
                    (timestamp, frame_uuid, frame, frame_count), block=False
                )
            except (queue.Empty, queue.Full):
                pass


class CentralPipeline:
    """
    Centralized Pipeline Orchestrator cho People Counting.
    Phân phối frame từ shared queue tới từng camera pipeline instance.
    """

    def __init__(self, shared_frame_queue: queue.Queue):
        self.shared_frame_queue = shared_frame_queue
        self.shutdown = threading.Event()
        self.pipelines: Dict[str, CameraPipelineInstance] = {}
        self._camera_params: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._dispatcher_thread: Optional[threading.Thread] = None

    def start(self):
        LOGGER.info("CentralPipeline khởi động...")
        self._dispatcher_thread = threading.Thread(
            target=self._dispatcher_loop, name="pipeline-dispatcher", daemon=True
        )
        self._dispatcher_thread.start()
        LOGGER.info("CentralPipeline đã sẵn sàng")

    def stop(self):
        LOGGER.info("CentralPipeline đang dừng...")
        self.shutdown.set()
        with self._lock:
            instances = list(self.pipelines.values())
            self.pipelines.clear()

        for instance in instances:
            try:
                instance.stop()
            except Exception as exc:
                LOGGER.error(f"Lỗi khi dừng pipeline [{instance.cam_name}]: {exc}")
        LOGGER.info("CentralPipeline đã dừng")

    def register_camera(
        self,
        cam_id: str,
        name: Optional[str] = None,
        line_config=None,
        direction_config=None,
        **_ignored,
    ):
        with self._lock:
            self._camera_params[cam_id] = {"line_config": line_config, "direction_config": direction_config}
            if cam_id not in self.pipelines:
                instance = CameraPipelineInstance(
                    cam_id=cam_id, name=name, line_config=line_config,
                    direction_config=direction_config,
                )
                instance.start()
                self.pipelines[cam_id] = instance
                LOGGER.info(f"Đã đăng ký pipeline đếm người cho [{name or cam_id}]")
            else:
                self.pipelines[cam_id].update_params(line_config=line_config, direction_config=direction_config)

    def update_params(self, cam_id: str, line_config=None, direction_config=None, **_ignored):
        with self._lock:
            self._camera_params[cam_id] = {"line_config": line_config, "direction_config": direction_config}
            instance = self.pipelines.get(cam_id)
            if instance:
                instance.update_params(line_config=line_config, direction_config=direction_config)
                LOGGER.info(f"[{instance.cam_name}] Cập nhật cấu hình line + direction động")

    def get_camera_params(self, cam_id: str) -> dict:
        with self._lock:
            return self._camera_params.get(cam_id, {})

    def unregister_camera(self, cam_id: str):
        with self._lock:
            self._camera_params.pop(cam_id, None)
            instance = self.pipelines.pop(cam_id, None)
        if instance:
            instance.stop()
            LOGGER.info(f"Đã hủy đăng ký pipeline cho [{instance.cam_name}]")

    def _dispatcher_loop(self):
        """Đọc FramePackets từ shared queue và phân phối tới camera pipeline tương ứng."""
        while not self.shutdown.is_set():
            try:
                pkt: FramePacket = self.shared_frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            with self._lock:
                instance = self.pipelines.get(pkt.cam_id)

            if instance is None:
                continue

            if (
                pkt.motion_detected
                or not getattr(settings, "USE_MOTION_DETECTOR", False)
            ) and pkt.frame is not None:
                instance.enqueue_frame(
                    timestamp=pkt.timestamp,
                    frame_uuid=pkt.frame_uuid,
                    frame=pkt.frame,
                    frame_count=pkt.frame_count,
                )
