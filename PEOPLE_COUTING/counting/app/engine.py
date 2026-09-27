"""
engine.py — Unified CentralEngine template (identical across ALL orchestrator services).

Single-process camera orchestrator:
  - Polls camera list from CMS API every POLL_INTERVAL seconds
  - Keeps only cameras whose service_type intersects settings.SERVICE_TYPES
  - Starts/stops readers (GPU NVDEC -> CPU fallback handled inside create_reader)
  - Pushes FramePackets into one shared frame queue consumed by CentralPipeline

Service-specific behaviour lives in:
  - core/settings.py  (SERVICE_TYPES + per-service knobs)
  - central_pipeline.py (CentralPipeline class + parse_camera_params(cam) hook)
"""
import json
import logging
import os
import queue
import signal
import threading
import time
from typing import Dict

import psutil
import requests

from core.settings import settings
from api_clients.api_clients import get_cameras, patch_camera, get_rules
from camera_reader import BaseCameraReader, create_reader
from central_pipeline import CentralPipeline, parse_camera_params

LOGGER = logging.getLogger("engine")


def _motion_param(name: str, default, cast):
    """Đọc tham số motion theo thứ tự: settings của service -> env -> default."""
    val = getattr(settings, name, None)
    if val is None:
        val = os.getenv(name, default)
    try:
        return cast(val)
    except (TypeError, ValueError):
        return cast(default)


class CentralEngine:
    """Orchestrates all camera readers and the central processing pipeline."""

    POLL_INTERVAL = int(getattr(settings, 'POLL_INTERVAL', 20))  # seconds between camera list refreshes
    # Số vòng poll liên tiếp phải thấy camera mất kết nối trước khi hạ trạng thái
    # offline. Lên online thì báo ngay. Mục đích: một lần giật mạng ngắn không làm
    # camera biến mất trên UI.
    OFFLINE_STRIKES_REQUIRED = int(os.getenv("OFFLINE_STRIKES_REQUIRED", "2"))

    # Shared frame queue: nếu FRAME_QUEUE_SIZE được set thì dùng cố định, ngược
    # lại tự co giãn theo số camera (FRAME_QUEUE_PER_CAM slot mỗi camera) trong
    # khoảng [MIN, MAX]. Queue quá nhỏ -> camera nhanh đè frame của camera chậm;
    # quá lớn -> tốn RAM và xử lý frame đã cũ. Mỗi slot ~2.8MB ở 1280x720 BGR.
    FRAME_QUEUE_SIZE = int(os.getenv("FRAME_QUEUE_SIZE", "0"))
    FRAME_QUEUE_PER_CAM = int(os.getenv("FRAME_QUEUE_PER_CAM", "8"))
    FRAME_QUEUE_MIN = int(os.getenv("FRAME_QUEUE_MIN", "128"))
    FRAME_QUEUE_MAX = int(os.getenv("FRAME_QUEUE_MAX", "512"))

    def __init__(self, gpuid: int = 0):
        self.gpuid = gpuid
        self._shutdown_done = False
        # Khi True, mọi thay đổi trạng thái online bị chặn: tắt service phải GIỮ
        # NGUYÊN trạng thái CMS đang có, không được đánh offline toàn bộ camera.
        self._shutting_down = False
        self.shutdown = threading.Event()

        # Shared frame queue for ALL camera readers
        self.frame_queue: queue.Queue = queue.Queue(
            maxsize=self.FRAME_QUEUE_SIZE or self.FRAME_QUEUE_MIN)
        self._last_dropped_total = 0

        # Active readers {cam_id: BaseCameraReader}
        self.readers: Dict[str, BaseCameraReader] = {}
        self.camera_configs: Dict[str, dict] = {}
        # extra_params ĐÃ GỬI cho pipeline lần trước. Phải so với cái này, không
        # so với get_camera_params(): mỗi pipeline lưu lại một hình dạng khác
        # (fire lưu {"zone": None} trong khi parse trả {}), nên so chéo thì lần
        # nào cũng "khác" và toàn bộ reader bị dựng lại mỗi vòng poll.
        self.camera_extra: Dict[str, dict] = {}
        self.camera_online_status: Dict[str, int] = {}
        self.camera_offline_strikes: Dict[str, int] = {}

        # Central pipeline
        self.pipeline = CentralPipeline(self.frame_queue)

        # Signal handlers
        signal.signal(signal.SIGINT, self._signal)
        signal.signal(signal.SIGTERM, self._signal)

        LOGGER.info(f"CentralEngine init — GPU_ID={gpuid} "
                    f"service_types={settings.SERVICE_TYPES}")
        LOGGER.info(f"[Config] CMS Backend URL: {settings.BASE_URL or '(CHƯA CẤU HÌNH)'}")
        if settings.SESSION_KEY:
            masked_key = (settings.SESSION_KEY[:4] + "****" + settings.SESSION_KEY[-4:]) if len(settings.SESSION_KEY) > 8 else "****"
            LOGGER.info(f"[Config] CMS Session Key: {masked_key} (Đã thiết lập)")
        else:
            LOGGER.warning("[Config] CMS Session Key: CHƯA CẤU HÌNH (trống)")

    def _signal(self, signum, frame):
        LOGGER.info(f"Signal {signum} received, shutting down …")
        # Đặt TRƯỚC shutdown.set() để không có khoảng trống nào cho poll loop
        # kịp patch trạng thái trong lúc đang tắt.
        self._shutting_down = True
        self.shutdown.set()

    # ── main loop ─────────────────────────────────────────────────────
    def run(self):
        """Main entry point — runs until shutdown."""
        self.pipeline.start()

        while not self.shutdown.is_set():
            try:
                self._sync_cameras()
            except Exception as exc:
                LOGGER.error(f"Camera sync error: {exc}")

            self._log_status()

            for _ in range(self.POLL_INTERVAL):
                if self.shutdown.is_set():
                    break
                time.sleep(1)

        self._shutdown_all()

    # ── camera sync ───────────────────────────────────────────────────
    def _sync_cameras(self):
        """Fetch camera list & rules from API and start/stop readers."""
        cameras = self._fetch_cameras()
        if not cameras:
            return

        rules = self._fetch_rules()

        self._resize_frame_queue(len(cameras))
        desired_ids = set()

        for cam in cameras:
            # Một camera có parameter lỗi (JSON rác, zone thiếu key…) không được
            # phép làm hỏng cả vòng sync và treo toàn bộ camera còn lại.
            try:
                self._sync_one_camera(cam, desired_ids, rules=rules)
            except Exception as exc:
                LOGGER.error(f"[{(cam.get('name') or str(cam.get('id')))[:24]}] "
                             f"Sync camera lỗi, bỏ qua vòng này: {exc}")

        self._reap_and_report(desired_ids)

    def _sync_one_camera(self, cam: dict, desired_ids: set, rules: list = None):
        """Đồng bộ MỘT camera: start/stop/restart reader theo config mới nhất.

        Ném exception thì chỉ camera này bị bỏ qua trong vòng poll hiện tại.
        """
        cam_id = cam.get('id')
        cam_name = cam.get('name') or cam_id
        status = cam.get('status', 1)  # Default to 1 (active) if status field is missing
        svc_types = cam.get('service_type', [])

        if isinstance(svc_types, str):
            svc_types = [t.strip() for t in svc_types.split(",") if t.strip()]
        if not isinstance(svc_types, list):
            svc_types = []

        # Keep camera if ANY of its service_types matches engine's filter
        if not set(svc_types) & set(settings.SERVICE_TYPES):
            return
        if not cam_id:
            return

        use_sdk = cam.get('use_sdk', 0)
        rtsp_url = cam.get('url', '')
        if not rtsp_url and use_sdk != 1:
            LOGGER.warning(f"[{cam_name}] Skipped: no RTSP url provided and use_sdk != 1")
            return

        # Check if camera is explicitly disabled (status is 0 or False)
        if status in (0, '0', False, 'false'):
            if cam_id in self.readers:
                LOGGER.info(f"[{cam_name}] Camera disabled (status={status}), stopping reader")
                self._stop_reader(cam_id, patch_offline=True)
            else:
                LOGGER.info(f"[{cam_name}] Skipped: camera status is disabled (status={status})")
            return

        cam_config = {
            'name': cam.get('name') or cam_id,
            'use_sdk': use_sdk,
            'manufacturer': cam.get('manufacturer', ''),
            'url': rtsp_url,
            'storage_url': cam.get('storage_url'),
            'storage_port': cam.get('storage_port'),
            'storage_username': cam.get('storage_username'),
            'storage_password': cam.get('storage_password'),
            'storage_channel': cam.get('storage_channel'),
            # Ưu tiên settings của service (mỗi service tune riêng, ví dụ
            # LITTERING dùng alpha 0.03) rồi mới tới env, thay vì đọc thẳng
            # os.getenv và ghi đè mất giá trị trong core/settings.py.
            'motion_threshold': _motion_param('MOTION_THRESHOLD', 12.0, float),
            'motion_alpha': _motion_param('MOTION_ALPHA', 0.005, float),
            'min_motion_pixels': _motion_param('MIN_MOTION_PIXELS', 400, int),
        }

        # Service-specific params (zone/direction/rules …) parsed by pipeline.
        # Return {} = no extra params; return None = SKIP this camera
        # (e.g. vehicle_counting camera without an active line rule, or camera outside night schedule).
        try:
            extra_params = parse_camera_params(cam, rules=rules)
        except TypeError:
            extra_params = parse_camera_params(cam)
        if extra_params is None:
            return

        desired_ids.add(cam_id)

        if cam_id in self.readers:
            reader = self.readers[cam_id]
            old_config = self.camera_configs.get(cam_id, {})
            # CHỈ những thứ này đổi mới cần dựng lại luồng video.
            stream_changed = any(
                cam_config.get(key) != old_config.get(key)
                for key in ['use_sdk', 'manufacturer', 'url', 'storage_url', 'storage_port',
                            'storage_username', 'storage_password', 'storage_channel', 'name']
            )
            # Tham số nghiệp vụ (ROI, ngưỡng…) đổi thì chỉ cần nạp lại vào
            # pipeline. Dựng lại reader là mất luôn kết nối RTSP/SDK, mất nền
            # đã học và mất tuổi đối tượng đang theo dõi — không cần thiết.
            extra_changed = extra_params != self.camera_extra.get(cam_id)

            if stream_changed:
                LOGGER.info(f"[{reader.cam_name}] Stream config đổi, dựng lại reader")
                self._stop_reader(cam_id, patch_offline=False)
                self._start_reader(cam_id, cam_config, extra_params)
                self.camera_configs[cam_id] = cam_config
            elif not reader.alive():
                LOGGER.warning(f"[{reader.cam_name}] Reader dead, restarting")
                self._stop_reader(cam_id, patch_offline=False)
                self._start_reader(cam_id, cam_config, extra_params)
                self.camera_configs[cam_id] = cam_config
            elif extra_changed:
                LOGGER.info(f"[{reader.cam_name}] Tham số nghiệp vụ đổi, "
                            f"nạp lại pipeline (giữ nguyên luồng video)")
                try:
                    self.pipeline.register_camera(
                        cam_id, cam_config.get('name') or cam_id, **(extra_params or {}))
                    self.camera_extra[cam_id] = extra_params
                except Exception as exc:
                    LOGGER.error(f"[{reader.cam_name}] Nạp lại tham số thất bại: {exc}")
        else:
            self._start_reader(cam_id, cam_config, extra_params)
            self.camera_configs[cam_id] = cam_config

    def _reap_and_report(self, desired_ids: set):
        """Dừng reader không còn thuộc service này, rồi cập nhật trạng thái online."""

        # Stop readers for cameras no longer active for this engine's service types
        for cam_id in list(self.readers.keys()):
            if cam_id not in desired_ids:
                reader = self.readers.get(cam_id)
                name = reader.cam_name if reader else cam_id[:8]
                LOGGER.info(f"[{name}] Camera no longer active for "
                            f"{settings.SERVICE_TYPES}, stopping reader")
                self._stop_reader(cam_id, patch_offline=False)

        # Update online status in API only if it changed
        for cam_id, reader in self.readers.items():
            is_connected = getattr(reader, 'is_connected', False)
            has_frames = getattr(reader, 'frame_count', 0) > 0
            current_status = 1 if (reader.alive() and (is_connected or has_frames)) else 0

            if current_status == 1:
                # Online: báo ngay, không debounce — camera sống lại phải hiện
                # ngay trên UI.
                self.camera_offline_strikes.pop(cam_id, None)
                self._set_camera_online(cam_id, 1)
            else:
                # Offline: phải mất kết nối liên tục qua nhiều vòng poll mới hạ
                # trạng thái, để một lần giật mạng ngắn (reader tự reconnect sau
                # 2s) không làm camera biến mất trên UI.
                strikes = self.camera_offline_strikes.get(cam_id, 0) + 1
                self.camera_offline_strikes[cam_id] = strikes
                if strikes >= self.OFFLINE_STRIKES_REQUIRED:
                    self._set_camera_online(
                        cam_id, 0, reason=f"({strikes} vòng poll mất kết nối)")

    def _set_camera_online(self, cam_id: str, online: int, reason: str = ""):
        """Điểm DUY NHẤT được phép đổi trạng thái online của camera trên CMS.

        - Không patch khi service đang tắt: giữ nguyên trạng thái CMS đang có, để
          Ctrl+C / SIGTERM / restart không làm camera biến mất trên UI.
        - Không patch lại giá trị đã đúng: tránh spam CMS và tránh nháy trên UI.
        - Patch thất bại thì KHÔNG ghi cache, để vòng poll sau thử lại.
        """
        if self._shutting_down:
            LOGGER.info(f"[{cam_id[:8]}] Đang tắt service — giữ nguyên trạng thái "
                        f"CMS (bỏ qua patch online={online})")
            return
        if self.camera_online_status.get(cam_id) == online:
            return
        try:
            patch_camera(cam_id, online)
            self.camera_online_status[cam_id] = online
            LOGGER.info(f"[{cam_id[:8]}] Camera online={online} {reason}".rstrip())
        except Exception as e:
            LOGGER.error(f"Failed to patch camera status for {cam_id[:8]}: {e}")


    def _start_reader(self, cam_id: str, cam_config: dict, extra_params: dict = None):
        try:
            reader = create_reader(
                cam_id, cam_config, self.frame_queue,
                self.shutdown, self.gpuid)
            reader.start()
            self.readers[cam_id] = reader
            cam_name = cam_config.get('name') or cam_id
            self.pipeline.register_camera(cam_id, cam_name, **(extra_params or {}))
            self.camera_extra[cam_id] = extra_params
            LOGGER.info(f"[{cam_name}] Started {type(reader).__name__}")
        except Exception as exc:
            cam_name = cam_config.get('name') or cam_id[:8]
            LOGGER.error(f"[{cam_name}] Start failed: {exc}")
            self._set_camera_online(cam_id, 0, reason="(reader start failed)")

    def _stop_reader(self, cam_id: str, patch_offline: bool = False):
        self.camera_configs.pop(cam_id, None)
        self.camera_extra.pop(cam_id, None)
        reader = self.readers.pop(cam_id, None)
        name = reader.cam_name if reader else cam_id[:8]
        if reader:
            reader.stop()
        self.pipeline.unregister_camera(cam_id)
        if patch_offline:
            # Camera bị tắt có chủ đích trên CMS (status=0) → đánh offline.
            self._set_camera_online(cam_id, 0, reason="(camera disabled on CMS)")
        # Xóa cache SAU khi patch, để _set_camera_online còn so sánh được.
        self.camera_online_status.pop(cam_id, None)
        self.camera_offline_strikes.pop(cam_id, None)
        LOGGER.info(f"[{name}] Stopped")

    # ── helpers ───────────────────────────────────────────────────────
    def _resize_frame_queue(self, n_cameras: int):
        """Co giãn shared queue theo số camera (bỏ qua nếu FRAME_QUEUE_SIZE đã set).

        queue.Queue.maxsize chỉ được đọc trong put()/get() nên đổi nóng là an toàn:
        không mất phần tử nào đang nằm trong queue.
        """
        if self.FRAME_QUEUE_SIZE:
            return
        want = max(self.FRAME_QUEUE_MIN,
                   min(self.FRAME_QUEUE_MAX, n_cameras * self.FRAME_QUEUE_PER_CAM))
        if want != self.frame_queue.maxsize:
            LOGGER.info(f"[Engine] Shared frame queue {self.frame_queue.maxsize} -> {want} "
                        f"({n_cameras} camera x {self.FRAME_QUEUE_PER_CAM} slot)")
            self.frame_queue.maxsize = want

    def _fetch_cameras(self) -> list:
        for attempt in range(3):
            try:
                return get_cameras()
            except requests.exceptions.RequestException as exc:
                LOGGER.warning(f"Fetch cameras attempt {attempt + 1} failed: {exc}")
                time.sleep(2 ** attempt)
            except Exception as exc:
                LOGGER.error(f"Fetch cameras unexpected error: {exc}")
                break
        return []

    def _fetch_rules(self) -> list:
        rule_type = getattr(settings, 'RULE_TYPE', None)
        for attempt in range(3):
            try:
                return get_rules(rule_type=rule_type)
            except requests.exceptions.RequestException as exc:
                LOGGER.warning(f"Fetch rules attempt {attempt + 1} failed: {exc}")
                time.sleep(2 ** attempt)
            except Exception as exc:
                LOGGER.error(f"Fetch rules unexpected error: {exc}")
                break
        return []

    def _log_status(self):
        alive = sum(1 for r in self.readers.values() if r.alive())
        mem = psutil.virtual_memory()
        proc = psutil.Process()
        proc_ram = proc.memory_info().rss / 1024 / 1024

        dropped_total = sum(getattr(r, 'dropped_frames', 0) for r in self.readers.values())
        dropped_delta = dropped_total - self._last_dropped_total
        self._last_dropped_total = dropped_total

        qsize = self.frame_queue.qsize()
        LOGGER.info(
            f"[Engine] [HEARTBEAT] cameras={len(self.readers)} alive={alive} "
            f"frame_q={qsize}/{self.frame_queue.maxsize} drop+{dropped_delta} "
            f"proc_RAM={proc_ram:.0f}MB sys_RAM={mem.percent:.1f}%")

        # Cảnh báo quá tải: queue gần đầy VÀ đang mất frame liên tục nghĩa là
        # pipeline không tiêu thụ kịp -> giảm TARGET_FPS hoặc bớt camera.
        if dropped_delta > 0 and qsize >= 0.9 * self.frame_queue.maxsize:
            LOGGER.warning(
                f"[Engine] QUÁ TẢI: mất {dropped_delta} frame trong ~{self.POLL_INTERVAL}s "
                f"(queue {qsize}/{self.frame_queue.maxsize}). Hạ TARGET_FPS, tăng "
                f"DETECT_INTERVAL hoặc giảm số camera trên tiến trình này.")

    def _shutdown_all(self):
        if self._shutdown_done:
            return
        self._shutdown_done = True
        # Chốt chặn kể cả khi vào đây không qua _signal (ví dụ run() thoát do lỗi).
        self._shutting_down = True
        LOGGER.info("Shutting down all cameras … (giữ nguyên trạng thái online trên CMS)")
        for cam_id in list(self.readers.keys()):
            self._stop_reader(cam_id)
        self.pipeline.stop()
        LOGGER.info("Engine shutdown complete")


def main():
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    gpuid = int(os.getenv("GPU_ID", "0"))
    engine = CentralEngine(gpuid=gpuid)
    try:
        engine.run()
    except KeyboardInterrupt:
        LOGGER.info("Keyboard interrupt")
    finally:
        engine._shutdown_all()


if __name__ == "__main__":
    main()
