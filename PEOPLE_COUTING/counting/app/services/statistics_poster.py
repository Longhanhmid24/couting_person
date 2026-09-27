"""
statistics_poster.py — Gom số đếm người và POST định kỳ lên CMS.
"""
import logging
import threading
import time

from core.settings import settings
from api_clients.api_clients import post_statistics

logger = logging.getLogger(__name__)


class PersonCounter:
    """Bộ đếm người theo camera, thread-safe. Không bao giờ làm mất một lần đếm."""

    def __init__(self, stream_id, cam_name=None):
        self.stream_id = stream_id
        self.cam_name = cam_name or stream_id
        self._lock = threading.Lock()
        self._person_count = 0
        self._person_in = 0
        self._person_out = 0
        self._total_ever = 0

    def increment(self, person_class: str = "person", amount: int = 1):
        """Cộng một lần đếm. person_class = 'person_in' | 'person_out'."""
        with self._lock:
            self._person_count += amount
            self._total_ever += amount
            if person_class == "person_in":
                self._person_in += amount
            elif person_class == "person_out":
                self._person_out += amount

    def snapshot_and_reset(self):
        """Lấy số đếm hiện tại và reset, nguyên tử."""
        with self._lock:
            snapshot = {
                "count": self._person_count,
                "person": self._person_count,
                "person_in": self._person_in,
                "person_out": self._person_out,
            }
            self._person_count = 0
            self._person_in = 0
            self._person_out = 0
            return snapshot

    def restore(self, snapshot: dict):
        """Trả số đếm về bộ đếm khi POST thất bại."""
        with self._lock:
            self._person_count += int(snapshot.get("person", snapshot.get("count", 0)))
            self._person_in += int(snapshot.get("person_in", 0))
            self._person_out += int(snapshot.get("person_out", 0))

    def get_total(self):
        with self._lock:
            return self._total_ever


class StatisticsPoster:
    """Thread nền POST thống kê đếm người lên CMS. Một instance mỗi camera."""

    def __init__(self, stream_id, counter: PersonCounter, shutdown_flag, cam_name=None):
        self.stream_id = stream_id
        self.cam_name = cam_name or stream_id
        self.counter = counter
        self.shutdown_flag = shutdown_flag
        self.post_interval = max(1, int(settings.STATISTICS_POST_INTERVAL))

    def run(self):
        """Vòng lặp POST định kỳ. Thoát ngay khi shutdown_flag được set."""
        logger.info(
            f"[{self.cam_name}] StatisticsPoster bắt đầu "
            f"(chu kỳ {self.post_interval}s, metric_type={settings.METRIC_TYPE})"
        )
        while not self.shutdown_flag.is_set():
            # Event.wait() thay vì sleep() để khi stop() tỉnh ngay lập tức
            if self.shutdown_flag.wait(timeout=self.post_interval):
                break
            self._flush_once()

        # Khi tắt service: flush lần cuối để không mất số đếm dở dang
        logger.info(f"[{self.cam_name}] StatisticsPoster đang flush lần cuối...")
        self._flush_once()
        logger.info(f"[{self.cam_name}] StatisticsPoster dừng")

    def _flush_once(self):
        """Lấy snapshot, POST lên CMS. Thất bại thì restore lại bộ đếm."""
        snapshot = self.counter.snapshot_and_reset()
        total = snapshot.get("count", 0)

        # Ngay cả khi không có người qua (count=0), nhiều CMS vẫn cần heartbeat
        # định kỳ để biết camera còn online.
        in_count = snapshot.get("person_in", 0)
        out_count = snapshot.get("person_out", 0)
        payload_data = {
            "count": total,
            "person": snapshot.get("person", 0),
            "enter": in_count,
            "exit": out_count,
            "person_in": in_count,
            "person_out": out_count,
        }

        try:
            ok = post_statistics(
                stream_id=self.stream_id,
                metric_type=settings.METRIC_TYPE,
                data=payload_data,
                time_point=time.time(),
            )
            if ok:
                if total > 0:
                    logger.info(
                        f"[{self.cam_name}] Đã POST thống kê {settings.METRIC_TYPE}: "
                        f"{total} người (IN={in_count}, OUT={out_count}, "
                        f"tổng từ trước đến nay: {self.counter.get_total()})"
                    )
            else:
                logger.warning(
                    f"[{self.cam_name}] POST thống kê thất bại (API trả False), "
                    f"trả {total} người về bộ đếm"
                )
                self.counter.restore(snapshot)
        except Exception as exc:
            logger.error(
                f"[{self.cam_name}] Lỗi khi POST thống kê: {exc}, "
                f"trả {total} người về bộ đếm"
            )
            self.counter.restore(snapshot)
