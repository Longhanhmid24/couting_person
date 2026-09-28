"""
person_process.py — Detect người + tracking + đếm vượt vạch. Một instance mỗi camera.
Được kế thừa và tối ưu hóa từ thuật toán đếm xe, chuyên biệt cho người đi bộ:
  - Bám vết bằng IoU + DIoU và vòng ghép khoảng cách tới vị trí dự đoán.
  - Sử dụng vị trí đáy bbox (chân người) và tâm bbox để kiểm tra cắt vạch chính xác.
  - Nội suy chuyển động giữa các frame YOLO bằng vận tốc và HOG descriptor.
  - Bộ lọc chống rung theo tổng dịch chuyển quỹ đạo (MIN_PATH_MOVEMENT_PIXELS).
  - Không bao giờ đếm trùng (mỗi track ID đếm đúng 1 lần).
"""
import concurrent.futures
import logging
import queue
import time
from collections import deque

import cv2
import numpy as np

from core.settings import settings
from grpc_clients.grpc_clients import GRPCClient
from utils.line_crossing import (
    get_bbox_center,
    get_bbox_bottom_center,
    parse_line_points,
    path_crosses_line,
    path_net_displacement,
    segments_intersect,
)
from utils.helper import normalize_person_class, save_counted_person_images, cleanup_old_images

logger = logging.getLogger(__name__)

# Sentinel để pipeline yêu cầu thoát ngay mà không phải chờ hết timeout queue.
_EOS = object()

# HOG descriptor singleton — dùng chung cho mọi object nếu cv2 hỗ trợ
_hog = None
if hasattr(cv2, "HOGDescriptor"):
    try:
        _hog = cv2.HOGDescriptor(
            _winSize=(64, 64),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9,
        )
    except Exception:
        _hog = None



def compute_iou(box1, box2):
    """IoU giữa hai bbox (x, y, w, h), kết hợp DIoU phạt khoảng cách tâm."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    inter_w = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    inter_h = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    intersection = inter_w * inter_h

    area1, area2 = w1 * h1, w2 * h2
    if area1 <= 0 or area2 <= 0:
        return 0.0

    union = area1 + area2 - intersection
    iou = intersection / union if union > 0 else 0.0

    min_area = min(area1, area2)
    containment = (intersection / min_area) if min_area > 0 else 0.0

    # DIoU: phạt theo khoảng cách tâm đã chuẩn hoá
    c1x, c1y = x1 + w1 / 2.0, y1 + h1 / 2.0
    c2x, c2y = x2 + w2 / 2.0, y2 + h2 / 2.0
    enc_w = max(x1 + w1, x2 + w2) - min(x1, x2)
    enc_h = max(y1 + h1, y2 + h2) - min(y1, y2)
    diag_sq = enc_w ** 2 + enc_h ** 2
    diou = iou - (((c1x - c2x) ** 2 + (c1y - c2y) ** 2) / diag_sq if diag_sq > 0 else 0.0)

    area_ratio = max(area1, area2) / max(1.0, float(min_area))
    containment_score = min(containment * 0.45, 0.45) if area_ratio <= 2.5 else 0.0
    return max(iou, containment_score, diou)


def search_by_hog(obj, frame):
    """Nội suy vị trí giữa hai vòng YOLO: vận tốc trước, HOG có giới hạn cửa sổ."""
    x, y, w, h = obj.bbox
    vx, vy = obj.velocity
    frame_h, frame_w = frame.shape[:2]

    pred_x = max(0, min(frame_w - w, int(round(x + vx))))
    pred_y = max(0, min(frame_h - h, int(round(y + vy))))

    obj._heavy_hog_counter += 1
    heavy_every = max(1, int(settings.HOG_HEAVY_EVERY))
    if (obj.hog_descriptor is None or w < 15 or h < 15
            or obj._heavy_hog_counter % heavy_every != 1
            or abs(vx) + abs(vy) < max(w, h) * 0.05):
        obj.update_position((pred_x, pred_y, w, h))
        return

    margin_x = max(int(w * 0.45), 15)
    margin_y = max(int(h * 0.45), 15)
    sx1 = max(0, pred_x - margin_x)
    sy1 = max(0, pred_y - margin_y)
    sx2 = min(frame_w, pred_x + w + margin_x)
    sy2 = min(frame_h, pred_y + h + margin_y)
    step = max(6, min(w, h) // 3)

    positions = [(px, py)
                 for py in range(sy1, max(sy1 + 1, sy2 - h + 1), step)
                 for px in range(sx1, max(sx1 + 1, sx2 - w + 1), step)]
    max_windows = max(1, int(settings.HOG_MAX_WINDOWS))
    if len(positions) > max_windows:
        positions = positions[::-(-len(positions) // max_windows)]
    if (pred_x, pred_y) not in positions:
        positions.insert(0, (pred_x, pred_y))

    reference = obj.hog_descriptor
    ref_norm = float(np.linalg.norm(reference))
    if ref_norm <= 0:
        obj.update_position((pred_x, pred_y, w, h))
        return

    best_sim, best_pos, pred_sim = -1.0, (pred_x, pred_y), -1.0
    try:
        for px, py in positions:
            crop = frame[py:py + h, px:px + w]
            if crop.shape[0] < 10 or crop.shape[1] < 10:
                continue
            gray = cv2.cvtColor(
                cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2GRAY)
            desc = _hog.compute(gray).flatten()
            desc_norm = float(np.linalg.norm(desc))
            if desc_norm <= 0:
                continue
            sim = float(np.dot(reference, desc)) / (ref_norm * desc_norm)
            if (px, py) == (pred_x, pred_y):
                pred_sim = sim
            if sim > best_sim:
                best_sim, best_pos = sim, (px, py)
    except Exception as exc:
        logger.debug(f"[HOG] search lỗi cho ID#{obj.id}: {exc}")
        obj.update_position((pred_x, pred_y, w, h))
        return

    # Chỉ rời vị trí dự đoán khi HOG thắng rõ rệt
    if (best_sim >= float(settings.HOG_MIN_SIMILARITY)
            and best_pos != (pred_x, pred_y)
            and best_sim >= pred_sim + float(settings.HOG_GAIN_MARGIN)):
        obj.update_position((best_pos[0], best_pos[1], w, h))
        obj.compute_hog_descriptor(frame)
    else:
        obj.update_position((pred_x, pred_y, w, h))


class CountingTrackedPerson:
    """Đối tượng người đang được bám vết (tracking), lưu toàn bộ quỹ đạo."""

    def __init__(self, obj_id, bbox, frame, conf=None, cls=None):
        self.id = obj_id
        self.bbox = tuple(int(v) for v in bbox)
        self.conf = conf
        self.cls = cls
        self.lost = 0
        self.frame_count = 1
        self.velocity = (0.0, 0.0)
        self.last_update_frame = 0
        self.created_time = time.time()
        self.counted = False

        history = max(3, int(settings.PATH_HISTORY_LEN))
        self.path_center = deque(maxlen=history)
        self.path_bottom = deque(maxlen=history)
        self.path_center.append(get_bbox_center(self.bbox))
        self.path_bottom.append(get_bbox_bottom_center(self.bbox))

        self.hog_descriptor = None
        self._heavy_hog_counter = 0
        self.compute_hog_descriptor(frame)

    @property
    def curr_center(self):
        return self.path_center[-1] if self.path_center else None

    @property
    def prev_center(self):
        return self.path_center[-2] if len(self.path_center) >= 2 else None

    @property
    def curr_bottom(self):
        return self.path_bottom[-1] if self.path_bottom else None

    @property
    def prev_bottom(self):
        return self.path_bottom[-2] if len(self.path_bottom) >= 2 else None

    def update_position(self, bbox):
        self.bbox = tuple(int(v) for v in bbox)
        self.path_center.append(get_bbox_center(self.bbox))
        self.path_bottom.append(get_bbox_bottom_center(self.bbox))

    def predict_forward(self, frame_shape):
        vx, vy = self.velocity
        if abs(vx) < 0.01 and abs(vy) < 0.01:
            return
        x, y, w, h = self.bbox
        frame_h, frame_w = frame_shape[:2]
        nx = max(-w + 1, min(frame_w - 1, int(round(x + vx))))
        ny = max(-h + 1, min(frame_h - 1, int(round(y + vy))))
        self.update_position((nx, ny, w, h))

    def predicted_bbox(self, frames_ahead=1.0):
        x, y, w, h = self.bbox
        vx, vy = self.velocity
        return (x + vx * frames_ahead, y + vy * frames_ahead, w, h)

    def compute_hog_descriptor(self, frame):
        if _hog is None:
            return
        x, y, w, h = self.bbox
        frame_h, frame_w = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame_w, x + w), min(frame_h, y + h)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return
        try:
            gray = cv2.cvtColor(
                cv2.resize(frame[y1:y2, x1:x2], (64, 64), interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2GRAY)
            self.hog_descriptor = _hog.compute(gray).flatten()
        except Exception:
            pass

    def get_class(self):
        return "person"

    def path_movement(self):
        """Tổng dịch chuyển thực tế của cả quỹ đạo (chống rung lắc camera)."""
        return max(path_net_displacement(self.path_bottom),
                   path_net_displacement(self.path_center))


class CountedPersonSnapshot:
    """Bản chụp bất biến dữ liệu đối tượng vượt vạch để xử lý và lưu trữ thread-safe trong image_writer_pool."""
    def __init__(self, obj):
        self.id = int(obj.id)
        self.bbox = tuple(int(v) for v in obj.bbox)
        self.conf = float(obj.conf) if getattr(obj, 'conf', None) is not None else 0.8
        self.cls = getattr(obj, 'cls', None)
        self.path_bottom = [tuple(float(c) for c in p) for p in getattr(obj, 'path_bottom', []) if p is not None]
        self.path_center = [tuple(float(c) for c in p) for p in getattr(obj, 'path_center', []) if p is not None]


class PersonProcessor:
    """Detect người + tracking + đếm vượt vạch. Một instance mỗi camera."""

    def __init__(self, stream_id, frame_queue, counting_callback, shutdown_flag,
                 line_config=None, direction_config=None, cam_name=None):
        self.stream_id = stream_id
        self.cam_name = cam_name or stream_id
        self.client = GRPCClient()
        self.frame_queue = frame_queue
        self.counting_callback = counting_callback  # callable(class_name: str)
        self.shutdown_flag = shutdown_flag
        self.line_config = line_config
        self.direction_config = direction_config

        self._line_points = None
        self._line_resolved = False
        self._direction_vector = None
        self._direction_resolved = False

        detect_fps = float(getattr(settings, "DETECT_FPS", 0) or 0)
        self._detect_interval = (1.0 / detect_fps) if detect_fps > 0 else 0.0
        self._next_detect_at = 0.0

        self._object_id_counter = 0
        self.total_counted = 0
        self._stats = {
            "frames": 0, "frames_dropped": 0, "detect_rounds": 0, "detections": 0,
            "matched_iou": 0, "matched_distance": 0, "tracks_created": 0,
            "expired": 0, "counted": 0, "gate_frames": 0, "gate_movement": 0,
            "grpc_errors": 0,
        }
        self._stats_at = time.time()

        # Thread pool ghi file ảnh bất đồng bộ ra thư mục chung
        workers = max(1, int(getattr(settings, "IMAGE_WRITER_WORKERS", 2)))
        self.image_writer_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix=f"img-writer-{self.cam_name}"
        )
        self._last_cleanup_at = time.time()

    def detect_person(self, frame):
        return self.client.detect_person_yolo(frame)

    def _resolve_line(self, frame):
        if self._line_resolved:
            return
        self._line_resolved = True
        h, w = frame.shape[:2]
        self._line_points = parse_line_points(self.line_config, w, h)
        if self._line_points:
            logger.info(f"[{self.cam_name}] Vạch đếm người (W={w}, H={h}): {self._line_points}")
        else:
            logger.warning(f"[{self.cam_name}] KHÔNG phân giải được vạch đếm từ "
                           f"config: {self.line_config} — sẽ không đếm được người")
        self._resolve_direction(frame)

    def _resolve_direction(self, frame):
        """Chuyển direction_config (tọa độ chuẩn hóa 0-1) thành vector hướng."""
        if self._direction_resolved:
            return
        self._direction_resolved = True
        cfg = self.direction_config
        if not cfg or not isinstance(cfg, (list, tuple)) or len(cfg) < 2:
            logger.warning(f"[{self.cam_name}] Không có direction_config — không phân biệt được IN/OUT")
            return
        try:
            d1, d2 = cfg[0], cfg[1]
            vx = float(d2[0]) - float(d1[0])
            vy = float(d2[1]) - float(d1[1])
            if abs(vx) < 1e-9 and abs(vy) < 1e-9:
                logger.warning(f"[{self.cam_name}] Direction vector = (0,0) — 2 điểm trùng nhau")
                return
            self._direction_vector = (vx, vy)
            logger.info(f"[{self.cam_name}] Direction vector: ({vx:.4f}, {vy:.4f}) "
                        f"[d1={d1} → d2={d2}]")
        except Exception as exc:
            logger.warning(f"[{self.cam_name}] Lỗi parse direction_config: {exc}")

    def _next_frame(self):
        try:
            item = self.frame_queue.get(timeout=1.0)
        except queue.Empty:
            return None
        if item is None:
            return _EOS

        backlog = max(1, int(settings.FRAME_BACKLOG_MAX))
        while self.frame_queue.qsize() > backlog:
            try:
                nxt = self.frame_queue.get_nowait()
            except queue.Empty:
                break
            if nxt is None:
                return _EOS
            item = nxt
            self._stats["frames_dropped"] += 1

        self._stats["frames"] += 1
        return item

    def process_person(self):
        """Vòng lặp chính: Frame → Detect/Track → Xét vượt vạch → Báo cáo đếm."""
        tracked_objects = []
        while not self.shutdown_flag.is_set():
            item = self._next_frame()
            if item is None:
                continue
            if item is _EOS:
                break

            _frame_time, frame_uuid, frame, frame_count = item
            if frame is None:
                continue
            self._resolve_line(frame)

            now = time.time()
            # Định kỳ kích hoạt dọn dẹp ảnh cũ (mỗi 24 giờ một lần)
            if now - self._last_cleanup_at > 86400:
                self._last_cleanup_at = now
                try:
                    self.image_writer_pool.submit(cleanup_old_images)
                except Exception:
                    pass

            if self._detect_interval > 0:
                do_detect = now >= self._next_detect_at
                if do_detect:
                    self._next_detect_at = now + self._detect_interval
            else:
                skip = max(1, int(settings.SKIP_FRAME))
                do_detect = skip <= 1 or (frame_count % skip == 1)

            if do_detect:
                tracked_objects = self._detect_round(tracked_objects, frame, frame_count, frame_uuid=frame_uuid)
            else:
                tracked_objects = self._track_round(tracked_objects, frame, frame_uuid=frame_uuid)

            self._log_stats_if_due(len(tracked_objects))

        logger.info(f"[{self.cam_name}] PersonProcessor dừng — "
                    f"{self.total_counted} người đã đếm, "
                    f"{len(tracked_objects)} track còn trên tay")

    def _detect_round(self, tracked_objects, frame, frame_count, frame_uuid=None):
        try:
            dets = self.detect_person(frame)
        except Exception as exc:
            self._stats["grpc_errors"] += 1
            logger.error(f"[{self.cam_name}] lỗi detect người: {exc}")
            time.sleep(0.05)
            return tracked_objects

        self._stats["detect_rounds"] += 1
        min_conf = float(settings.CONFIDENT_PERSON)
        detections = []
        for det in dets or []:
            try:
                x1, y1, x2, y2, conf, cls = det
            except (TypeError, ValueError):
                continue
            if conf is not None and float(conf) < min_conf:
                continue
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if x2 - x1 <= 1 or y2 - y1 <= 1:
                continue
            detections.append(((x1, y1, x2 - x1, y2 - y1), float(conf or 0.0), cls))
        self._stats["detections"] += len(detections)

        matched_dets = set()
        matched_objs = set()
        self._match_by_iou(tracked_objects, detections, frame, frame_count,
                           matched_dets, matched_objs, frame_uuid=frame_uuid)
        if settings.MATCH_BY_DISTANCE:
            self._match_by_distance(tracked_objects, detections, frame, frame_count,
                                    matched_dets, matched_objs, frame_uuid=frame_uuid)

        survivors = [obj for obj in tracked_objects if id(obj) in matched_objs]

        max_lost = max(1, int(settings.MAX_LOST_ROUNDS))
        for obj in tracked_objects:
            if id(obj) in matched_objs:
                continue
            obj.lost += 1
            if obj.lost >= max_lost:
                self._stats["expired"] += 1
                continue
            obj.predict_forward(frame.shape)
            self._check_line_crossing(obj, frame=frame, frame_uuid=frame_uuid)
            survivors.append(obj)

        for det_idx, (det_bbox, conf, cls) in enumerate(detections):
            if det_idx in matched_dets:
                continue
            obj = CountingTrackedPerson(self._object_id_counter, det_bbox, frame, conf, cls)
            obj.last_update_frame = frame_count
            self._object_id_counter += 1
            self._stats["tracks_created"] += 1
            survivors.append(obj)

        return survivors

    def _gates_ok(self, obj, det_bbox):
        area_obj = max(1, obj.bbox[2] * obj.bbox[3])
        area_det = max(1, det_bbox[2] * det_bbox[3])
        ratio = max(area_obj, area_det) / float(min(area_obj, area_det))
        return ratio <= float(settings.MATCH_MAX_AREA_RATIO)

    def _match_by_iou(self, tracked_objects, detections, frame, frame_count,
                      matched_dets, matched_objs, frame_uuid=None):
        threshold = float(settings.MATCH_IOU_THRESHOLD)
        pairs = []
        for det_idx, (det_bbox, _conf, _cls) in enumerate(detections):
            for obj in tracked_objects:
                if not self._gates_ok(obj, det_bbox):
                    continue
                score = compute_iou(obj.bbox, det_bbox)
                if score >= threshold:
                    pairs.append((score, det_idx, obj))
        pairs.sort(key=lambda p: p[0], reverse=True)

        for _score, det_idx, obj in pairs:
            if det_idx in matched_dets or id(obj) in matched_objs:
                continue
            matched_dets.add(det_idx)
            matched_objs.add(id(obj))
            det_bbox, conf, cls = detections[det_idx]
            self._apply_match(obj, det_bbox, conf, cls, frame, frame_count, frame_uuid=frame_uuid)
            self._stats["matched_iou"] += 1

    def _match_by_distance(self, tracked_objects, detections, frame, frame_count,
                           matched_dets, matched_objs, frame_uuid=None):
        factor = float(settings.MATCH_DISTANCE_FACTOR)
        pairs = []
        for det_idx, (det_bbox, _conf, _cls) in enumerate(detections):
            if det_idx in matched_dets:
                continue
            dcx = det_bbox[0] + det_bbox[2] / 2.0
            dcy = det_bbox[1] + det_bbox[3] / 2.0
            for obj in tracked_objects:
                if id(obj) in matched_objs or not self._gates_ok(obj, det_bbox):
                    continue
                px, py, pw, ph = obj.predicted_bbox()
                dist = ((px + pw / 2.0 - dcx) ** 2 + (py + ph / 2.0 - dcy) ** 2) ** 0.5
                if dist <= factor * max(pw, ph, det_bbox[2], det_bbox[3]):
                    pairs.append((dist, det_idx, obj))
        pairs.sort(key=lambda p: p[0])

        for _dist, det_idx, obj in pairs:
            if det_idx in matched_dets or id(obj) in matched_objs:
                continue
            matched_dets.add(det_idx)
            matched_objs.add(id(obj))
            det_bbox, conf, cls = detections[det_idx]
            self._apply_match(obj, det_bbox, conf, cls, frame, frame_count, frame_uuid=frame_uuid)
            self._stats["matched_distance"] += 1

    def _apply_match(self, obj, det_bbox, conf, cls, frame, frame_count, frame_uuid=None):
        last = obj.last_update_frame or (frame_count - 1)
        frames_passed = max(1, frame_count - last)
        dx = (det_bbox[0] - obj.bbox[0]) / float(frames_passed)
        dy = (det_bbox[1] - obj.bbox[1]) / float(frames_passed)
        old_vx, old_vy = obj.velocity
        obj.velocity = (0.5 * old_vx + 0.5 * dx, 0.5 * old_vy + 0.5 * dy)

        obj.last_update_frame = frame_count
        obj.lost = 0
        obj.conf = conf
        obj.cls = cls
        obj.frame_count += 1
        obj.update_position(det_bbox)
        obj.compute_hog_descriptor(frame)
        self._check_line_crossing(obj, frame=frame, frame_uuid=frame_uuid)

    def _track_round(self, tracked_objects, frame, frame_uuid=None):
        for obj in tracked_objects:
            search_by_hog(obj, frame)
            obj.frame_count += 1
            self._check_line_crossing(obj, frame=frame, frame_uuid=frame_uuid)
        return tracked_objects

    def _check_line_crossing(self, obj, frame=None, frame_uuid=None):
        """Đếm người khi quỹ đạo chân hoặc tâm cắt qua vạch.
        Dùng dot product với direction vector để phân biệt IN/OUT.
        """
        if obj.counted or not self._line_points:
            return
        if obj.frame_count < int(settings.MIN_FRAMES_BEFORE_COUNT):
            self._stats["gate_frames"] += 1
            return

        if obj.path_movement() < float(settings.MIN_PATH_MOVEMENT_PIXELS):
            self._stats["gate_movement"] += 1
            return

        if not (path_crosses_line(obj.path_bottom, self._line_points)
                or path_crosses_line(obj.path_center, self._line_points)):
            return

        obj.counted = True
        self._stats["counted"] += 1
        self.total_counted += 1
        conf = obj.conf if obj.conf is not None else 0.0

        # Xác định hướng IN/OUT bằng dot product
        person_class = "person_in"  # mặc định IN
        dot_val = None
        if self._direction_vector is not None:
            # Lấy vector chuyển động: ưu tiên đoạn thực sự cắt qua vạch,
            # fallback sang tổng dịch chuyển của toàn quỹ đạo (last - first)
            move_pts = [p for p in (obj.path_bottom if len(obj.path_bottom) >= 2 else obj.path_center) if p is not None]
            move_x, move_y = 0.0, 0.0
            crossing_vec = None
            if len(self._line_points) >= 2 and len(move_pts) >= 2:
                c, d = self._line_points[0], self._line_points[1]
                for i in range(len(move_pts) - 1):
                    a, b = move_pts[i], move_pts[i + 1]
                    if a is not None and b is not None and segments_intersect(a, b, c, d):
                        crossing_vec = (float(b[0] - a[0]), float(b[1] - a[1]))
                        break

            if crossing_vec is not None and (crossing_vec[0] != 0 or crossing_vec[1] != 0):
                move_x, move_y = crossing_vec
            elif len(move_pts) >= 2:
                move_x = float(move_pts[-1][0] - move_pts[0][0])
                move_y = float(move_pts[-1][1] - move_pts[0][1])

            dir_x, dir_y = self._direction_vector
            dot = move_x * dir_x + move_y * dir_y
            dot_val = dot
            person_class = "person_in" if dot > 0 else "person_out"

        direction_label = "IN" if person_class == "person_in" else "OUT"
        dot_str = f"dot={dot_val:.2f}" if dot_val is not None else "no_dir"
        logger.info(f"[{self.cam_name}] VƯỢT VẠCH {direction_label} ({dot_str}) — người #{obj.id} "
                    f"conf={conf:.2f} bbox={obj.bbox} frames={obj.frame_count}")

        # Kích hoạt lưu ảnh người được đếm vào thư mục dùng chung (bất đồng bộ)
        if getattr(settings, "ENABLE_SAVE_IMAGE", True) and frame is not None:
            try:
                frame_snap = frame.copy()
                line_pts = list(self._line_points) if self._line_points else None
                dir_vec = tuple(self._direction_vector) if self._direction_vector else None
                obj_snap = CountedPersonSnapshot(obj)
                self.image_writer_pool.submit(
                    save_counted_person_images,
                    frame=frame_snap,
                    obj=obj_snap,
                    line_points=line_pts,
                    direction_vector=dir_vec,
                    direction_label=direction_label,
                    cam_id=self.stream_id,
                    frame_uuid=frame_uuid,
                    cam_name=self.cam_name,
                )
            except Exception as e:
                logger.warning(f"[{self.cam_name}] Không thể submit tác vụ lưu ảnh: {e}")

        if self.counting_callback:
            try:
                self.counting_callback(person_class)
            except Exception as exc:
                logger.error(f"[{self.cam_name}] callback đếm lỗi: {exc}")

    def _log_stats_if_due(self, active_tracks):
        interval = float(settings.STATS_LOG_INTERVAL)
        if interval <= 0:
            return
        now = time.time()
        elapsed = now - self._stats_at
        if elapsed < interval:
            return

        s = self._stats
        logger.info(
            f"[{self.cam_name}] {elapsed:.0f}s: frame={s['frames']} "
            f"(bỏ {s['frames_dropped']}) detect={s['detect_rounds']} "
            f"det={s['detections']} ghép_iou={s['matched_iou']} "
            f"ghép_kc={s['matched_distance']} track_mới={s['tracks_created']} "
            f"hết_hạn={s['expired']} ĐẾM_NGƯỜI={s['counted']} "
            f"chờ_frame={s['gate_frames']} chờ_dịch_chuyển={s['gate_movement']} "
            f"grpc_lỗi={s['grpc_errors']} track_sống={active_tracks} "
            f"tổng_đếm={self.total_counted}")
        for key in s:
            s[key] = 0
        self._stats_at = now

    def stop(self):
        """Dừng image_writer_pool an toàn."""
        try:
            self.image_writer_pool.shutdown(wait=False)
        except Exception:
            pass


# Alias để tương thích
VP_Person = PersonProcessor
VP_VHS = PersonProcessor
