"""
person_process.py — Detect người + tracking + đếm vượt vạch. Một instance mỗi camera.
Được kế thừa và tối ưu hóa từ thuật toán đếm xe, chuyên biệt cho người đi bộ:
  - ByteTrack hai tầng confidence với Kalman Filter 8 chiều.
  - Hysteresis FSM quanh vạch, lọc đứng im và cooldown không gian-thời gian.
"""
import concurrent.futures
import logging
import queue
import time
from collections import deque


from core.settings import settings
from grpc_clients.grpc_clients import GRPCClient
from utils.line_crossing import (
    get_bbox_center,
    get_bbox_bottom_center,
    parse_line_points,
    LineZoneCrossingFSM, SpatialCooldownRegistry,
)
from utils.helper import save_counted_person_images, cleanup_old_images
from tracker import BYTETracker, TrackStitcher
from tracker.matching import iou_matrix, linear_assignment

logger = logging.getLogger(__name__)

# Sentinel để pipeline yêu cầu thoát ngay mà không phải chờ hết timeout queue.
_EOS = object()

class CountingTrackedPerson:
    """Đối tượng người đang được bám vết (tracking), lưu toàn bộ quỹ đạo."""

    def __init__(self, obj_id, bbox, frame=None, conf=None, cls=None):
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
        self.crossing_fsm = None

        history = max(3, int(settings.PATH_HISTORY_LEN))
        self.path_center = deque(maxlen=history)
        self.path_bottom = deque(maxlen=history)
        self.path_center.append(get_bbox_center(self.bbox))
        self.path_bottom.append(get_bbox_bottom_center(self.bbox))


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

    def get_class(self):
        return "person"



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
        self.byte_tracker = BYTETracker(
            high_thresh=settings.TRACK_HIGH_THRESH, low_thresh=settings.TRACK_LOW_THRESH,
            new_track_thresh=settings.NEW_TRACK_THRESH, match_thresh=settings.MATCH_THRESH,
            track_buffer=settings.TRACK_BUFFER)
        self.track_stitcher = TrackStitcher() if settings.ENABLE_TRACK_STITCHER else None
        self._person_tracks = {}
        self._cooldown = SpatialCooldownRegistry(settings.COUNT_COOLDOWN_SECONDS)
        self.total_counted = 0
        self._stats = {
            "frames": 0, "frames_dropped": 0, "detect_rounds": 0, "detections": 0,
            "matched_high": 0, "matched_low": 0, "tracks_created": 0,
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

            if getattr(settings, 'ENABLE_LIVE_STREAM', False):
                try:
                    from utils.live_streamer import update_stream_frame
                    update_stream_frame(frame, tracked_objects, self._line_points, self._direction_vector, self.cam_name)
                except Exception:
                    pass

            self._log_stats_if_due(len(tracked_objects))

        logger.info(f"[{self.cam_name}] PersonProcessor dừng — "
                    f"{self.total_counted} người đã đếm, "
                    f"{len(tracked_objects)} track còn trên tay")

    def _detections(self, frame):
        try:
            dets = self.detect_person(frame)
        except Exception as exc:
            self._stats["grpc_errors"] += 1
            logger.error(f"[{self.cam_name}] lỗi detect người: {exc}")
            return None
        self._stats["detect_rounds"] += 1
        detections = []
        for det in dets or []:
            try:
                x1, y1, x2, y2, conf, cls = det
                conf = float(conf or 0.0)
                box = (float(x1), float(y1), float(x2)-float(x1), float(y2)-float(y1))
            except (TypeError, ValueError):
                continue
            if box[2] > 1 and box[3] > 1 and conf >= settings.TRACK_LOW_THRESH:
                detections.append((box, conf, cls))
        self._stats["detections"] += len(detections)
        return self._filter_nested_boxes(detections)

    @staticmethod
    def _filter_nested_boxes(detections, iom_threshold=0.60):
        """Loại bỏ box con bị lồng bên trong box lớn (ví dụ YOLO vừa detect cả người vừa detect nửa thân trên)."""
        if len(detections) <= 1:
            return detections
        # Ưu tiên box có chiều cao/diện tích lớn hơn (full người) và confidence
        sorted_dets = sorted(detections, key=lambda d: (d[0][3] * d[0][2], d[1]), reverse=True)
        kept = []
        for box, conf, cls in sorted_dets:
            bx, by, bw, bh = box
            ba = bw * bh
            duplicate = False
            for k_box, k_conf, k_cls in kept:
                kx, ky, kw, kh = k_box
                ka = kw * kh
                ix1, iy1 = max(bx, kx), max(by, ky)
                ix2, iy2 = min(bx + bw, kx + kw), min(by + bh, ky + kh)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                min_area = min(ba, ka)
                if min_area > 0 and (inter / min_area) > iom_threshold:
                    duplicate = True
                    break
            if not duplicate:
                kept.append((box, conf, cls))
        return kept

    def _process_tracker_results(self, results, expired, frame, frame_count, frame_uuid=None):
        h, w = frame.shape[:2] if frame is not None else (720, 1280)
        margin = int(getattr(settings, "BOUNDARY_MARGIN", 20))
        counted_retire = int(getattr(settings, "COUNTED_RETIRE_FRAMES", 5))

        early_retired = set()
        seen = set()

        for result in results:
            track_id = result.track_id
            box = tuple(int(round(v)) for v in result.bbox)
            obj = self._person_tracks.get(track_id)
            if obj is None and self.track_stitcher:
                obj = self.track_stitcher.recover(track_id, box)
                if obj is not None:
                    self._person_tracks[track_id] = obj
            if obj is None:
                obj = CountingTrackedPerson(track_id, box, conf=result.score, cls=result.cls)
                obj.last_update_frame = frame_count
                self._person_tracks[track_id] = obj
                self._stats["tracks_created"] += 1
            if result.matched:
                last = obj.bbox
                passed = max(1, frame_count - (obj.last_update_frame or frame_count-1))
                vx = (box[0]-last[0])/float(passed); vy = (box[1]-last[1])/float(passed)
                obj.velocity = (.5*obj.velocity[0]+.5*vx, .5*obj.velocity[1]+.5*vy)
                obj.last_update_frame = frame_count
                obj.lost = 0; obj.conf = result.score; obj.cls = result.cls; obj.frame_count += 1
                obj.update_position(box)
                self._check_line_crossing(obj, frame=frame, frame_uuid=frame_uuid)
            else:
                obj.lost = getattr(result, "lost", obj.lost + 1)
                obj.update_position(box)

            # Fast Retirement cho người đã đếm hoặc thoát mép khung hình
            bx, by, bw, bh = obj.bbox
            near_boundary = (bx <= margin or by <= margin or (bx + bw) >= (w - margin) or (by + bh) >= (h - margin))
            if obj.counted and obj.lost >= counted_retire:
                early_retired.add(track_id)
            elif near_boundary and obj.lost >= 2:
                early_retired.add(track_id)

            seen.add(track_id)

        all_expired = set(expired) | early_retired
        if early_retired and hasattr(self.byte_tracker, "retire"):
            self.byte_tracker.retire(early_retired)

        for track_id in all_expired:
            obj = self._person_tracks.pop(track_id, None)
            if obj is not None and self.track_stitcher:
                self.track_stitcher.archive(track_id, obj.bbox, obj.velocity, obj)
            if track_id in early_retired:
                self._stats["expired"] += 1

        active_objects = []
        for t in results:
            if t.track_id in self._person_tracks:
                obj = self._person_tracks[t.track_id]
                if obj.lost <= 1:
                    active_objects.append(obj)

        # Khử trùng lặp giữa các track đang active (nếu 2 track lồng nhau trên cùng 1 người)
        if len(active_objects) > 1:
            sorted_active = sorted(active_objects, key=lambda o: (getattr(o, "counted", False), o.bbox[3], o.conf or 0), reverse=True)
            kept_active = []
            dup_ids = set()
            for obj in sorted_active:
                if obj.id in dup_ids:
                    continue
                bx, by, bw, bh = obj.bbox
                ba = bw * bh
                kept_active.append(obj)
                for other in sorted_active:
                    if other.id == obj.id or other.id in dup_ids:
                        continue
                    ox, oy, ow, oh = other.bbox
                    oa = ow * oh
                    ix1, iy1 = max(bx, ox), max(by, oy)
                    ix2, iy2 = min(bx + bw, ox + ow), min(by + bh, oy + oh)
                    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                    min_a = min(ba, oa)
                    if min_a > 0 and (inter / min_a) > 0.60:
                        dup_ids.add(other.id)

            if dup_ids:
                for did in dup_ids:
                    self._person_tracks.pop(did, None)
                if hasattr(self.byte_tracker, "retire"):
                    self.byte_tracker.retire(dup_ids)
                active_objects = kept_active

        return active_objects

    def _detect_round(self, tracked_objects, frame, frame_count, frame_uuid=None):
        detections = self._detections(frame)
        if detections is None:
            return tracked_objects
        if not settings.USE_BYTETRACK:
            return self._fallback_detect_round(tracked_objects, detections, frame, frame_count, frame_uuid)
        results, expired = self.byte_tracker.update(detections)
        self._stats["matched_high"] += self.byte_tracker.last_high_matches
        self._stats["matched_low"] += self.byte_tracker.last_low_matches
        self._stats["expired"] += len(expired)
        return self._process_tracker_results(results, expired, frame, frame_count, frame_uuid)

    def _track_round(self, tracked_objects, frame, frame_uuid=None):
        if not settings.USE_BYTETRACK:
            survivors=[]
            for obj in tracked_objects:
                obj.predict_forward(frame.shape)
                obj.frame_count += 1
                obj.lost += 1
                if obj.lost < max(1, settings.MAX_LOST_ROUNDS):
                    survivors.append(obj)
                else:
                    self._stats["expired"] += 1
            return survivors
        results, expired = self.byte_tracker.predict()
        self._stats["expired"] += len(expired)
        return self._process_tracker_results(results, expired, frame, self._stats["frames"], frame_uuid)

    def _fallback_detect_round(self, tracked_objects, detections, frame, frame_count, frame_uuid=None):
        """High-confidence global IoU association used when rolling ByteTrack back."""
        detections=[d for d in detections if d[1]>=settings.CONFIDENT_PERSON]
        boxes=[d[0] for d in detections]
        overlap=iou_matrix([obj.bbox for obj in tracked_objects],boxes)
        pairs,_,_=linear_assignment(1.0-overlap)
        matched_objects=set(); matched_detections=set(); threshold=float(settings.MATCH_IOU_THRESHOLD)
        for oi,di in pairs:
            if overlap[oi,di] < threshold:
                continue
            obj=tracked_objects[oi]; bbox,conf,cls=detections[di]
            bbox=tuple(int(round(v)) for v in bbox)
            previous=obj.bbox; passed=max(1,frame_count-(obj.last_update_frame or frame_count-1))
            vx=(bbox[0]-previous[0])/passed; vy=(bbox[1]-previous[1])/passed
            obj.velocity=(.5*obj.velocity[0]+.5*vx,.5*obj.velocity[1]+.5*vy)
            obj.update_position(bbox); obj.conf=conf; obj.cls=cls; obj.lost=0
            obj.last_update_frame=frame_count; obj.frame_count+=1
            self._check_line_crossing(obj,frame,frame_uuid)
            matched_objects.add(oi); matched_detections.add(di)
        survivors=[]
        for oi,obj in enumerate(tracked_objects):
            if oi in matched_objects:
                survivors.append(obj); continue
            obj.lost+=1
            if obj.lost<max(1,settings.MAX_LOST_ROUNDS):
                obj.predict_forward(frame.shape); survivors.append(obj)
            else:
                self._stats["expired"]+=1
        for di,(bbox,conf,cls) in enumerate(detections):
            if di in matched_detections: continue
            obj=CountingTrackedPerson(self._object_id_counter,bbox,conf=conf,cls=cls)
            self._object_id_counter+=1; obj.last_update_frame=frame_count
            self._stats["tracks_created"]+=1; survivors.append(obj)
        return survivors

    def _check_line_crossing(self, obj, frame=None, frame_uuid=None):
        """Update per-track hysteresis state and emit a deduplicated crossing."""
        if not self._line_points:
            return
        if obj.crossing_fsm is None:
            obj.crossing_fsm = LineZoneCrossingFSM(
                self._line_points, settings.LINE_BUFFER_PIXELS, self._direction_vector,
                settings.STATIONARY_DISPLACEMENT_MAX, settings.STATIONARY_FRAMES)
        point = obj.curr_bottom or obj.curr_center
        event, stationary = obj.crossing_fsm.update(point, obj.bbox)
        if stationary:
            self._stats["gate_movement"] += 1
        if obj.counted or not event:
            return
        min_conf = float(getattr(settings, "CONFIDENT_PERSON", 0.25))
        if obj.conf is not None and float(obj.conf) < min_conf:
            return
        if obj.frame_count < settings.MIN_FRAMES_BEFORE_COUNT:
            self._stats["gate_frames"] += 1
            return
        if not self._cooldown.allow(point, obj.bbox, event):
            obj.counted = True
            logger.info(f"[{self.cam_name}] Bỏ đếm track #{obj.id}: spatial cooldown")
            return
        obj.counted = True
        self._stats["counted"] += 1
        self.total_counted += 1
        person_class = event
        direction_label = "IN" if event == "person_in" else "OUT"
        conf = obj.conf if obj.conf is not None else 0.0
        logger.info(f"[{self.cam_name}] VƯỢT VẠCH {direction_label} — người #{obj.id} "
                    f"conf={conf:.2f} bbox={obj.bbox} frames={obj.frame_count}")
        if getattr(settings, "ENABLE_SAVE_IMAGE", True) and frame is not None:
            try:
                self.image_writer_pool.submit(
                    save_counted_person_images, frame=frame.copy(), obj=CountedPersonSnapshot(obj),
                    line_points=list(self._line_points), direction_vector=self._direction_vector,
                    direction_label=direction_label, cam_id=self.stream_id, frame_uuid=frame_uuid,
                    cam_name=self.cam_name)
            except Exception as exc:
                logger.warning(f"[{self.cam_name}] Không thể submit tác vụ lưu ảnh: {exc}")
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
            f"det={s['detections']} ByteTrack_high={s['matched_high']} "
            f"ByteTrack_low={s['matched_low']} track_mới={s['tracks_created']} "
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
