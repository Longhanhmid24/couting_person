"""
line_crossing.py — Line crossing detection algorithm for vehicle counting.
Uses segment intersection (CCW test) to determine if a vehicle's movement
path crosses the counting line exactly once.
"""
import logging

logger = logging.getLogger(__name__)


def parse_line_points(line_config, frame_w, frame_h):
    """
    Parse line configuration into pixel coordinates.
    
    line_config can be:
      - List of 2 points: [[x1,y1], [x2,y2]]
      - List of dicts: [{"x": 0.3, "y": 0.5}, {"x": 0.8, "y": 0.5}]
      - Normalized (0-1) or pixel values (auto-scales if based on higher resolution)
    
    Returns: list of 2 tuples [(x1,y1), (x2,y2)] or empty list if invalid.
    """
    if not line_config:
        return []

    if isinstance(line_config, str):
        try:
            import json
            line_config = json.loads(line_config)
        except Exception:
            return []

    if not isinstance(line_config, list) or len(line_config) < 2:
        return []

    raw_points = []
    for pt in line_config[:2]:  # Only take first 2 points for a line
        x, y = None, None
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            x, y = pt[0], pt[1]
        elif isinstance(pt, dict):
            x = pt.get("x", pt.get("X"))
            y = pt.get("y", pt.get("Y"))

        if x is not None and y is not None:
            try:
                raw_points.append((float(x), float(y)))
            except (ValueError, TypeError):
                pass

    if len(raw_points) < 2:
        return []

    # Check if any coordinate exceeds 1.0 (indicating absolute pixels)
    is_normalized = all(x <= 1.0 and y <= 1.0 for x, y in raw_points)

    points = []
    if is_normalized:
        for x, y in raw_points:
            px = int(round(x * frame_w))
            py = int(round(y * frame_h))
            points.append((px, py))
    else:
        # Determine if pixel coordinates exceed current frame dimensions (e.g. 2560x1440 config on 1280x720 frame)
        max_x = max(x for x, y in raw_points)
        max_y = max(y for x, y in raw_points)

        scale_x = 1.0
        scale_y = 1.0
        if max_x > frame_w and frame_w > 0:
            ref_w = 2560.0 if max_x <= 2560.0 else 3840.0
            scale_x = float(frame_w) / ref_w
        if max_y > frame_h and frame_h > 0:
            ref_h = 1440.0 if max_y <= 1440.0 else 2160.0
            scale_y = float(frame_h) / ref_h

        for x, y in raw_points:
            px = int(round(x * scale_x))
            py = int(round(y * scale_y))
            # Clamp to frame boundaries with slight margin
            px = max(0, min(frame_w, px))
            py = max(0, min(frame_h, py))
            points.append((px, py))

    return points


_EPS = 1e-9


def _ccw(A, B, C):
    """Giữ lại cho tương thích: True nếu A, B, C ngược chiều kim đồng hồ."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def _orientation(a, b, c):
    """Dấu của tích có hướng: 1 / -1 / 0 (thẳng hàng)."""
    val = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if val > _EPS:
        return 1
    if val < -_EPS:
        return -1
    return 0


def segments_intersect(A, B, C, D):
    """AB có cắt CD không.

    Dùng orientation 3 trạng thái thay vì phép so sánh `>` của bản cũ. Bản cũ coi
    "thẳng hàng" giống hệt "cùng phía", nên khi tâm xe rơi ĐÚNG lên vạch (chuyện
    thường xuyên xảy ra vì toạ độ được làm tròn về pixel) thì lần vượt vạch đó bị
    bỏ, và xe không bao giờ được đếm nữa vì các frame sau đã ở hẳn bên kia vạch.
    """
    o1 = _orientation(A, B, C)
    o2 = _orientation(A, B, D)
    o3 = _orientation(C, D, A)
    o4 = _orientation(C, D, B)
    return o1 != o2 and o3 != o4


def path_crosses_line(path, line_points):
    """Có đoạn nào trong quỹ đạo *path* cắt vạch không.

    Xét TOÀN BỘ quỹ đạo gần đây chứ không chỉ bước cuối. Bản cũ chỉ so
    (prev -> curr): nếu đúng bước vượt vạch lại bị cổng chống rung loại bỏ, hoặc
    bị ghi đè bởi bước kế tiếp, thì lần vượt vạch mất luôn.
    """
    if not line_points or len(line_points) < 2 or not path or len(path) < 2:
        return False
    c, d = line_points[0], line_points[1]
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        if a is None or b is None:
            continue
        if segments_intersect(a, b, c, d):
            return True
    return False


def path_net_displacement(path):
    """Khoảng cách từ điểm đầu tới điểm cuối của quỹ đạo (pixel).

    Dùng làm cổng chống rung: xe rung tại chỗ có net displacement ~0 dù tổng
    đường đi có thể lớn, còn xe thật đi qua vạch thì net displacement lớn.
    """
    pts = [p for p in (path or []) if p is not None]
    if len(pts) < 2:
        return 0.0
    dx = pts[-1][0] - pts[0][0]
    dy = pts[-1][1] - pts[0][1]
    return (dx * dx + dy * dy) ** 0.5


def check_line_crossing(prev_pt, curr_pt, line_points):
    """
    Check if a point's movement (prev_pt → curr_pt) crosses the counting line.
    
    Args:
        prev_pt: (x, y) — previous frame position
        curr_pt: (x, y) — current frame position
        line_points: [(x1,y1), (x2,y2)] — counting line endpoints (pixel coords)
    
    Returns:
        True if the point just crossed the line.
    """
    if not line_points or len(line_points) < 2:
        return False

    if prev_pt is None or curr_pt is None:
        return False

    # Check segment intersection between movement path and counting line
    return segments_intersect(
        prev_pt, curr_pt,
        line_points[0], line_points[1]
    )


def get_bbox_center(bbox):
    """
    Get center point of bounding box.
    bbox format: (x, y, w, h) where x,y is top-left corner.
    """
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


def get_bbox_bottom_center(bbox):
    """
    Get bottom-center point of bounding box (wheel/ground contact point).
    bbox format: (x, y, w, h) where x,y is top-left corner.
    """
    x, y, w, h = bbox
    return (x + w / 2.0, y + float(h))


def check_vehicle_line_crossing(prev_center, curr_center, prev_bottom, curr_bottom, line_points):
    """
    Check if either the vehicle's bottom-center (wheels) or center path crosses the counting line.
    Provides robust detection for both tall vehicles (trucks/buses) and low vehicles (cars/motorbikes).
    """
    if not line_points or len(line_points) < 2:
        return False

    # 1. Check bottom-center trajectory (most accurate for road contact)
    if prev_bottom is not None and curr_bottom is not None:
        if check_line_crossing(prev_bottom, curr_bottom, line_points):
            return True

    # 2. Check center trajectory (fallback / verification)
    if prev_center is not None and curr_center is not None:
        if check_line_crossing(prev_center, curr_center, line_points):
            return True

    return False


def point_to_line_segment_distance(point, line_points):
    """
    Tính khoảng cách vuông góc từ điểm `point` (x, y) tới đoạn thẳng `line_points` [(x1,y1), (x2,y2)].
    Nếu hình chiếu vuông góc nằm ngoài đoạn thẳng, khoảng cách được tính tới đầu mút gần nhất.
    """
    if not line_points or len(line_points) < 2 or point is None:
        return 999999.0
    import math
    (x1, y1), (x2, y2) = line_points[0], line_points[1]
    px, py = float(point[0]), float(point[1])
    dx, dy = float(x2 - x1), float(y2 - y1)
    l2 = dx * dx + dy * dy
    if l2 < 1e-6:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / l2))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


class LineZoneCrossingFSM:
    """Per-track hysteresis state machine with pending crossing validation.
    
    Guarantees:
    1. Pending Crossing State: If an object crosses the line when displacement is still
       under threshold (e.g. starting close to the line), the crossing event is NOT dropped.
       Instead, it is held in pending status until the person takes subsequent steps to reach
       the required displacement, then emitted immediately.
    2. Stationary Suppression: Stationary objects (like parked motorbikes) jitter within 10-15px,
       so their pending crossing is never validated and safely expires with 0 count.
    3. Trajectory-Aware Direction: Direction is derived from the full trajectory vector and
       geometry-defined side transition (courtyard side +1 -> stairs side -1 is IN), not noisy single-frame jitter.
    4. Wide Margin Segment Projection: Points within [-0.10, 1.10] of the line are preserved,
       preventing drops when people walk near the stairs wall or boundaries.
    """
    def __init__(self, line_points, buffer_pixels=35.0, direction_vector=None,
                 stationary_displacement=35.0, stationary_frames=15,
                 first_point=None, min_entry_distance=25.0):
        from collections import deque
        self.line = line_points
        self.buffer = max(1.0, float(buffer_pixels))
        self.direction = tuple(direction_vector) if direction_vector else None
        self.history = deque(maxlen=max(2, int(stationary_frames)))
        self.stationary_displacement = float(stationary_displacement)
        self.min_entry_distance = float(min_entry_distance)
        self.first_point = tuple(float(c) for c in first_point) if first_point is not None else None
        self.state = None
        self.saw_transit = False
        self.transit_origin = None
        self.last_point = None
        self.out_of_bounds_frames = 0
        self.pending_event = None
        self.counted = False

    def _signed_distance(self, point):
        (x1, y1), (x2, y2) = self.line
        dx, dy = x2 - x1, y2 - y1
        length = max((dx * dx + dy * dy) ** 0.5, 1e-9)
        projection = ((point[0] - x1) * dx + (point[1] - y1) * dy) / (length * length)
        # Nới rộng biên [-0.10, 1.10] để không bỏ sót người đi bộ sát góc/tường cầu thang
        if projection < -0.10 or projection > 1.10:
            return None
        return (dx * (point[1] - y1) - dy * (point[0] - x1)) / length

    def _side(self, signed):
        if abs(signed) <= self.buffer / 2.0:
            return 0
        return 1 if signed > 0 else -1

    def _determine_direction(self, origin_side, dest_side, current_point):
        # 1. Dùng vector quỹ đạo tổng thể từ điểm bắt đầu (point - first_point)
        if self.direction and self.first_point is not None:
            dx = current_point[0] - self.first_point[0]
            dy = current_point[1] - self.first_point[1]
            dot = dx * self.direction[0] + dy * self.direction[1]
            if abs(dot) > 1e-4:
                return "person_in" if dot > 0 else "person_out"

        # 2. Suy luận từ hướng chuyển vế (side +1 ngoài sân -> side -1 cầu thang/cửa kính là IN)
        if origin_side == 1 and dest_side == -1:
            return "person_in"
        elif origin_side == -1 and dest_side == 1:
            return "person_out"

        # 3. Fallback theo bước chuyển cuối cùng
        if self.direction and self.last_point is not None:
            dx = current_point[0] - self.last_point[0]
            dy = current_point[1] - self.last_point[1]
            dot = dx * self.direction[0] + dy * self.direction[1]
            return "person_in" if dot > 0 else "person_out"

        return "person_in" if dest_side < 0 else "person_out"

    def update(self, point, bbox=None, now=None):
        if not self.line or point is None or self.counted:
            return None, False

        import math
        point = (float(point[0]), float(point[1]))
        signed = self._signed_distance(point)
        if signed is None:
            self.last_point = point
            self.out_of_bounds_frames += 1
            if self.out_of_bounds_frames >= 6:
                self.state = None
                self.saw_transit = False
                self.transit_origin = None
            return None, False

        self.out_of_bounds_frames = 0
        side = self._side(signed)

        if self.first_point is None:
            self.first_point = point

        self.history.append(point)

        previous = self.state
        if previous is None:
            self.state = side
            if side == 0:
                self.saw_transit = True
        elif side == 0:
            if previous != 0:
                self.transit_origin = previous
            self.state = 0
            self.saw_transit = True
        elif previous == 0:
            if self.saw_transit and self.transit_origin is not None and side != self.transit_origin:
                self.pending_event = self._determine_direction(self.transit_origin, side, point)
            self.state = side
            self.saw_transit = False
            self.transit_origin = None
        elif side != previous:
            self.state = side
            self.pending_event = self._determine_direction(previous, side, point)
            self.saw_transit = False
            self.transit_origin = None

        self.last_point = point

        # Nếu chưa ghi nhận vượt vạch thì không có gì để xử lý
        if self.pending_event is None:
            return None, False

        # Kiểm tra đứng yên / dịch chuyển
        net_disp = math.hypot(point[0] - self.first_point[0], point[1] - self.first_point[1])
        hist_stationary = (len(self.history) >= self.history.maxlen and max(
            (((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)
            for a in self.history for b in self.history
        ) < self.stationary_displacement)

        # Nếu chưa đủ dịch chuyển hoặc đang đứng yên: giữ nguyên pending_event, tiếp tục theo dõi
        if net_disp < self.stationary_displacement or hist_stationary:
            return None, True

        # Đã tích luỹ đủ dịch chuyển thực tế -> Phát sự kiện đếm thành công!
        event = self.pending_event
        self.pending_event = None
        self.counted = True
        return event, False


class SpatialCooldownRegistry:
    """Suppress nearby repeated counts during a short time window.

    Uses line-angle invariant geometry:
    - Calculates distance parallel to the counting line (d_parallel).
    - People walking side-by-side (2 or 3 people) cross at separated positions
      along the line (d_parallel >= min_parallel_dist), so they are never suppressed,
      regardless of whether the line is horizontal, vertical, or diagonal.
    - Two detections/tracks of the same person/vehicle cross at the same point
      along the line (d_parallel < min_parallel_dist) and within a short time window,
      so trailing/duplicate tracks are suppressed unless they have long independent history.
    """
    def __init__(self, cooldown_seconds=1.8, min_parallel_dist=60.0, min_independent_frames=12):
        from collections import deque
        self.cooldown = float(cooldown_seconds)
        self.min_parallel_dist = float(min_parallel_dist)
        self.min_independent_frames = int(min_independent_frames)
        self.events = deque()

    def allow(self, point, bbox, direction, now=None, track_id=None,
              track_frame_count=0, line_points=None):
        import time, math
        now = time.monotonic() if now is None else now

        # Expire old events
        while self.events and now - self.events[0][2] > self.cooldown:
            self.events.popleft()

        # Compute line unit direction vector if available
        u_line = None
        if line_points and len(line_points) >= 2:
            p1, p2 = line_points[0], line_points[1]
            lx = float(p2[0] - p1[0])
            ly = float(p2[1] - p1[1])
            llen = math.hypot(lx, ly)
            if llen > 1e-3:
                u_line = (lx / llen, ly / llen)

        w = max(float(bbox[2]), 1.0)
        radius = max(35.0, 0.5 * w)

        for ev in self.events:
            ex, ey, ets, edir, eradius, eid = ev
            if now - ets > self.cooldown:
                continue

            dx = point[0] - ex
            dy = point[1] - ey

            if u_line is not None:
                # Line-angle invariant: separation along the line segment
                d_parallel = abs(dx * u_line[0] + dy * u_line[1])
                # If crossing points are separated along the line (side-by-side persons), allow!
                if d_parallel >= self.min_parallel_dist:
                    continue
                # If both are established independent tracks (tracked together from far away)
                if track_frame_count >= self.min_independent_frames:
                    continue
                # Otherwise: same position along the line, within cooldown -> duplicate/fragment!
                return False
            else:
                # Fallback Euclidean distance check
                dist_sq = dx * dx + dy * dy
                r = max(radius, eradius)
                if dist_sq < r * r:
                    if track_frame_count >= self.min_independent_frames:
                        continue
                    return False

        self.events.append((float(point[0]), float(point[1]), now,
                            direction, radius, track_id))
        return True
