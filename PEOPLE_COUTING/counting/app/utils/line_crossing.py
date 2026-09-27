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
