import os


def _flag(name: str, default: str = "False") -> bool:
    return os.getenv(name, default).strip().lower() in ("true", "1", "yes", "on")


def _normalize_base_url(raw: str) -> str:
    """
    Chuẩn hoá BASE_URL linh hoạt:
    - Bỏ khoảng trắng thừa (leading/trailing whitespace)
    - Tự động thêm 'http://' nếu chưa có protocol (ví dụ: '192.168.1.100:8080' -> 'http://192.168.1.100:8080')
    - Loại bỏ dấu '/' ở cuối (ví dụ: 'http://cms.domain.com/' -> 'http://cms.domain.com')
    """
    val = (raw or "").strip()
    if not val:
        return ""
    if not (val.startswith("http://") or val.startswith("https://")):
        val = f"http://{val}"
    return val.rstrip("/")


class Settings:
    """PEOPLE_COUTING — đếm người vượt vạch (line crossing)."""
    # Engine
    SERVICE_TYPES = tuple(t.strip() for t in os.getenv(
        "SERVICE_TYPES", "people_counting,people_couting").split(",") if t.strip())
    POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", 20))

    # Metric type gửi CMS (mặc định: people_counting)
    METRIC_TYPE = os.getenv("METRIC_TYPE", "people_counting")

    # API CMS Backend (được cấu hình linh hoạt qua biến môi trường)
    BASE_URL = _normalize_base_url(os.getenv("BASE_URL_API", ""))
    SESSION_KEY = os.getenv("SESSION_KEY", "").strip()

    # Base URL kiểm định xe (chỉ dùng cho LPR nếu có)
    BASE_URL_EXPIRED_INSPECTION = _normalize_base_url(os.getenv("BASE_URL_EXPIRED_INSPECTION", ""))

    # Cảnh báo Telegram (tuỳ chọn)
    TELEGRAM_BASE_URL = _normalize_base_url(os.getenv("TELEGRAM_BASE_URL", ""))
    TELE_CHAT_ID = os.getenv("TELE_CHAT_ID", "").strip()

    # GRPC — Person Detector gRPC Server (port 50060)
    YOLO_PERSON_ADDR = os.getenv("YOLO_PERSON_ADDR", "localhost:50060")
    GRPC_TIMEOUT = float(os.getenv("GRPC_TIMEOUT", 3.0))
    GRPC_MAX_RETRIES = int(os.getenv("GRPC_MAX_RETRIES", 2))
    GRPC_RETRY_DELAY = float(os.getenv("GRPC_RETRY_DELAY", 0.2))

    # Path
    ROOT_PATH = os.getenv(
        "ROOT_PATH",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    )
    SAVE_IMAGE_DIR = os.getenv("SAVE_IMAGE_DIR", os.path.join(ROOT_PATH, "images"))

    # Decode
    FORCE_CPU_DECODE = _flag("FORCE_CPU_DECODE")
    TARGET_FPS = int(os.getenv("TARGET_FPS", 25))

    # ── Nhịp detect ───────────────────────────────────────────────────
    DETECT_FPS = float(os.getenv("DETECT_FPS", 10))
    SKIP_FRAME = int(os.getenv("SKIP_FRAME", 3))  # dùng khi DETECT_FPS <= 0
    FRAME_BACKLOG_MAX = int(os.getenv("FRAME_BACKLOG_MAX", 4))

    # ── Ngưỡng confidence ─────────────────────────────────────────────
    CONFIDENT_PERSON = float(os.getenv("CONFIDENT_PERSON", 0.35))

    # ── Tracking ──────────────────────────────────────────────────────
    MATCH_IOU_THRESHOLD = float(os.getenv("MATCH_IOU_THRESHOLD", 0.15))
    MATCH_MAX_AREA_RATIO = float(os.getenv("MATCH_MAX_AREA_RATIO", 4.0))
    MATCH_BY_DISTANCE = _flag("MATCH_BY_DISTANCE", "True")
    MATCH_DISTANCE_FACTOR = float(os.getenv("MATCH_DISTANCE_FACTOR", 1.8))
    MAX_LOST_ROUNDS = int(os.getenv("MAX_LOST_ROUNDS", 15))

    # ── HOG nội suy giữa hai vòng YOLO ────────────────────────────────
    HOG_HEAVY_EVERY = int(os.getenv("HOG_HEAVY_EVERY", 2))
    HOG_MAX_WINDOWS = int(os.getenv("HOG_MAX_WINDOWS", 16))
    HOG_MIN_SIMILARITY = float(os.getenv("HOG_MIN_SIMILARITY", 0.20))
    HOG_GAIN_MARGIN = float(os.getenv("HOG_GAIN_MARGIN", 0.05))

    # ── Điều kiện đếm ─────────────────────────────────────────────────
    MIN_FRAMES_BEFORE_COUNT = int(os.getenv("MIN_FRAMES_BEFORE_COUNT", 2))
    MIN_PATH_MOVEMENT_PIXELS = float(os.getenv("MIN_PATH_MOVEMENT_PIXELS", 5.0))
    PATH_HISTORY_LEN = int(os.getenv("PATH_HISTORY_LEN", 12))

    # ── Statistics ────────────────────────────────────────────────────
    STATISTICS_POST_INTERVAL = int(os.getenv("STATISTICS_POST_INTERVAL", 5))
    STATS_LOG_INTERVAL = float(os.getenv("STATS_LOG_INTERVAL", 60.0))

    # Motion Detection
    USE_MOTION_DETECTOR = _flag("USE_MOTION_DETECTOR")

    # Other
    TARGET_SIZE = int(os.getenv("TARGET_SIZE", 640))


settings = Settings()
