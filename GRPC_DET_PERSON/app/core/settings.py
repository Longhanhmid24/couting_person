import os


class Settings:
    """GRPC_DET_PERSON — YOLO person detector (port 50060). Dynamic batching enabled."""
    # gRPC
    GRPC_PORT = int(os.getenv("GRPC_PORT", 50060))
    LOG_TAG = os.getenv("LOG_TAG", "DET-PERSON")

    # Paths
    ROOT_PATH = os.getenv("ROOT_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    MODEL_PT = os.getenv("MODEL_PT", os.path.join(ROOT_PATH, "models", "yolo", "pytorch", "yolo_best3.pt"))
    MODEL_ENGINE = os.getenv("MODEL_ENGINE", os.path.join(ROOT_PATH, "models", "yolo", "trt", "yolo_best3.engine"))

    # Inference
    USE_TENSORRT = os.getenv("USE_TENSORRT", "false").lower() in ("true", "1", "yes")
    CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", 0.35))
    EXPORT_DYNAMIC_BATCH = os.getenv("EXPORT_DYNAMIC_BATCH", "true").lower() in ("true", "1", "yes")

    # Classes
    CLASS_NAMES = {
        0: "person",
    }


settings = Settings()
