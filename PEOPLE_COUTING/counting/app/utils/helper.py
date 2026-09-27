from time_uuid import TimeUUID
import os, re, json, logging, requests
from core.settings import settings
import cv2
import numpy as np

logger = logging.getLogger(__name__)

def normalize_vehicle_class(cls_name: str) -> str:
    """Normalize vehicle class names to consistent strings."""
    if not cls_name:
        return "unknown"
    c = str(cls_name).lower().strip()
    if c in ("car", "oto", "ô tô", "4wheels"):
        return "car"
    elif c in ("bus", "xe buýt", "xe khach", "xe khách"):
        return "bus"
    elif c in ("truck", "xe tải", "xe tai"):
        return "truck"
    elif c in ("motor", "motorbike", "motorcycle", "xe máy", "xe may", "2wheels"):
        return "motorbike"
def normalize_person_class(cls_name: str) -> str:
    """Normalize person class names to consistent strings."""
    if not cls_name:
        return "person"
    c = str(cls_name).lower().strip()
    if c in ("person", "human", "người", "nguoi", "pedestrian", "man", "woman", "child", "0"):
        return "person"
    return "person"

def get_images_path(uuid_str):
    """ Generate the directory path for images based on a UUID string.
    Args:
        uuid_str (str): The UUID string used to locate the image.
    Returns:
        str: The directory path where the image is stored.
    """
    try:
        t_uuid = TimeUUID(uuid_str)
        iso_date = t_uuid.get_datetime().date().isoformat()
        directories = uuid_str.split('-')[::-1]
        directories.pop(1)
        return os.path.join(
            settings.SAVE_IMAGE_DIR,
            directories[0],
            iso_date,
            *directories[1:-2],
            directories[-2][:2],
            directories[-2][2:],
            directories[-1][:2]
        )
    except Exception:
        return settings.SAVE_IMAGE_DIR

def get_path_frame_uuid(uuid_str):
    """ Generate the file path for an image based on a UUID string.
    Args:
        uuid_str (str): The UUID string used to locate the image.
    Returns:
        str: The full file path where the image is stored.
    """
    try:
        return os.path.join(
            get_images_path(uuid_str),
            f"{uuid_str}.jpg"
        )
    except Exception:
        return os.path.join(settings.SAVE_IMAGE_DIR, f"{uuid_str}.jpg")
def get_vehicle_with_uuid(uuid_str,bbox):
    """ Retrieve a cropped face image using a UUID string and a bounding box.
    Args:
        uuid_str (str): The UUID string used to locate the image.
        bbox (list): Bounding box in the format [x1, y1, x2, y2] scaled to 0-1 range.
    Returns:
        numpy.ndarray: The cropped face image if found, otherwise None.
    """
    image_path = get_path_frame_uuid(uuid_str)
    if not os.path.exists(image_path):
        return None
    image = cv2.imread(image_path)
    if image is None:
        return None
    image_crop = crop_image_bbox_scaled(image, bbox)
    if image_crop is None:
        return None
    return image_crop

## Scale xyxy to 0-1 range
def scale_xyxy(xyxy, width, height):
    """
    Scale xyxy coordinates to a 0-1 range based on the given width and height.
    
    Args:
        xyxy (list): List of coordinates in the format [x1, y1, x2, y2].
        width (int): Width of the image.
        height (int): Height of the image.
    
    Returns:
        list: Scaled coordinates in the format [x1_scaled, y1_scaled, x2_scaled, y2_scaled].
    """
    return [xyxy[0] / width, xyxy[1] / height, xyxy[2] / width, xyxy[3] / height]

def crop_image_bbox_scaled(image, bbox):
    """
    Crop an image using a bounding box with coordinates scaled to a 0-1 range.

    Args:
        image (numpy.ndarray): The input image.
        bbox (list): Bounding box in the format [x1, y1, x2, y2] scaled to 0-1 range.

    Returns:
        numpy.ndarray: Cropped image.
    """
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(coord * dim) for coord, dim in zip(bbox, (width, height, width, height))]
    return image[y1:y2, x1:x2] if x1 < x2 and y1 < y2 else None

def get_image_with_uuid(uuid_str):
    """
    Retrieve an image using a UUID string.
    
    Args:
        uuid_str (str): The UUID string used to locate the image.
    
    Returns:
        numpy.ndarray: The image if found, otherwise None.
    """
    image_path = get_path_frame_uuid(uuid_str)
    if not os.path.exists(image_path):
        return None
    image = cv2.imread(image_path)
    return image if image is not None else None

def save_image_with_uuid(uuid_str, image, quality):
    """
    Save an image to a path determined by the UUID string.
    Falls back to local fallback directory if permission denied on configured mount path.
    """
    path = get_path_frame_uuid(uuid_str)
    try:
        os.makedirs(os.path.dirname(path), mode=0o777, exist_ok=True)
        success = cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if success:
            try:
                os.chmod(path, 0o666)
            except Exception:
                pass
            return path
    except Exception as e:
        logger.warning(f"Failed to write image to {path}: {e}")
        
    # Fallback to local images folder inside ROOT_PATH
    try:
        fallback_dir = os.path.join(settings.ROOT_PATH, "images", "fallback")
        os.makedirs(fallback_dir, mode=0o777, exist_ok=True)
        fallback_path = os.path.join(fallback_dir, f"{uuid_str}.jpg")
        cv2.imwrite(fallback_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        try:
            os.chmod(fallback_path, 0o666)
        except Exception:
            pass
        return fallback_path
    except Exception as ex:
        logger.error(f"Fallback image write also failed for {uuid_str}: {ex}")
        return path

def save_frame_with_name(name_frame: str, image, quality=95):
    path = f"{settings.SAVE_IMAGE_DIR}/{name_frame}.jpg"
    try:
        os.makedirs(os.path.dirname(path), mode=0o777, exist_ok=True)
        cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        try:
            os.chmod(path, 0o666)
        except Exception:
            pass
        return path
    except Exception as e:
        fallback_path = os.path.join(settings.ROOT_PATH, "images", f"{name_frame}.jpg")
        os.makedirs(os.path.dirname(fallback_path), mode=0o777, exist_ok=True)
        cv2.imwrite(fallback_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        try:
            os.chmod(fallback_path, 0o666)
        except Exception:
            pass
        return fallback_path

def draw_person_counting_evidence(frame: np.ndarray, obj, line_points=None,
                                  direction_vector=None, direction_label: str = "IN",
                                  max_w: int = 1280, max_h: int = 720) -> np.ndarray:
    """Vẽ ảnh bằng chứng khoảnh khắc người vượt qua vạch đếm."""
    if frame is None:
        return None
    evidence = frame.copy()
    h_orig, w_orig = evidence.shape[:2]

    # 1. Vẽ vạch đếm (Counting Line) màu vàng nét đậm
    if line_points and len(line_points) >= 2:
        try:
            pt1 = (int(round(line_points[0][0])), int(round(line_points[0][1])))
            pt2 = (int(round(line_points[1][0])), int(round(line_points[1][1])))
            cv2.line(evidence, pt1, pt2, (0, 255, 255), 3, cv2.LINE_AA)
            cv2.circle(evidence, pt1, 5, (0, 200, 255), -1, cv2.LINE_AA)
            cv2.circle(evidence, pt2, 5, (0, 200, 255), -1, cv2.LINE_AA)

            mid_x = (pt1[0] + pt2[0]) // 2
            mid_y = (pt1[1] + pt2[1]) // 2
            cv2.putText(evidence, "COUNTING LINE", (mid_x - 40, max(20, mid_y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        except Exception as e:
            logger.debug(f"Lỗi vẽ vạch đếm: {e}")

    # 2. Vẽ vector hướng (Direction Arrow) nếu có
    if direction_vector is not None and line_points and len(line_points) >= 2:
        try:
            mid_x = (line_points[0][0] + line_points[1][0]) / 2.0
            mid_y = (line_points[0][1] + line_points[1][1]) / 2.0
            vx, vy = direction_vector
            norm = (vx ** 2 + vy ** 2) ** 0.5
            if norm > 1e-6:
                arr_len = 50.0
                ax = int(round(mid_x + (vx / norm) * arr_len))
                ay = int(round(mid_y + (vy / norm) * arr_len))
                cv2.arrowedLine(evidence, (int(round(mid_x)), int(round(mid_y))),
                                (ax, ay), (255, 100, 0), 2, tipLength=0.3)
        except Exception:
            pass

    # 3. Vẽ quỹ đạo di chuyển (path bottom)
    if hasattr(obj, 'path_bottom') and obj.path_bottom:
        pts = [p for p in obj.path_bottom if p is not None]
        for i in range(len(pts) - 1):
            p1 = (int(round(pts[i][0])), int(round(pts[i][1])))
            p2 = (int(round(pts[i + 1][0])), int(round(pts[i + 1][1])))
            cv2.line(evidence, p1, p2, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(evidence, p1, 3, (0, 255, 255), -1, cv2.LINE_AA)
        if pts:
            last_p = (int(round(pts[-1][0])), int(round(pts[-1][1])))
            cv2.circle(evidence, last_p, 4, (0, 0, 255), -1, cv2.LINE_AA)

    # 4. Vẽ Bounding Box người được đếm
    bx, by, bw, bh = obj.bbox
    x1, y1 = max(0, int(bx)), max(0, int(by))
    x2, y2 = min(w_orig, int(bx + bw)), min(h_orig, int(by + bh))

    is_in = (direction_label.upper() == "IN")
    box_color = (0, 255, 0) if is_in else (0, 165, 255)  # Xanh lá = IN, Cam = OUT
    cv2.rectangle(evidence, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

    # 5. Vẽ Badge Tag thông tin người được đếm
    conf_str = f" {int(obj.conf * 100)}%" if getattr(obj, 'conf', None) else ""
    tag_text = f"Person #{obj.id} [{direction_label}]{conf_str}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    (tw, th), _baseline = cv2.getTextSize(tag_text, font, font_scale, thickness)

    tag_y1 = max(0, y1 - th - 8)
    tag_y2 = y1
    tag_x2 = min(w_orig, x1 + tw + 10)
    cv2.rectangle(evidence, (x1, tag_y1), (tag_x2, tag_y2), box_color, -1)
    text_color = (0, 0, 0) if is_in else (255, 255, 255)
    cv2.putText(evidence, tag_text, (x1 + 5, y1 - 5), font, font_scale, text_color, thickness, cv2.LINE_AA)

    # 6. Resize tối ưu max 1280x720 nếu vượt quá
    if max_w > 0 and max_h > 0 and (w_orig > max_w or h_orig > max_h):
        scale = min(max_w / w_orig, max_h / h_orig)
        nw, nh = int(round(w_orig * scale)), int(round(h_orig * scale))
        evidence = cv2.resize(evidence, (nw, nh), interpolation=cv2.INTER_AREA)

    return evidence


def crop_person_image(frame: np.ndarray, bbox, padding_ratio: float = 0.1) -> np.ndarray:
    """Cắt ảnh cận cảnh dáng người với lề padding an toàn."""
    if frame is None or bbox is None:
        return None
    try:
        h, w = frame.shape[:2]
        bx, by, bw, bh = bbox
        pad_w = int(round(bw * padding_ratio))
        pad_h = int(round(bh * padding_ratio))
        x1 = max(0, int(bx - pad_w))
        y1 = max(0, int(by - pad_h))
        x2 = min(w, int(bx + bw + pad_w))
        y2 = min(h, int(by + bh + pad_h))
        if x2 > x1 and y2 > y1:
            crop = frame[y1:y2, x1:x2].copy()
            return crop
    except Exception as e:
        logger.debug(f"Lỗi crop ảnh người: {e}")
    return None


def save_counted_person_images(frame: np.ndarray, obj, line_points=None,
                               direction_vector=None, direction_label: str = "IN",
                               cam_id: str = "camera", frame_uuid: str = None,
                               cam_name: str = None) -> dict:
    """Lưu ảnh evidence và crop của người vừa đếm vào thư mục dùng chung (Shared Storage),
    đồng thời đẩy sự kiện kèm ảnh lên CMS qua POST /api/alarms."""
    if not getattr(settings, 'ENABLE_SAVE_IMAGE', True) or frame is None:
        return {}

    from datetime import datetime, timezone, timedelta
    import uuid as _uuid

    tz_vn = timezone(timedelta(hours=7))
    now_dt = datetime.now(tz_vn)
    date_str = now_dt.strftime("%Y-%m-%d")
    ts_int = int(now_dt.timestamp())

    # Tạo frame_uuid nếu chưa có (dùng TimeUUID)
    if not frame_uuid:
        try:
            node = _uuid.UUID(cam_id).node if isinstance(cam_id, str) and len(cam_id) == 36 else _uuid.getnode()
            frame_uuid = str(_uuid.uuid1(node=node))
        except Exception:
            frame_uuid = str(_uuid.uuid4())

    quality = max(30, min(100, int(getattr(settings, 'JPEG_QUALITY', 75))))
    max_w = int(getattr(settings, 'MAX_IMAGE_WIDTH', 1280))
    max_h = int(getattr(settings, 'MAX_IMAGE_HEIGHT', 720))
    mode = str(getattr(settings, 'IMAGE_STORAGE_MODE', 'both')).lower().strip()
    save_crop = getattr(settings, 'SAVE_PERSON_CROP', True)

    evidence = draw_person_counting_evidence(
        frame=frame,
        obj=obj,
        line_points=line_points,
        direction_vector=direction_vector,
        direction_label=direction_label,
        max_w=max_w,
        max_h=max_h
    )
    crop = crop_person_image(frame, obj.bbox) if save_crop else None

    saved_paths = {"frame_uuid": frame_uuid}

    # 1. Chế độ TimeUUID (Khớp chuẩn FileStorage CMS)
    if mode in ("uuid", "both"):
        try:
            uuid_path = save_image_with_uuid(frame_uuid, evidence, quality)
            saved_paths["uuid_path"] = uuid_path
            if crop is not None:
                crop_uuid = f"crop_{frame_uuid}"
                crop_path = save_image_with_uuid(crop_uuid, crop, quality)
                saved_paths["crop_uuid_path"] = crop_path
        except Exception as e:
            logger.warning(f"Lỗi lưu ảnh mode uuid: {e}")

    # 2. Chế độ phân cấp theo Camera & Ngày tháng (Trực quan & Static serve)
    if mode in ("hierarchy", "both"):
        try:
            base_dir = getattr(settings, 'SAVE_IMAGE_DIR', os.path.join(settings.ROOT_PATH, "images"))
            target_dir = os.path.join(base_dir, str(cam_id), date_str)
            os.makedirs(target_dir, mode=0o777, exist_ok=True)

            fname = f"{ts_int}_{direction_label}_track_{obj.id}.jpg"
            fpath = os.path.join(target_dir, fname)
            cv2.imwrite(fpath, evidence, [cv2.IMWRITE_JPEG_QUALITY, quality])
            try:
                os.chmod(fpath, 0o666)
            except Exception:
                pass
            saved_paths["hierarchy_path"] = fpath

            if crop is not None:
                crop_fname = f"crop_{ts_int}_{direction_label}_track_{obj.id}.jpg"
                crop_fpath = os.path.join(target_dir, crop_fname)
                cv2.imwrite(crop_fpath, crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
                try:
                    os.chmod(crop_fpath, 0o666)
                except Exception:
                    pass
                saved_paths["crop_hierarchy_path"] = crop_fpath
        except Exception as e:
            logger.warning(f"Lỗi lưu ảnh mode hierarchy: {e}")

    # 3. Gửi sự kiện đếm người kèm ảnh evidence lên CMS Backend (POST /api/alarms)
    try:
        from api_clients.api_clients import send_person_counting_event
        frame_h, frame_w = frame.shape[:2]
        x, y, w, h = obj.bbox
        norm_bbox = [
            max(0.0, min(1.0, round(float(x) / frame_w, 4))),
            max(0.0, min(1.0, round(float(y) / frame_h, 4))),
            max(0.0, min(1.0, round(float(x + w) / frame_w, 4))),
            max(0.0, min(1.0, round(float(y + h) / frame_h, 4))),
        ]
        send_person_counting_event(
            stream_id=str(cam_id),
            direction=direction_label,
            bbox=norm_bbox,
            conf=float(obj.conf if obj.conf is not None else 0.8),
            track_id=int(obj.id),
            frame_id=frame_uuid,
            frame=evidence,
            cam_name=cam_name or str(cam_id),
        )
    except Exception as e:
        logger.warning(f"[{cam_id}] Không thể gửi sự kiện đếm người lên CMS: {e}")

    logger.info(f"[{cam_id}] Đã lưu ảnh người #{obj.id} [{direction_label}]: {saved_paths.get('hierarchy_path') or saved_paths.get('uuid_path')}")
    return saved_paths


def cleanup_old_images(base_dir: str = None, retention_days: int = None):
    """Xóa các thư mục ảnh cũ hơn retention_days để tránh đầy đĩa."""
    import shutil
    from datetime import datetime, timedelta
    days = retention_days if retention_days is not None else getattr(settings, 'IMAGE_RETENTION_DAYS', 30)
    if days <= 0:
        return
    b_dir = base_dir or getattr(settings, 'SAVE_IMAGE_DIR', os.path.join(settings.ROOT_PATH, "images"))
    if not os.path.exists(b_dir):
        return

    cutoff_date = datetime.now() - timedelta(days=days)
    logger.info(f"Bắt đầu dọn dẹp ảnh cũ trước ngày: {cutoff_date.strftime('%Y-%m-%d')}")
    try:
        for root, dirs, _files in os.walk(b_dir, topdown=False):
            for d in list(dirs):
                try:
                    dir_dt = datetime.strptime(d, "%Y-%m-%d")
                    if dir_dt < cutoff_date:
                        full_dir = os.path.join(root, d)
                        logger.info(f"Xóa thư mục ảnh quá hạn: {full_dir}")
                        shutil.rmtree(full_dir, ignore_errors=True)
                except ValueError:
                    pass
    except Exception as e:
        logger.error(f"Lỗi trong quá trình dọn dẹp ảnh cũ: {e}")

def verify_plate_pattern(plate_text, pattern):
    """
    Kiểm tra biển số có khớp với pattern không
    Hỗ trợ ký tự đại diện % (wildcard) và _ (single character)
    
    Args:
        plate_text (str): Biển số cần kiểm tra
        pattern (str): Pattern để so sánh
        
    Returns:
        bool: True nếu khớp, False nếu không khớp
    """
    if not plate_text or not pattern:
        return False

    _plate = str(plate_text).strip().lower()
    _pattern = str(pattern).strip().lower()

    if not _plate or not _pattern:
        return False

    clean_plate = re.sub(r'[\s\.\-_]', '', _plate)
    clean_pattern = re.sub(r'[\s\.\-_]', '', _pattern)

    try:
        # Xử lý ký tự đại diện %
        if "%" in _pattern:
            parts = [re.escape(p) for p in _pattern.split('%')]
            regex_body = '.*'.join(parts)
            prefix = '' if _pattern.startswith('%') else '^'
            suffix = '' if _pattern.endswith('%') else '$'
            regex_str = prefix + regex_body + suffix
            
            match = re.search(regex_str, _plate) or re.search(regex_str, clean_plate)
            return match is not None

        # Xử lý ký tự đại diện _
        elif "_" in _pattern:
            parts = [re.escape(p) for p in _pattern.split('_')]
            regex_body = '.'.join(parts)
            regex_str = '^' + regex_body + '$'
            match = re.search(regex_str, _plate) or re.search(regex_str, clean_plate)
            return match is not None

        # So sánh chính xác
        else:
            return (_plate == _pattern) or (clean_plate == clean_pattern)

    except re.error as e:
        logger.error(f"Invalid regex pattern '{pattern}': {e}")
        return False

def send_message_telegram(caption: str) -> bool:
    """Gửi tin nhắn văn bản thuần túy qua Telegram bot (/sendMessage)."""
    base_url = getattr(settings, 'TELEGRAM_BASE_URL', '') or ''
    chat_id = getattr(settings, 'TELE_CHAT_ID', '') or ''
    if not base_url or not chat_id:
        logger.debug("Telegram: Chưa cấu hình TELEGRAM_BASE_URL hoặc TELEGRAM_CHAT_ID")
        return False

    url = f"{base_url.rstrip('/')}/sendMessage"
    try:
        data = {
            "chat_id": chat_id,
            "text": caption,
            "parse_mode": "HTML"
        }
        resp = requests.post(url, json=data, timeout=10)
        if resp.status_code == 200 and resp.json().get("ok"):
            logger.info("Telegram: Gửi tin nhắn văn bản thành công")
            return True
        else:
            logger.error(f"Telegram sendMessage HTTP {resp.status_code}: {resp.text}")
            return False
    except Exception as e:
        logger.error(f"Telegram sendMessage error: {e}")
        return False

def send_image_telegram(image_path: str, caption: str) -> None:
    base_url = getattr(settings, 'TELEGRAM_BASE_URL', '') or ''
    chat_id = getattr(settings, 'TELE_CHAT_ID', '') or ''
    if not base_url or not chat_id:
        logger.debug("Telegram: Chưa cấu hình TELEGRAM_BASE_URL hoặc TELEGRAM_CHAT_ID")
        return

    url = f"{base_url.rstrip('/')}/sendPhoto"
    try:
        with open(image_path, "rb") as photo_file:
            files = {"photo": photo_file}
            data = {
                "chat_id": chat_id,
                "caption": caption,
                "parse_mode": "HTML"
            }
            response = requests.post(url, data=data, files=files, timeout=15)

        if response.status_code == 200:
            resp_json = response.json()
            if resp_json.get("ok"):
                logger.info("Telegram: Gửi hình ảnh thành công")
            else:
                logger.warning(f"Telegram bot error: {resp_json.get('description', 'Không rõ')}")
        else:
            logger.error(f"Telegram HTTP {response.status_code}: {response.text}")
    except FileNotFoundError:
        logger.error(f"Telegram: File not found: {image_path}, fallback to text message")
        send_message_telegram(caption)
    except Exception as e:
        logger.error(f"Telegram send error: {e}")

def send_media_group_telegram(image_paths: list, caption: str) -> None:
    import json
    base_url = getattr(settings, 'TELEGRAM_BASE_URL', '') or ''
    chat_id = getattr(settings, 'TELE_CHAT_ID', '') or ''
    if not base_url or not chat_id:
        logger.debug("Telegram: Chưa cấu hình TELEGRAM_BASE_URL hoặc TELEGRAM_CHAT_ID")
        return

    url = f"{base_url.rstrip('/')}/sendMediaGroup"
    
    media = []
    files = {}
    
    try:
        for idx, path in enumerate(image_paths):
            if not os.path.exists(path):
                continue
            
            file_key = f"img{idx}"
            files[file_key] = (os.path.basename(path), open(path, "rb"))
            
            media_item = {
                "type": "photo",
                "media": f"attach://{file_key}"
            }
            
            if idx == 0 and caption:
                media_item["caption"] = caption
                # Enable HTML parsing for bold text
                media_item["parse_mode"] = "HTML"
                
            media.append(media_item)

        if not media:
            logger.warning("Telegram: No valid image files for media group, fallback to sendMessage")
            send_message_telegram(caption)
            return

        data = {
            "chat_id": chat_id,
            "media": json.dumps(media)
        }

        response = requests.post(url, data=data, files=files, timeout=20)

        if response.status_code == 200:
            resp_json = response.json()
            if resp_json.get("ok"):
                logger.info("Telegram: Gửi album hình ảnh thành công")
            else:
                logger.warning(f"Telegram bot error: {resp_json.get('description', 'Không rõ')}")
                send_message_telegram(caption)
        else:
            logger.error(f"Telegram HTTP {response.status_code} khi gửi album: {response.text}")
            send_message_telegram(caption)
    except Exception as e:
        logger.error(f"Telegram album send error: {e}")
        send_message_telegram(caption)
    finally:
        for f in files.values():
            if hasattr(f[1], "close"):
                f[1].close()


class FrameCache:
    """Thread-safe in-memory frame cache with TTL.
    
    Lưu frame ở chất lượng gốc (numpy array) thay vì save xuống disk ở 20% quality.
    Tự động cleanup frame đã quá hạn để tránh memory leak.
    """
    
    def __init__(self, ttl=30):
        """
        Args:
            ttl: Time-to-live in seconds. Frames older than this will be cleaned up.
        """
        import threading
        self._cache = {}  # {frame_id: (frame_numpy, timestamp)}
        self._lock = threading.Lock()
        self._ttl = ttl
    
    def put(self, frame_id, frame):
        """Lưu frame vào cache.
        
        Args:
            frame_id: UUID string của frame
            frame: numpy array (BGR) - chất lượng gốc
        """
        import time
        with self._lock:
            self._cache[frame_id] = (frame.copy(), time.time())
            # Cleanup cũ nếu cache quá lớn
            if len(self._cache) > 100:
                self._cleanup_expired()
    
    def get(self, frame_id):
        """Lấy frame từ cache.
        
        Args:
            frame_id: UUID string của frame
            
        Returns:
            numpy array hoặc None nếu không tìm thấy hoặc đã hết hạn.
        """
        import time
        with self._lock:
            entry = self._cache.get(frame_id)
            if entry is None:
                return None
            frame, ts = entry
            if time.time() - ts > self._ttl:
                del self._cache[frame_id]
                return None
            return frame
    
    def remove(self, frame_id):
        """Xóa frame khỏi cache."""
        with self._lock:
            self._cache.pop(frame_id, None)
    
    def _cleanup_expired(self):
        """Xóa tất cả frame đã quá hạn (gọi khi đã giữ lock)."""
        import time
        now = time.time()
        expired = [fid for fid, (_, ts) in self._cache.items() if now - ts > self._ttl]
        for fid in expired:
            del self._cache[fid]
    
    def cleanup(self):
        """Public cleanup method - thread-safe."""
        with self._lock:
            self._cleanup_expired()
    
    def clear(self):
        """Xóa toàn bộ cache."""
        with self._lock:
            self._cache.clear()
    
    def __len__(self):
        with self._lock:
            return len(self._cache)


def extract_rule_config(cam_id: str, rules: list = None,
                        rule_types: tuple = (), param: dict = None) -> dict:
    """
    Trích xuất cấu hình luật từ /api/rules cho 1 camera cụ thể.

    Quy tắc ưu tiên:
      1) Tìm trong `rules` có status == 1, type khớp rule_types (nếu truyền),
         và stream_ids chứa cam_id HOẶC có filter chứa stream_id == cam_id.
         Lấy các filters:
           - 'region': toạ độ polygon ROI (zone)
           - 'direction': vector hướng di chuyển
           - 'line': vạch cắt (cũng gán cho direction nếu direction chưa có)
           - 'light_box': hộp toạ độ đèn tín hiệu giao thông (rlt)
           - 'object_type': danh sách class đối tượng (person, car, motorbike...)
         Lấy các thuộc tính luật: threshold, dwell, realert.
      2) Fallback tương thích ngược: nếu không tìm thấy trong rules hoặc trường còn thiếu,
         đọc từ `param` (cam.parameter).
    Trả về dict chuẩn:
      {
        "zone": list or None,
        "direction": list or None,
        "line": list or None,
        "light_box": list or None,
        "classes": list or None,
        "threshold": int/float or None,
        "dwell": float or None,
        "realert": float or None,
        "rule_found": bool,
        "rule_name": str or None,
        "rule_id": str or None,
        "rule_type": str or None,
      }
    """
    zone = None
    direction = None
    line = None
    light_box = None
    classes = None
    threshold = None
    dwell = None
    realert = None
    rule_found = False
    rule_name = None
    rule_id = None
    matched_type = None

    # Parse param nếu là chuỗi JSON
    if isinstance(param, str):
        try:
            param = json.loads(param)
        except Exception:
            param = {}
    if not isinstance(param, dict):
        param = {}

    norm_rule_types = tuple(str(t).lower().strip() for t in rule_types if str(t).strip())

    # 1. Tìm trong danh sách rules từ /api/rules
    if rules and isinstance(rules, list) and cam_id:
        for r in rules:
            if not isinstance(r, dict):
                continue
            if r.get('status') != 1:
                continue

            r_type = str(r.get('type', '')).lower().strip()
            if norm_rule_types and r_type not in norm_rule_types and '*' not in norm_rule_types:
                continue

            stream_ids = r.get('stream_ids') or []
            filters = r.get('filters') or []

            has_stream = (cam_id in stream_ids)
            if not has_stream:
                for f in filters:
                    if isinstance(f, dict) and f.get('stream_id') == cam_id:
                        has_stream = True
                        break

            if not has_stream:
                continue

            rule_found = True
            rule_name = r.get('name')
            rule_id = r.get('id')
            matched_type = r_type

            # Đọc các thuộc tính cấp rule nếu có
            if r.get('threshold') is not None:
                try:
                    threshold = float(r.get('threshold'))
                    if threshold.is_integer():
                        threshold = int(threshold)
                except (ValueError, TypeError):
                    pass
            if r.get('dwell') is not None:
                try:
                    dwell = float(r.get('dwell'))
                except (ValueError, TypeError):
                    pass
            if r.get('realert') is not None:
                try:
                    realert = float(r.get('realert'))
                except (ValueError, TypeError):
                    pass

            # Lặp qua các filters của rule
            for f in filters:
                if not isinstance(f, dict) or f.get('status') != 1:
                    continue
                fstream = f.get('stream_id')
                # Filter phải gán riêng cho camera này hoặc gán chung cho cả rule (stream_id is None / '')
                if fstream is not None and fstream != '' and fstream != cam_id:
                    continue

                ftype = str(f.get('type', '')).lower().strip()
                cond = f.get('condition')
                if isinstance(cond, str):
                    try:
                        cond = json.loads(cond)
                    except Exception:
                        pass

                if ftype in ('region', 'zone', 'roi'):
                    if isinstance(cond, list) and len(cond) >= 3:
                        zone = cond
                    elif isinstance(cond, list) and len(cond) == 0:
                        zone = []  # Vùng cả khung hình
                elif ftype == 'direction':
                    if isinstance(cond, list) and len(cond) >= 2:
                        direction = cond
                    elif cond is not None:
                        direction = cond
                elif ftype == 'line':
                    if isinstance(cond, list) and len(cond) >= 2:
                        line = cond
                        if direction is None:
                            direction = cond
                elif ftype == 'light_box':
                    if isinstance(cond, list) and len(cond) >= 2:
                        light_box = cond
                elif ftype == 'object_type':
                    if isinstance(cond, list) and cond:
                        classes = cond
                elif ftype in ('duration', 'dwell', 'stay_duration', 't_static', 'time'):
                    try:
                        dwell = float(cond)
                    except (ValueError, TypeError):
                        pass

            break  # Đã tìm thấy rule active phù hợp nhất
            
    # 2. Fallback tương thích ngược: lấy từ camera parameter nếu rule chưa có
    if not rule_found or zone is None or direction is None or dwell is None:
        # Tìm trong parameter.<service_type> trước
        param_zone = None
        param_direction = None
        param_line = None
        param_classes = None
        param_threshold = None
        param_dwell = None

        search_keys = norm_rule_types if norm_rule_types else ('license_plate', 'traffic', 'crowd', 'intrusion')
        for k in search_keys:
            sub = param.get(k)
            if isinstance(sub, str):
                try:
                    sub = json.loads(sub)
                except Exception:
                    sub = {}
            if isinstance(sub, dict):
                if param_zone is None:
                    param_zone = sub.get('zone') or sub.get('roi_zone')
                if param_direction is None:
                    param_direction = sub.get('direction')
                if param_line is None:
                    param_line = sub.get('line')
                if param_classes is None:
                    param_classes = sub.get('classes')
                if param_threshold is None:
                    param_threshold = sub.get('threshold')
                if param_dwell is None:
                    param_dwell = sub.get('duration') or sub.get('dwell') or sub.get('stay_duration')

        if param_zone is None:
            param_zone = param.get('zone') or param.get('roi_zone')
        if param_direction is None:
            param_direction = param.get('direction')
        if param_line is None:
            param_line = param.get('line')
        if param_classes is None:
            param_classes = param.get('classes')
        if param_threshold is None:
            param_threshold = param.get('threshold')
        if param_dwell is None:
            param_dwell = param.get('duration') or param.get('dwell') or param.get('stay_duration')

        # Parse string nếu cần
        for var_name, var_val in [('zone', param_zone), ('direction', param_direction),
                                  ('line', param_line), ('classes', param_classes)]:
            if isinstance(var_val, str):
                try:
                    var_val = json.loads(var_val)
                except Exception:
                    var_val = None
            if var_name == 'zone' and zone is None and var_val is not None:
                zone = var_val
            elif var_name == 'direction' and direction is None and var_val is not None:
                direction = var_val
            elif var_name == 'line' and line is None and var_val is not None:
                line = var_val
            elif var_name == 'classes' and classes is None and var_val is not None:
                classes = var_val

        if threshold is None and param_threshold is not None:
            try:
                threshold = float(param_threshold)
                if threshold.is_integer():
                    threshold = int(threshold)
            except (ValueError, TypeError):
                pass

        if dwell is None and param_dwell is not None:
            try:
                dwell = float(param_dwell)
            except (ValueError, TypeError):
                pass

    return {
        "zone": zone,
        "direction": direction,
        "line": line,
        "light_box": light_box,
        "classes": classes,
        "threshold": threshold,
        "dwell": dwell,
        "duration": dwell,
        "realert": realert,
        "rule_found": rule_found,
        "rule_name": rule_name,
        "rule_id": rule_id,
        "rule_type": matched_type,
    }

def extract_lpr_schedule(param=None, rules=None, cam_id=None):
    """
    Trích xuất cấu hình khung giờ ban đêm (lpr_schedule) của camera.
    Ưu tiên tìm trong danh sách rules (/api/rules), sau đó mới fallback sang camera parameter.
    
    Trả về tuple (timefrom, timeto) dạng chuỗi "HH:MM", hoặc None nếu không cấu hình.
    """
    # 1. Tìm trong danh sách rules (/api/rules)
    if rules and isinstance(rules, list) and cam_id:
        for r in rules:
            if not isinstance(r, dict) or r.get('status') != 1:
                continue
            r_type = str(r.get('type', '')).lower().strip()
            if r_type not in ('license_plate', 'lpr', 'plate', 'traffic'):
                continue
            stream_ids = r.get('stream_ids') or []
            has_stream = (cam_id in stream_ids)
            if not has_stream:
                for f in (r.get('filters') or []):
                    if isinstance(f, dict) and f.get('stream_id') == cam_id:
                        has_stream = True
                        break
            if not has_stream:
                continue

            # Thuộc tính lpr_schedule cấp rule
            sched = r.get('lpr_schedule')
            if isinstance(sched, str):
                try:
                    sched = json.loads(sched)
                except Exception:
                    sched = None
            if isinstance(sched, dict):
                t_from = sched.get('from') or sched.get('start') or sched.get('timefrom')
                t_to = sched.get('to') or sched.get('end') or sched.get('timeto')
                if t_from and t_to:
                    return str(t_from).strip(), str(t_to).strip()

            if r.get('time_from') and r.get('time_to'):
                return str(r.get('time_from')).strip(), str(r.get('time_to')).strip()
            if r.get('nightmodefrom') and r.get('nightmodeto'):
                return str(r.get('nightmodefrom')).strip(), str(r.get('nightmodeto')).strip()

            # Kiểm tra filters của rule
            for f in (r.get('filters') or []):
                if not isinstance(f, dict) or f.get('status') != 1:
                    continue
                fstream = f.get('stream_id')
                if fstream is not None and fstream != '' and fstream != cam_id:
                    continue
                ftype = str(f.get('type', '')).lower().strip()
                if ftype in ('schedule', 'lpr_schedule', 'time'):
                    cond = f.get('condition')
                    if isinstance(cond, str):
                        try:
                            cond = json.loads(cond)
                        except Exception:
                            cond = None
                    if isinstance(cond, dict):
                        t_from = cond.get('from') or cond.get('start') or cond.get('timefrom')
                        t_to = cond.get('to') or cond.get('end') or cond.get('timeto')
                        if t_from and t_to:
                            return str(t_from).strip(), str(t_to).strip()

    # 2. Fallback: camera parameter
    if isinstance(param, str):
        try:
            param = json.loads(param)
        except Exception:
            param = {}
    if not isinstance(param, dict):
        param = {}

    sched = param.get('lpr_schedule')
    if not sched and isinstance(param.get('license_plate'), dict):
        sched = param['license_plate'].get('lpr_schedule')
    if not sched and isinstance(param.get('license_plate_night'), dict):
        sched = param['license_plate_night'].get('lpr_schedule')

    if isinstance(sched, str):
        try:
            sched = json.loads(sched)
        except Exception:
            sched = None

    if isinstance(sched, dict):
        t_from = sched.get('from') or sched.get('start') or sched.get('timefrom')
        t_to = sched.get('to') or sched.get('end') or sched.get('timeto')
        if t_from and t_to:
            return str(t_from).strip(), str(t_to).strip()

    # Fallback trực tiếp nightmodefrom / nightmodeto
    t_from = param.get('nightmodefrom')
    t_to = param.get('nightmodeto')
    if t_from and t_to:
        return str(t_from).strip(), str(t_to).strip()

    return None


def is_nightmode(timefrom, timeto, tz_offset_hours: int = 7, now_time=None) -> bool:
    """
    Kiểm tra thời gian hiện tại (theo UTC+7) có nằm trong khung giờ nightmode không.
    - timefrom, timeto: chuỗi "HH:MM" (vd: "18:00", "06:00")
    - now_time: tùy chọn để test (datetime.time hoặc datetime.datetime)
    
    Trả về: True nếu đang trong khung giờ ban đêm, False nếu ban ngày hoặc format lỗi.
    """
    if not timefrom or not timeto:
        return False

    try:
        from datetime import datetime, timezone, timedelta

        if now_time is None:
            tz_vn = timezone(timedelta(hours=tz_offset_hours))
            now = datetime.now(tz_vn).time()
        elif hasattr(now_time, 'time'):
            now = now_time.time()
        else:
            now = now_time

        t_start = datetime.strptime(str(timefrom).strip(), "%H:%M").time()
        t_end = datetime.strptime(str(timeto).strip(), "%H:%M").time()

        if t_start < t_end:
            # Khung giờ cùng ngày (vd: 19:00 -> 23:00)
            return t_start <= now <= t_end
        else:
            # Khung giờ qua đêm (vd: 18:00 -> 06:00)
            return now >= t_start or now <= t_end

    except Exception as e:
        logger.error(f"Lỗi phân tích thời gian is_nightmode ({timefrom} - {timeto}): {e}")
        return False
