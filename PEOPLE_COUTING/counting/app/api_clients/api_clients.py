"""
api_clients.py — Unified CMS API client (identical across ALL services).

Every orchestrator service talks to the same CMS endpoints:
  GET    /api/cameras                — camera list (readers are built from this)
  GET    /api/cameras/{id}           — single camera
  PATCH  /api/cameras/{id}           — camera online status
  POST   /api/alarms                 — push alarm events
  GET    /api/events?type=           — pull event rule configs
  GET    /api/rules                  — pull per-camera rules (counting lines, light boxes…)
  POST   /api/vhc-detected-objects   — push detected vehicle/plate objects
  POST   /api/camera-statistics[/batch] — push counting statistics
  POST   {INSPECTION}/check          — expired-inspection lookup (LPR only)
"""
import logging
import time
from datetime import datetime, timedelta, timezone

import requests

from core.settings import settings

logger = logging.getLogger(__name__)

session = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    pool_connections=50,
    pool_maxsize=100,
    max_retries=requests.adapters.Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=[500, 502, 503, 504],
        raise_on_status=False
    )
)
session.mount('http://', adapter)
session.mount('https://', adapter)


def _validate_config():
    if not settings.BASE_URL:
        raise ValueError(
            "[CONFIG ERROR] Biến môi trường 'BASE_URL_API' chưa được thiết lập! "
            "Vui lòng cấu hình BASE_URL_API trong file .env (ví dụ: BASE_URL_API=http://<IP_OR_DOMAIN>:8080)"
        )


def _headers():
    _validate_config()
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "session": settings.SESSION_KEY,
    }


# ── cameras ────────────────────────────────────────────────────────────
def get_cameras():
    url = f"{settings.BASE_URL.rstrip('/')}/api/cameras"
    resp = session.get(url, headers=_headers(), verify=False, timeout=(10.0, 20.0))
    resp.raise_for_status()

    payload = resp.json()
    cameras = payload.get('data', [])
    if not isinstance(cameras, list):
        raise ValueError("API did not return a camera list in field 'data'")

    camera_list = []
    for cam in cameras:
        camera_list.append({
            'id': cam.get('id'),
            'name': cam.get('name'),
            'status': cam.get('status'),
            'url': cam.get('url'),
            'service_type': cam.get('service_type'),
            'parameter': cam.get('parameter'),
            'use_sdk': cam.get('use_sdk'),
            'storage_url': cam.get('storage_url'),
            'storage_port': cam.get('storage_port'),
            'storage_username': cam.get('storage_username'),
            'storage_password': cam.get('storage_password'),
            'storage_channel': cam.get('storage_channel'),
            'manufacturer': cam.get('manufacturer'),
            'process_type': cam.get('process_type'),
        })
    return camera_list


def get_cameras_id(id: str):
    url = f"{settings.BASE_URL.rstrip('/')}/api/cameras/{id}"
    resp = session.get(url, headers=_headers(), verify=False, timeout=(10.0, 20.0))
    resp.raise_for_status()
    return resp.json()


def patch_camera(stream_id: str, online: int):
    url = f"{settings.BASE_URL.rstrip('/')}/api/cameras/{stream_id}"
    resp = session.patch(url, json={"online": online}, headers=_headers(),
                         verify=False, timeout=(10.0, 15.0))
    logger.info(f"STATUS-PATCH-CAMERA: {resp.status_code}")
    resp.raise_for_status()
    return resp.json()


# ── alarms / events ───────────────────────────────────────────────────
def create_alarm(stream_id: str, type: str, source: str, attributes: dict = None):
    url = f"{settings.BASE_URL.rstrip('/')}/api/alarms"
    payload = {
        'stream_id': stream_id,
        'type': type,
        'source': source,
        'attributes': attributes or {},
        'time': datetime.now(timezone(timedelta(hours=7))).timestamp()  # UTC+7
    }
    resp = session.post(url, json=payload, headers=_headers(),
                        verify=False, timeout=(10.0, 20.0))
    logger.info(f"STATUS-CREATE-ALARM: {resp.status_code}")
    resp.raise_for_status()
    return resp.json()


def get_event(type: str):
    url = f"{settings.BASE_URL.rstrip('/')}/api/events"
    params = {"type": type}
    resp = session.get(url, headers=_headers(), params=params,
                       verify=False, timeout=(10.0, 20.0))
    logger.info(f"STATUS-GET-EVENT: {resp.status_code}")
    resp.raise_for_status()
    return resp.json().get("data", [])


def get_event_license_plate():
    return get_event("license_plate")



def get_rules(rule_type: str = None, stream_id: str = None):
    """Per-camera rules from /api/rules (counting lines, traffic-light boxes, zones …)."""
    url = f"{settings.BASE_URL.rstrip('/')}/api/rules"
    params = {}
    if rule_type:
        params["type"] = rule_type
    if stream_id:
        params["stream_id"] = stream_id
    resp = session.get(url, headers=_headers(), params=params, verify=False, timeout=(10.0, 20.0))
    logger.info(f"STATUS-GET-RULES: {resp.status_code}")
    resp.raise_for_status()
    payload = resp.json()
    return payload.get("data", []) if isinstance(payload, dict) else payload



# ── detected objects (LPR) ────────────────────────────────────────────
def create_det_object(object_id: str,
                      stream_id: str,
                      name: str,
                      text_plate: str,
                      frame_id: str,
                      bbox,
                      bbox_attr,
                      prob_vehicle: float,
                      prob_ocr: float,
                      tracks):
    url = f"{settings.BASE_URL.rstrip('/')}/api/vhc-detected-objects"
    payload = {
        "object_id": object_id,
        "stream_id": stream_id,
        "name": name,
        "frame_id": frame_id,
        "bbox": bbox,
        "attributes": [
            {
                "bbox": bbox_attr,
                "value": text_plate,
                "prob": prob_ocr
            }
        ],
        "prob": prob_vehicle,
        "tracks": tracks,
        "time": time.time(),
    }
    resp = session.post(url, json=payload, headers=_headers(),
                        verify=False, timeout=(10.0, 20.0))
    resp.raise_for_status()
    return resp.json()


# ── counting statistics ────────────────────────────────────────────────
def post_statistics(stream_id: str, count: int = 0, car: int = 0,
                    motorbike: int = 0, truck: int = 0, bus: int = 0,
                    metric_type: str = None, data: dict = None,
                    time_point: float = None):
    """POST counting statistics to CMS: /api/camera-statistics"""
    url = f"{settings.BASE_URL.rstrip('/')}/api/camera-statistics"
    m_type = metric_type or getattr(settings, 'METRIC_TYPE', 'people_counting')
    if data is not None:
        payload_data = data
    else:
        payload_data = {
            "count": count,
            "person": count,
            "car": car,
            "motorbike": motorbike,
            "truck": truck,
            "bus": bus,
        }
    payload = {
        "stream_id": stream_id,
        "metric_type": m_type,
        "data": payload_data,
        "time": time_point if time_point is not None else time.time(),
    }
    resp = session.post(url, json=payload, headers=_headers(),
                        verify=False, timeout=(10.0, 20.0))
    cnt = payload_data.get("count", count)
    logger.info(f"POST statistics for {stream_id} ({m_type}): status={resp.status_code} count={cnt}")
    resp.raise_for_status()
    return True



def post_statistics_batch(records: list):
    """POST batch counting statistics to CMS: /api/camera-statistics/batch"""
    url = f"{settings.BASE_URL.rstrip('/')}/api/camera-statistics/batch"
    payload = {"records": records}
    resp = session.post(url, json=payload, headers=_headers(),
                        verify=False, timeout=(10.0, 25.0))
    logger.info(f"POST batch statistics: status={resp.status_code} records={len(records)}")
    resp.raise_for_status()
    return resp.json()


# ── expired inspection (LPR only) ─────────────────────────────────────
def check_expired_inspection(text: str):
    url = f"{settings.BASE_URL_EXPIRED_INSPECTION.rstrip('/')}/check"
    payload = {'bien_so': text}
    resp = session.post(url, json=payload, headers={
        "Accept": "application/json",
        "Content-Type": "application/json",
    }, verify=False, timeout=(5.0, 30.0))
    resp.raise_for_status()
    return resp.json()
