# 🚶 People Counting Service (Dịch vụ Đếm Người Vượt Vạch)

Dịch vụ đếm người thời gian thực theo kiến trúc Microservice chuẩn Core AI:
- **Detector Microservice (`detector/`)**: Chạy YOLOv8 Person Detection qua gRPC trên **Port 50060**, hỗ trợ Dynamic Batching và tăng tốc GPU TensorRT 10.
- **Counting Orchestrator Microservice (`counting/`)**: Quản lý đa luồng camera RTSP/Dahua/Hikvision, lọc chuyển động GPU FP16, bám vết quỹ đạo (IoU + Distance + HOG), phát hiện cắt vạch ảo (Line Crossing theo Direction Vector tính toán IN/OUT) và đẩy thống kê lên CMS API định kỳ.

---

## 1. Luồng Hoạt Động (Architecture Flow)

```
Camera RTSP/SDK → CameraReader (NVDEC GPU / CPU Fallback)
    → GPUMotionDetector (CUDA FP16 in-place)
    → CentralPipeline Dispatcher
    → YOLO Person Detection via gRPC (:50060, Dynamic Batching)
    → Tracking (IoU + Predicted Distance + HOG Interpolation)
    → Line Crossing Detection (CCW Test + Direction Vector Dot Product)
    → Phân loại: Người Vào (enter) / Người Ra (exit) / Tổng (count)
    → StatisticsPoster → POST /api/camera-statistics
```

---

## 2. Cấu Hình Camera & Vạch Cắt trên CMS

Camera được tự động đồng bộ và kích hoạt khi:
1. `service_type` trên CMS chứa `"people_counting"` (hoặc `"people_couting"`).
2. Quy tắc vạch cắt được cấu hình tại `/api/rules` hoặc `parameter.line` và `parameter.direction`.

### Định dạng Line Crossing:
```json
{
  "parameter": {
    "line": [
      {"x": 0.1, "y": 0.5},
      {"x": 0.9, "y": 0.5}
    ],
    "direction": [
      {"x": 0.5, "y": 0.3},
      {"x": 0.5, "y": 0.7}
    ]
  }
}
```
*Tọa độ hỗ trợ cả chuẩn hóa (0.0 - 1.0) và pixel tuyệt đối.*

---

## 3. Biến Môi Trường (Environment Variables)

Hệ thống sử dụng file `.env` để bảo mật và quản lý tập trung toàn bộ cấu hình. Khi chuyển sang máy chủ chính thức với backend CMS khác, **chỉ cần cập nhật file `.env`**.

### Bắt buộc cấu hình (CMS Backend):
| Biến | Ví dụ giá trị | Mô tả |
|---|---|---|
| `BASE_URL_API` | `http://<IP_OR_DOMAIN>:8080` | URL máy chủ CMS Backend (hỗ trợ linh hoạt http, https, domain, IP, port, trailing slash) |
| `SESSION_KEY` | `<YOUR_SESSION_KEY>` | Khóa phiên / Token xác thực API CMS |
| `SERVICE_TYPES` | `people_counting,people_couting` | Lọc các camera có kiểu dịch vụ tương ứng trên CMS |
| `METRIC_TYPE` | `people_counting` | Loại metric đẩy lên CMS |

### Tùy chọn nâng cao (Detector & Performance):
| Biến | Mặc định | Mô tả |
|---|---|---|
| `DETECTOR_PORT_HOST` | `50061` | Cổng host map ra ngoài cho gRPC Person Detector |
| `YOLO_PERSON_ADDR` | `detector:50060` | Địa chỉ gRPC kết nối Detector |
| `USE_TENSORRT` | `false` | Bật/tắt suy luận bằng TensorRT FP16 |
| `CONF_THRESHOLD` | `0.35` | Ngưỡng tin cậy phát hiện người của YOLO |
| `CONFIDENT_PERSON` | `0.35` | Ngưỡng lọc phát hiện người trong orchestrator |
| `STATISTICS_POST_INTERVAL` | `5` | Chu kỳ gửi số liệu thống kê lên CMS (giây) |
| `STATS_LOG_INTERVAL` | `60` | Chu kỳ in log thống kê chi tiết ra màn hình (giây) |
| `TARGET_FPS` | `25` | FPS đọc luồng camera |
| `DETECT_FPS` | `10` | Tần số gửi frame qua YOLO detect người |
| `GPU_ID` | `0` | GPU xử lý giải mã và tracking |
| `FORCE_CPU_DECODE` | `False` | Bắt buộc giải mã bằng CPU |
| `TELEGRAM_BASE_URL` | `""` | URL Telegram bot (nếu dùng thông báo) |
| `TELE_CHAT_ID` | `""` | Telegram Chat ID nhận thông báo |

---

## 4. Dữ Liệu Đẩy Lên CMS (`/api/camera-statistics`)

POST định kỳ lên CMS endpoint `/api/camera-statistics`:
```json
{
  "stream_id": "9a9e21b4-1dbd-4457-a6ff-f7b29779383f",
  "metric_type": "people_counting",
  "data": {
    "count": 40,
    "enter": 19,
    "exit": 21,
    "person": 40
  },
  "time": 1774584288.0
}
```
- `enter`: Số người đi theo hướng vạch (Người Vào).
- `exit`: Số người đi ngược hướng vạch (Người Ra).
- `count`: Tổng số người (`count = enter + exit`).

---

## 5. Hướng Dẫn Triển Khai (Deployment Guide)

### Bước 1: Chuẩn bị file cấu hình môi trường `.env`
Sao chép từ file mẫu và điền thông tin máy chủ CMS:
```bash
cd PEOPLE_COUTING
cp .env.example .env
nano .env
```
Cập nhật 2 thông số chính:
```bash
BASE_URL_API=http://<IP_HOAC_DOMAIN_CMS_CUA_BAN>:8080
SESSION_KEY=<SESSION_KEY_CUA_BAN>
```

### Bước 2: Khởi chạy toàn bộ dịch vụ bằng Docker Compose
```bash
docker compose up -d --build
```

### Bước 3: Kiểm tra nhật ký hoạt động (Logs)
```bash
# Xem log toàn bộ hệ thống
docker compose logs -f

# Hoặc chỉ xem log service counting
docker compose logs -f counting
```

---

## 6. Chạy Kiểm Thử Trực Tiếp (Local Python Debugging)

Khi phát triển hoặc kiểm tra trực tiếp mà không qua Docker:

**1. Khởi chạy Detector Server (Terminal 1):**
```bash
cd PEOPLE_COUTING/detector/app
python3 server.py
```

**2. Khởi chạy Counting Orchestrator (Terminal 2):**
```bash
cd PEOPLE_COUTING/counting/app
export BASE_URL_API="http://<IP_HOAC_DOMAIN_CMS>:8080"
export SESSION_KEY="<YOUR_SESSION_KEY>"
export YOLO_PERSON_ADDR="localhost:50060"
python3 main.py
```
