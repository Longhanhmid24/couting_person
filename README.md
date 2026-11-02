# 🚶 Real-Time AI People Counting System (Hệ Thống Đếm Người Thời Gian Thực)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![NVIDIA CUDA](https://img.shields.io/badge/NVIDIA-CUDA_FP16_%2F_TensorRT-76B900?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/tensorrt)
[![gRPC](https://img.shields.io/badge/gRPC-Microservice-244c5a?style=flat&logo=grpc&logoColor=white)](https://grpc.io/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8_Person-00FFFF?style=flat)](https://ultralytics.com/)
[![Architecture](https://img.shields.io/badge/Architecture-Microservices-orange?style=flat)](#-3-kien-truc-he-thong--luong-xu-ly)

Hệ thống Core AI phân tích thị giác máy tính đếm lượt người ra/vào (**IN / OUT / TOTAL**) thời gian thực từ đa luồng camera RTSP (Dahua, Hikvision, Axis,...). Dự án xây dựng theo kiến trúc **Microservices chuẩn Core AI**: tách rời module nhận diện AI (**YOLO Person Detector gRPC**) và module điều phối luồng (**Counting Orchestrator**), tích hợp bám vết **ByteTrack 8D Kalman Filter**, tăng tốc suy luận **TensorRT 10**, và cơ chế **Pending Crossing Finite State Machine (FSM)** loại bỏ triệt để sai số đếm do rung lắc hoặc vật thể đứng yên.

---

## 📑 Mục Lục
- [1. Tính Năng Nổi Bật](#-1-tinh-nang-noi-bat)
- [2. Kiến Trúc Hệ Thống & Luồng Xử Lý](#-2-kien-truc-he-thong--luong-xu-ly)
- [3. Cấu Trúc Repository](#-3-cau-truc-repository)
- [4. Cơ Chế Đếm & Pending Crossing FSM](#-4-co-che-dem--pending-crossing-fsm)
- [5. Cấu Hình Vạch Cắt & Quy Tắc Trên CMS](#-5-cau-hinh-vach-cat--quy-tac-tren-cms)
- [6. Hướng Dẫn Cài Đặt & Triển Khai](#-6-huong-dan-cai-dat--trien-khai)
  - [Yêu cầu hệ thống](#yeu-cau-he-thong)
  - [Triển khai với Docker Compose (Khuyến nghị)](#trien-khai-voi-docker-compose-production)
  - [Chạy trực tiếp Local (Debugging)](#chay-truc-tiep-local-debugging)
- [7. Bảng Chi Tiết Biến Môi Trường (.env)](#-7-bang-chi-tiet-bien-moi-truong-env)
- [8. Đặc Tả Dữ Liệu Đẩy Lên CMS Backend](#-8-dac-ta-du-lieu-day-len-cms-backend)
- [9. Xử Lý Sự Cố Thường Gặp (Troubleshooting)](#-9-xu-ly-su-co-thuong-gap-troubleshooting)

---

## 🚀 1. Tính Năng Nổi Bật

- 🎯 **Kiến trúc Microservices hiệu năng cao**: Tách biệt gRPC Detector Server và Multi-stream Counting Orchestrator, dễ dàng scale độc lập.
- ⚡ **Dynamic Batching & TensorRT 10**: Detector gom batch động thông minh (`BATCH_SIZE=8`, `BATCH_TIMEOUT_MS=10`), giảm tối đa độ trễ suy luận, hỗ trợ tăng tốc GPU TensorRT FP16.
- 📹 **Giải mã phần cứng NVDEC GPU**: Hỗ trợ giải mã luồng RTSP trực tiếp trên GPU NVIDIA với cơ chế tự động fallback sang CPU khi GPU quá tải.
- 🛡️ **Lọc chuyển động GPU Motion Detector (CUDA FP16)**: Phát hiện chuyển động in-place trên GPU trước khi kích hoạt deep model, tiết kiệm từ 60-80% tài nguyên tính toán.
- 🔍 **Bám vết thông minh ByteTrack + TrackStitcher**: Khắc phục tình trạng mất dấu người đi bộ do che khuất tạm thời hoặc thay đổi góc nhìn.
- 📐 **Pending Crossing FSM độc quyền**:
  - Không nuốt sự kiện khi người xuất hiện sát vạch đếm.
  - Phân định hướng di chuyển (IN/OUT) chuẩn xác bằng tích vô hướng vector quỹ đạo (*Trajectory Dot Product*) và kiểm tra CCW.
  - Triệt tiêu hoàn toàn báo ảo (*0 False Alarm*) từ xe máy hoặc vật thể đỗ yên gần vạch.
- 🌐 **Tự động đồng bộ với CMS Backend**: Polling danh sách camera theo chu kỳ, tự động cập nhật tọa độ vạch ảo và đẩy dữ liệu thống kê đếm thời gian thực.
- 📸 **Lưu trữ bằng chứng & Tích hợp Telegram**: Tự động crop ảnh sự kiện chất lượng cao, quản lý vòng đời file ảnh theo ngày (`IMAGE_RETENTION_DAYS`), hỗ trợ gửi thông báo Telegram tức thì.

---

## 🏗️ 2. Kiến Trúc Hệ Thống & Luồng Xử Lý

```
                     ┌────────────────────────────────────────────────────────┐
                     │           Camera RTSP / Dahua / Hikvision              │
                     └──────────────────────────┬─────────────────────────────┘
                                                │ (RTSP Stream)
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Counting Orchestrator (Container: grpc-main-people-counting)                                │
│                                                                                             │
│  ┌──────────────────────┐        ┌──────────────────────┐        ┌───────────────────────┐  │
│  │   CameraReader       │  Frame │  GPU Motion Detector │ Passed │ CentralPipeline       │  │
│  │  (NVDEC GPU / CPU)   ├───────►│  (CUDA FP16 Filters) ├───────►│ Dispatcher Queue     │  │
│  └──────────────────────┘        └──────────────────────┘        └───────────┬───────────┘  │
└──────────────────────────────────────────────────────────────────────────────┼──────────────┘
                                                                               │ gRPC Frame Request
                                                                               │ (:50060)
┌──────────────────────────────────────────────────────────────────────────────┼──────────────┐
│ Person Detector Microservice (Container: grpc_person_detector)               ▼              │
│  ┌───────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ gRPC YoloServiceServicer                                                              │  │
│  │   - Dynamic Batch Queue (Worker Thread, Batch 8, 10ms window)                         │  │
│  │   - YOLOv8 Person Model (TensorRT FP16 / PyTorch CUDA)                                │  │
│  └───────────────────────────────────────────────────────────────────────────┬───────────┘  │
└──────────────────────────────────────────────────────────────────────────────┼──────────────┘
                                                                               │ Bounding Boxes
                                                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Counting Orchestrator (tiếp tục)                                                            │
│                                                                                             │
│  ┌──────────────────────┐        ┌──────────────────────┐        ┌───────────────────────┐  │
│  │ ByteTrack (8D Kalman)├───────►│ Pending Crossing FSM ├───────►│ Direction Classifier  │  │
│  │ + TrackStitcher      │ Tracks │ (Hysteresis Buffer)  │ Events │ (IN / OUT / COUNT)    │  │
│  └──────────────────────┘        └──────────────────────┘        └───────────┬───────────┘  │
│                                                                              │              │
│                                      ┌───────────────────────────────────────┴──────────┐   │
│                                      ▼                                                  ▼   │
│                       ┌─────────────────────────────┐            ┌──────────────────────┐   │
│                       │   CMS Statistics Pusher     │            │ Evidence Snapshot    │   │
│                       │ POST /api/camera-statistics │            │ & Telegram Notifier  │   │
│                       └──────────────┬──────────────┘            └──────────────────────┘   │
└──────────────────────────────────────┼──────────────────────────────────────────────────────┘
                                       ▼
                       ┌─────────────────────────────┐
                       │   CMS Backend & Dashboard   │
                       └─────────────────────────────┘
```

---

## 📂 3. Cấu Trúc Repository

```
couting_person/
├── .env.example                 # File cấu hình môi trường mẫu (trong PEOPLE_COUTING/)
├── core_ai_integration_guide.md # Hướng dẫn đặc tả tích hợp API với CMS Backend
├── README.md                    # Tài liệu tổng quan dự án trên GitHub
│
├── PEOPLE_COUTING/              # Gói triển khai hoàn chỉnh dịch vụ Đếm Người
│   ├── docker-compose.yml       # File khởi chạy đồng thời cả 2 microservices
│   ├── .env.example             # Biến môi trường mẫu cho Compose
│   ├── PLAN_PENDING_CROSSING_FSM.md # Kế hoạch & chi tiết thuật toán FSM
│   ├── README.md                # Tài liệu chi tiết nội bộ
│   │
│   ├── detector/                # [MICROSERVICE] gRPC YOLO Person Detector
│   │   ├── Dockerfile           # Image build cho Detector
│   │   ├── requirements.txt     # Dependencies (torch, ultralytics, grpcio,...)
│   │   └── app/                 # Server gRPC, Dynamic Batching, YOLO Models
│   │
│   └── counting/                # [MICROSERVICE] Multi-camera Counting Orchestrator
│       ├── Dockerfile           # Image build cho Orchestrator
│       ├── requirements.txt     # Dependencies (opencv, shapely, requests,...)
│       └── app/                 # CentralEngine, CameraReader, ByteTrack, FSM, CMS API
│
└── GRPC_DET_PERSON/             # Microservice gRPC Detector độc lập
    ├── Dockerfile
    ├── docker-compose.yml
    ├── requirements.txt
    └── app/
```

---

## 🧠 4. Cơ Chế Đếm & Pending Crossing FSM

Dịch vụ áp dụng thuật toán **Finite State Machine (FSM)** với bộ đệm trạng thái mở rộng nhằm giải quyết triệt để các bài toán thực tế:

```
                  ┌──────────────────────┐
                  │ Track Mới Xuất Hiện  │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  Tính Signed Distance│
                  │  tới Vạch Đếm        │
                  └──────────┬───────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   |Distance| <= Buffer/2             |Distance| > Buffer/2
   (Trong vùng vạch - Transit)        (Bên ngoài vạch - Side +1 / -1)
            │                                 │
            │                         Cắt qua vạch?
            │                         (Đổi dấu Side)
            │                                 │
            └───────────────┬─────────────────┘
                            ▼
               ┌────────────────────────┐
               │ Đặt `pending_event`    │
               │ Xác định hướng IN/OUT  │
               └────────────┬───────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
    Dịch chuyển < 35px             Dịch chuyển >= 35px
    (Người mới bước sát vạch       (Hoàn thành hành trình)
     hoặc Xe máy rung lắc)                  │
            │                               ▼
            │ Giữ pending event       ┌────────────────────────┐
            │ Chờ frame tiếp theo     │ Phát sự kiện ĐẾM       │
            └───────────────────────► │ Chụp ảnh bằng chứng     │
                                      │ Kích hoạt Cooldown     │
                                      └────────────────────────┘
```

### Điểm Vượt Trội của Pending Crossing FSM:
1. **Chống nuốt sự kiện**: Khi người đi bộ xuất hiện cách vạch chỉ $20 - 30\text{px}$, FSM không vội vàng đánh giá trượt mà chuyển sang trạng thái `pending_event`. Khi người đó bước thêm 1–2 bước tiếp theo ($net\_disp \ge 35\text{px}$), sự kiện đếm được kích hoạt ngay lập tức.
2. **Khử báo ảo từ xe cộ đỗ yên (0 False Count)**: Xe máy đỗ gần vạch có bounding box rung lắc trong phạm vi $5 - 15\text{px}$. Do không bao giờ đạt ngưỡng chuyển dịch $35\text{px}$, `pending_event` sẽ tự động hủy bỏ khi track kết thúc.
3. **Phân loại IN / OUT chuẩn xác**: Dựa trên tích vô hướng giữa vector quỹ đạo tổng thể $(P_{\text{current}} - P_{\text{first}})$ và vector định hướng của vạch cấu hình, miễn nhiễm với rung lắc một khung hình.
4. **Mở rộng biên hình chiếu (Projection [-0.10, 1.10])**: Đảm bảo đếm chính xác cả những người đi sát tường hoặc rìa mép cầu thang.

---

## 📍 5. Cấu Hình Vạch Cắt & Quy Tắc Trên CMS

Camera được hệ thống tự động đồng bộ khi `service_type` trên CMS chứa `"people_counting"` (hoặc `"people_couting"`).

Quy tắc vạch cắt được cấu hình tại mục **Rule** trên CMS hoặc trong `parameter.line` và `parameter.direction`:

```json
{
  "parameter": {
    "line": [
      {"x": 0.15, "y": 0.60},
      {"x": 0.85, "y": 0.60}
    ],
    "direction": [
      {"x": 0.50, "y": 0.40},
      {"x": 0.50, "y": 0.80}
    ]
  }
}
```

- `line`: Tọa độ 2 điểm mút xác định đường cắt ảo.
- `direction`: Vector chỉ hướng từ điểm đầu đến điểm cuối. 
  - Người di chuyển **cùng chiều** với vector `direction` được tính là **IN** (`enter`).
  - Người di chuyển **ngược chiều** với vector `direction` được tính là **OUT** (`exit`).
- *Hỗ trợ linh hoạt cả tọa độ chuẩn hóa $[0.0 - 1.0]$ lẫn tọa độ pixel tuyệt đối.*

---

## 🛠️ 6. Hướng Dẫn Cài Đặt & Triển Khai

### Yêu Cầu Hệ Thống
- **Hệ điều hành**: Ubuntu 20.04 / 22.04 LTS (x86_64)
- **NVIDIA Driver**: Phiên bản $\ge 535$
- **Docker**: Docker Engine $\ge 24.0$ & Docker Compose v2
- **NVIDIA Container Toolkit**: Đã cấu hình runtime `nvidia` mặc định

### Triển Khai Với Docker Compose (Production)

#### Bước 1: Chuẩn bị file cấu hình môi trường
```bash
cd PEOPLE_COUTING
cp .env.example .env
nano .env
```
Điền tối thiểu thông tin máy chủ CMS:
```bash
BASE_URL_API=http://<IP_HOAC_DOMAIN_CMS>:8080
SESSION_KEY=<YOUR_CMS_SESSION_KEY>
```

#### Bước 2: Khởi chạy toàn bộ hệ thống
```bash
# Build và chạy ngầm toàn bộ dịch vụ
docker compose up -d --build
```

#### Bước 3: Kiểm tra logs hoạt động
```bash
# Xem log tổng hợp
docker compose logs -f

# Hoặc xem chi tiết log của bộ đếm (Counting)
docker compose logs -f counting

# Xem log của bộ nhận diện AI (Detector)
docker compose logs -f detector
```

#### Quản lý dịch vụ:
```bash
# Dừng hệ thống
docker compose down

# Khởi động lại
docker compose restart
```

---

### Chạy Trực Tiếp Local (Debugging)

Dành cho nhà phát triển muốn chạy debug trực tiếp từng service bằng Python:

#### 1. Chạy Detector Service (Terminal 1)
```bash
cd PEOPLE_COUTING/detector
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cd app
python3 server.py
```
*Server gRPC sẽ lắng nghe tại cổng `localhost:50060`.*

#### 2. Chạy Counting Service (Terminal 2)
```bash
cd PEOPLE_COUTING/counting
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cd app
export BASE_URL_API="http://192.168.1.100:8080"
export SESSION_KEY="your-session-key"
export YOLO_PERSON_ADDR="localhost:50060"
python3 main.py
```

---

## ⚙️ 7. Bảng Chi Tiết Biến Môi Trường (.env)

| Nhóm | Biến Môi Trường | Mặc định | Mô tả |
| :--- | :--- | :--- | :--- |
| **CMS Backend** | `BASE_URL_API` | *Bắt buộc* | URL CMS Backend (VD: `http://192.168.1.100:8080`) |
| | `SESSION_KEY` | *Bắt buộc* | Session Token dùng để xác thực các API CMS |
| | `SERVICE_TYPES` | `people_counting` | Lọc các camera có service tương ứng |
| | `METRIC_TYPE` | `people_counting` | Tên metric gửi dữ liệu lên CMS |
| | `STATISTICS_POST_INTERVAL` | `5` | Chu kỳ đẩy số liệu đếm lên CMS (giây) |
| | `STATS_LOG_INTERVAL` | `60` | Chu kỳ in log tổng kết ra màn hình (giây) |
| **Detector AI** | `DETECTOR_PORT_HOST` | `50061` | Cổng host map ra ngoài cho gRPC Detector |
| | `YOLO_PERSON_ADDR` | `detector:50060` | Địa chỉ gRPC kết nối Detector |
| | `USE_TENSORRT` | `false` | Bật suy luận TensorRT engine (.engine) |
| | `CONF_THRESHOLD` | `0.10` | Ngưỡng tin cậy detection của YOLO Server |
| | `CONFIDENT_PERSON` | `0.30` | Ngưỡng lọc phát hiện người của Orchestrator |
| | `BATCH_SIZE` | `8` | Kích thước gom batch động tối đa |
| | `BATCH_TIMEOUT_MS` | `10` | Thời gian chờ tối đa gom batch (mili-giây) |
| **Pipeline & GPU** | `GPU_ID` | `0` | GPU xử lý giải mã và tracking |
| | `FORCE_CPU_DECODE` | `False` | Bắt buộc giải mã luồng bằng CPU |
| | `USE_MOTION_DETECTOR` | `False` | Bật bộ lọc phát hiện chuyển động GPU FP16 |
| | `TARGET_FPS` | `25` | Tốc độ đọc frame từ camera |
| | `DETECT_FPS` | `10` | Tần số gửi frame qua AI detector |
| **Tracking & FSM** | `USE_BYTETRACK` | `True` | Kích hoạt bộ bám vết ByteTrack |
| | `LINE_BUFFER_PIXELS` | `35.0` | Độ dày vùng đệm quanh vạch ảo (px) |
| | `STATIONARY_DISPLACEMENT_MAX`| `40.0` | Ngưỡng dịch chuyển tối đa coi là đứng yên |
| | `MIN_FRAMES_BEFORE_COUNT` | `8` | Số frame tối thiểu track tồn tại trước khi đếm |
| | `MIN_PATH_MOVEMENT_PIXELS` | `40.0` | Quãng đường tối thiểu track phải di chuyển |
| | `MIN_ENTRY_DISTANCE_LINE` | `35.0` | Khoảng cách tối thiểu từ điểm bắt đầu tới vạch |
| | `COUNT_COOLDOWN_SECONDS` | `1.8` | Thời gian cooldown chống đếm lặp (giây) |
| **Hình Ảnh & Log** | `SHARED_IMAGE_DIR` | `/mnt/data/alarm`| Thư mục lưu ảnh snapshot trên host |
| | `ENABLE_SAVE_IMAGE` | `True` | Cho phép lưu ảnh bằng chứng khi cắt vạch |
| | `SAVE_PERSON_CROP` | `True` | Lưu ảnh crop riêng đối tượng người đếm |
| | `IMAGE_RETENTION_DAYS` | `30` | Tự động dọn dẹp ảnh cũ quá số ngày chỉ định |
| | `TELEGRAM_BASE_URL` | `""` | Base URL Telegram Bot gửi cảnh báo |
| | `TELE_CHAT_ID` | `""` | Chat ID nhận thông báo Telegram |

---

## 📡 8. Đặc Tả Dữ Liệu Đẩy Lên CMS Backend

Hệ thống gửi dữ liệu thống kê tổng hợp định kỳ thông qua REST API của CMS:

- **Endpoint**: `POST /api/camera-statistics`
- **Headers**:
  ```http
  Content-Type: application/json
  session: <SESSION_KEY>
  ```
- **Payload Đếm Người**:
  ```json
  {
    "stream_id": "9a9e21b4-1dbd-4457-a6ff-f7b29779383f",
    "metric_type": "people_counting",
    "data": {
      "count": 42,
      "enter": 23,
      "exit": 19,
      "person": 42
    },
    "time": 1774584288.0
  }
  ```

| Trường dữ liệu | Ý nghĩa |
| :--- | :--- |
| `stream_id` | UUID định danh của Camera trên CMS |
| `metric_type` | Loại chỉ số thống kê (`people_counting`) |
| `data.enter` | Số lượt người đi **Vào** (cùng hướng vector cấu hình) |
| `data.exit` | Số lượt người đi **Ra** (ngược hướng vector cấu hình) |
| `data.count` | Tổng số lượt người đã vượt vạch (`enter + exit`) |
| `data.person` | Số lượng người ghi nhận |
| `time` | Dấu thời gian Unix timestamp (giây) |

---

## ❓ 9. Xử Lý Sự Cố Thường Gặp (Troubleshooting)

### 1. Lỗi kết nối gRPC (`StatusCode.UNAVAILABLE: failed to connect to all addresses`)
- **Nguyên nhân**: Service `counting` khởi động trước khi service `detector` sẵn sàng lắng nghe ở port 50060, hoặc sai `YOLO_PERSON_ADDR`.
- **Khắc phục**: Khi chạy docker-compose, `depends_on` đã được cấu hình sẵn. Nếu chạy ngoài host, hãy đảm bảo chạy `python3 server.py` của detector trước, và kiểm tra `YOLO_PERSON_ADDR=localhost:50061` (hoặc `50060`).

### 2. Camera không đếm người dù có người đi qua
- **Nguyên nhân 1**: Tọa độ vạch cắt (`line`) hoặc vector hướng (`direction`) chưa được thiết lập trên CMS.
- **Nguyên nhân 2**: `MIN_PATH_MOVEMENT_PIXELS` đặt quá lớn so với khoảng cách người di chuyển trong góc quay.
- **Khắc phục**: Kiểm tra log console `docker compose logs -f counting` để xem camera đã nhận rule vạch cắt chưa (`Rules loaded: ...`). Giảm `MIN_PATH_MOVEMENT_PIXELS=20.0` nếu camera đặt góc nhìn quá xa/nhỏ.

### 3. Xe máy đỗ yên trước sảnh bị đếm nhầm thành người
- **Khắc phục**: Thuật toán `Pending Crossing FSM` đã có ngưỡng lọc `STATIONARY_DISPLACEMENT_MAX=40.0`. Hãy đảm bảo không tắt FSM và giữ `CONFIDENT_PERSON >= 0.30` để loại bỏ các nhận diện sai từ đồ vật tĩnh.

### 4. GPU Out of Memory (OOM)
- **Khắc phục**:
  - Giảm `BATCH_SIZE` trong file `.env` xuống `4`.
  - Giảm số lượng camera xử lý đồng thời trên 1 card GPU.
  - Sử dụng TensorRT FP16 (`USE_TENSORRT=true`) để tối ưu dung lượng VRAM.

---

## 👥 Tác Giả & Đóng Góp
- **Phát triển bởi**: Đội ngũ Core AI Vision
- **Liên hệ**: `vophilong069@gmail.com`
- **Mã nguồn**: [Longhanhmid24/couting_person](https://github.com/Longhanhmid24/couting_person.git)
