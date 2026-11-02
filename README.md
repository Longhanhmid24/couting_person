# 🚶 Real-Time AI People Counting System (Hệ Thống Đếm Người Thời Gian Thực)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Containers-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![NVIDIA CUDA](https://img.shields.io/badge/NVIDIA-CUDA_FP16_%2F_TensorRT-76B900?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/tensorrt)
[![gRPC](https://img.shields.io/badge/gRPC-Microservice-244c5a?style=flat&logo=grpc&logoColor=white)](https://grpc.io/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8_Person-00FFFF?style=flat)](https://ultralytics.com/)
[![Architecture](https://img.shields.io/badge/Architecture-Microservices-orange?style=flat)](#-2-kien-truc-he-thong-tong-the)

Hệ thống Core AI phân tích thị giác máy tính đếm lượt người ra/vào (**IN / OUT / TOTAL**) thời gian thực từ đa luồng camera giám sát an ninh (RTSP, Dahua, Hikvision, Axis...). Hệ thống được thiết kế theo kiến trúc **Microservices** chuyên biệt cho các bài toán Computer Vision công nghiệp, tối ưu hóa xử lý song song trên GPU và xử lý hiệu quả các tình huống thực tế phức tạp.

---

## 📑 Mục Lục

- [1. Tính Năng Nổi Bật](#-1-tinh-nang-noi-bat)
- [2. Kiến Trúc Hệ Thống Tổng Thể](#-2-kien-truc-he-thong-tong-the)
- [3. Ngăn Xếp Công Nghệ (Tech Stack)](#-3-ngan-xep-cong-nghe-tech-stack)
- [4. Cấu Trúc Mã Nguồn (Project Structure)](#-4-cau-truc-ma-nguon-project-structure)
- [5. Nguyên Lý & Giải Pháp Kỹ Thuật Cốt Lõi](#-5-nguyen-ly--giai-phap-ky-thuat-cot-loi)
  - [Nhận diện & Tối ưu hóa suy luận GPU](#nhan-dien--toi-uu-hoa-suy-luan-gpu)
  - [Bám vết quỹ đạo đa đối tượng (Multi-Object Tracking)](#bam-vet-quy-dao-da-doi-tuong-multi-object-tracking)
  - [Xác thực vượt vạch ảo & Phân định chiều di chuyển](#xac-thuc-vuot-vach-ao--phan-dinh-chieu-di-chuyen)
  - [Cơ chế khử nhiễu & Lọc báo ảo thông minh](#co-che-khu-nhieu--loc-bao-ao-thong-minh)
- [6. Khả Năng Tích Hợp & Dữ Liệu Đầu Ra](#-6-kha-nang-tich-hop--du-lieu-dau-ra)

---

## 🚀 1. Tính Năng Nổi Bật

- 🎯 **Xử lý đa luồng camera đồng thời**: Khả năng phân tích song song nhiều luồng video RTSP với độ trễ cực thấp.
- ⚡ **Tăng tốc suy luận phần cứng**: Tận dụng giải mã phần cứng GPU (NVDEC) kết hợp suy luận mô hình tăng tốc qua TensorRT và Dynamic Batching, tối ưu hóa hiệu suất GPU.
- 🔍 **Bám vết quỹ đạo mượt mà**: Theo dõi liên tục từng đối tượng trong khung hình, có khả năng phục hồi quỹ đạo khi đối tượng bị che khuất tạm thời.
- 📐 **Đếm chính xác theo chiều di chuyển**: Tự động phân định chiều Vào (`IN`) và Ra (`OUT`) dựa trên quan hệ hình học giữa quỹ đạo chuyển động và vạch ảo được thiết lập.
- 🛡️ **Khử báo sai thông minh**: Loại bỏ hiệu quả các hiện tượng nhảy số đếm do rung lắc camera, người đứng sát vạch, hoặc các vật thể tĩnh (phương tiện đỗ yên) trong góc quay.
- 📸 **Thu thập bằng chứng tự động**: Hỗ trợ lưu trữ ảnh chụp sự kiện phục vụ hậu kiểm và trực quan hóa dữ liệu.
- 🌐 **Sẵn sàng tích hợp hệ thống**: Dễ dàng kết nối với các hệ thống quản trị tập trung (CMS/VMS) thông qua API tiêu chuẩn.

---

## 🏗️ 2. Kiến Trúc Hệ Thống Tổng Thể

Hệ thống được module hóa theo kiến trúc Microservices phân tách rõ ràng giữa tầng suy luận AI và tầng điều phối nghiệp vụ:

```
┌────────────────────────────────────────────────────────┐
│             Nguồn Video (IP Camera / RTSP)            │
└──────────────────────────┬─────────────────────────────┘
                           │ Luồng Video
                           ▼
┌────────────────────────────────────────────────────────┐
│               Tầng Thu Nhận & Tiền Xử Lý               │
│     - Giải mã phần cứng GPU (NVDEC / Fallback)         │
│     - Lọc chuyển động & Điều phối phân luồng           │
└──────────────────────────┬─────────────────────────────┘
                           │ Giao tiếp RPC tốc độ cao (gRPC)
                           ▼
┌────────────────────────────────────────────────────────┐
│               Tầng Nhận Diện AI (Detector)             │
│     - Dynamic Batching tối ưu băng thông               │
│     - Mô hình Deep Learning tăng tốc qua TensorRT      │
└──────────────────────────┬─────────────────────────────┘
                           │ Tọa độ đối tượng (Bounding Boxes)
                           ▼
┌────────────────────────────────────────────────────────┐
│            Tầng Bám Vết & Xử Lý Nghiệp Vụ              │
│     - Bám vết quỹ đạo đa đối tượng (Kalman Filter)     │
│     - Máy trạng thái (FSM) xác thực cắt vạch ảo        │
│     - Phân loại chiều di chuyển (IN / OUT)             │
│     - Bộ lọc khử nhiễu không gian - thời gian          │
└──────────────────────────┬─────────────────────────────┘
                           │ Sự kiện & Số liệu thống kê
                           ▼
┌────────────────────────────────────────────────────────┐
│             Tầng Tích Hợp & Quản Trị Hệ Thống          │
│     - CMS / VMS Dashboard (REST API / Socket)          │
│     - Lưu trữ ảnh bằng chứng sự kiện                   │
└────────────────────────────────────────────────────────┘
```

---

## 💻 3. Ngăn Xếp Công Nghệ (Tech Stack)

| Thành phần | Công nghệ sử dụng | Mục đích |
| :--- | :--- | :--- |
| **Ngôn ngữ nền tảng** | Python 3.10+ | Lập trình logic điều phối, bám vết và phân tích quỹ đạo |
| **Deep Learning** | PyTorch, Ultralytics YOLOv8 | Nhận diện đối tượng người trong khung hình |
| **GPU Acceleration** | NVIDIA CUDA, TensorRT | Tăng tốc độ suy luận mô hình và xử lý ma trận |
| **Giao tiếp IPC** | gRPC, Protocol Buffers | Kênh truyền thông nhị phân nội bộ tốc độ cao giữa các services |
| **Xử lý Video** | OpenCV, FFmpeg / NVDEC Hardware Decode | Giải mã video camera phần cứng trên GPU |
| **Bám vết (Tracking)**| ByteTrack, Kalman Filter | Ước lượng trạng thái và duy trì quỹ đạo chuyển động liên tục |
| **Hình học không gian** | Shapely, NumPy, SciPy | Xử lý giao cắt đoạn thẳng và phép chiếu vector |
| **Ảo hóa & Đóng gói** | Docker, Docker Compose | Chuẩn hóa môi trường chạy và hỗ trợ mở rộng linh hoạt |

---

## 📂 4. Cấu Trúc Mã Nguồn (Project Structure)

```
couting_person/
├── README.md                 # Tài liệu tổng quan dự án
├── PEOPLE_COUTING/           # Module triển khai dịch vụ đếm người
│   ├── README.md             # Tài liệu kỹ thuật chi tiết của module
│   ├── docker-compose.yml    # Cấu hình containerization cho toàn bộ dịch vụ
│   │
│   ├── detector/             # [Microservice] Nhận diện đối tượng AI qua gRPC
│   │   ├── Dockerfile        # Môi trường chạy suy luận GPU
│   │   └── app/              # Server gRPC, Dynamic Batch Worker, YOLO Model Loader
│   │
│   └── counting/             # [Microservice] Điều phối luồng và đếm người
│       ├── Dockerfile        # Môi trường chạy orchestrator
│       └── app/              # Camera Reader, Multi-object Tracker, Line Crossing FSM
│
└── GRPC_DET_PERSON/          # [Microservice] Detector phiên bản độc lập
```

---

## 🧠 5. Nguyên Lý & Giải Pháp Kỹ Thuật Cốt Lõi

### Nhận diện & Tối ưu hóa suy luận GPU
- **Xử lý tập trung**: Toàn bộ luồng camera chia sẻ cụm Detector tập trung, giúp quản lý tài nguyên GPU hiệu quả thay vì khởi tạo mô hình riêng cho từng luồng.
- **Dynamic Batching**: Tự động gom các yêu cầu suy luận từ nhiều camera trong một cửa sổ thời gian cực ngắn thành batch duy nhất, tối đa hóa hiệu suất tính toán song song của nhân CUDA.
- **Tối ưu hóa TensorRT**: Mô hình được lượng tử hóa nửa độ chính xác (FP16), giảm dung lượng VRAM và tăng thông lượng xử lý khung hình trên giây (FPS).

### Bám vết quỹ đạo đa đối tượng (Multi-Object Tracking)
- **Cơ chế liên kết hai tầng**: Kết hợp ma trận tương đồng không gian (IoU) với phân loại độ tin cậy để duy trì vết chuyển động của người đi bộ ngay cả trong điều kiện ánh sáng yếu hoặc bị che khuất cục bộ.
- **Bộ lọc Kalman đa chiều**: Dự đoán mượt mà hướng đi và bù đắp các khung hình bị suy giảm tín hiệu hoặc bỏ qua để tối ưu tải xử lý.
- **Tái kết nối quỹ đạo**: Nhận diện và ghép nối lại các đối tượng bị mất dấu tạm thời, hạn chế việc gán định danh mới gây sai lệch dữ liệu.

### Xác thực vượt vạch ảo & Phân định chiều di chuyển
- **Máy trạng thái hữu hạn (FSM)**: Thay vì chỉ kiểm tra giao cắt điểm-đoạn đơn thuần tại một khung hình đơn lẻ, hệ thống áp dụng máy trạng thái theo dõi toàn bộ hành trình chuyển tiếp của đối tượng từ vùng tiếp cận, qua vạch ảo, đến vùng thoát.
- **Vector định hướng học**: Chiều di chuyển (Vào/Ra) được xác định bằng quan hệ góc giữa vector chuyển động thực tế của đối tượng và vector quy chuẩn của vạch ảo, đảm bảo độ chính xác bất kể góc nghiêng của camera.

### Cơ chế khử nhiễu & Lọc báo ảo thông minh
- **Phát hiện chuyển động sơ cấp**: Tự động bỏ qua các khung hình tĩnh nhằm giảm thiểu tải suy luận không cần thiết cho mô hình AI.
- **Bộ lọc biên độ dịch chuyển**: Phân biệt chuyển động tịnh tiến có chủ đích của người đi bộ với các dao động cục bộ tại chỗ (như xe máy đỗ yên trước sảnh, bóng cây, rung lắc camera do gió).
- **Bộ đệm không gian - thời gian**: Khóa tạm thời điểm kích hoạt sau mỗi sự kiện đếm thành công để loại trừ hiện tượng đếm lặp đối với các đối tượng di chuyển theo nhóm hoặc đi sát nhau.

---

## 📡 6. Khả Năng Tích Hợp & Dữ Liệu Đầu Ra

Hệ thống được thiết kế dạng mở để dễ dàng tích hợp vào các hệ sinh thái giám sát an ninh:
- **Đồng bộ tham số động**: Nhận cấu hình vị trí camera và vạch ảo trực tiếp từ máy chủ quản lý trung tâm.
- **Dữ liệu phân tích chuẩn hóa**: Cung cấp số liệu thống kê thời gian thực gồm tổng lưu lượng (`count`), số lượt vào (`enter`), số lượt ra (`exit`) kèm dấu thời gian chuẩn.
- **Hình ảnh trích xuất sự kiện**: Tự động trích xuất và lưu trữ ảnh chụp toàn cảnh cùng ảnh cận cảnh đối tượng tại thời điểm xảy ra sự kiện đếm phục vụ tra cứu và kiểm toán dữ liệu.

---

## 👥 Bản Quyền & Phát Triển
- **Đơn vị phát triển**: Đội ngũ Kỹ thuật Core AI Vision
- **Mã nguồn**: [Longhanhmid24/couting_person](https://github.com/Longhanhmid24/couting_person)
