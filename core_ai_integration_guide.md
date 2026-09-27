# Hướng Dẫn Tích Hợp Đếm Người & Phương Tiện Cho CORE AI (CMS Backend)

Tài liệu hướng dẫn dành cho **Đội ngũ Phát triển CORE AI** để gửi dữ liệu đếm số lượng người (`people_counting`) và đếm số lượng phương tiện (`vehicle_counting`) về CMS Backend.

---

## 1. Nguyên Lý Tích Hợp
* **Địa chỉ CMS API**: `http://<CMS_HOST>:8080/api/camera-statistics`
* **Cơ chế Kiểm tra License**: CMS Backend sẽ tự động xác thực License của Camera (`service_type` có chứa `people_counting` hoặc `vehicle_counting`).
  * Nếu camera **có License**: Dữ liệu đếm được ghi nhận vào DB và phát SocketIO realtime cho UI.
  * Nếu camera **chưa có License**: CMS Backend từ chối lưu dữ liệu (`return null`).

---

## 2. API Push Dữ Liệu Đếm

### 2.1 API Gửi Đơn Lẻ 1 Camera (Synchronous)
* **HTTP Method**: `POST`
* **URL**: `/api/camera-statistics`
* **Headers**:
  ```http
  Content-Type: application/json
  session: <ADMIN_OR_SYSTEM_SESSION_KEY>
  ```

#### Payload Đếm Người (`people_counting`):
```json
{
  "stream_id": "64a8fa75-316d-49f3-8e19-dca09db78e19",
  "metric_type": "people_counting",
  "data": {
    "count": 15,
    "person": 15
  },
  "time": 1722780000.0
}
```

#### Payload Đếm Phương Tiện (`vehicle_counting`):
```json
{
  "stream_id": "64a8fa75-316d-49f3-8e19-dca09db78e19",
  "metric_type": "vehicle_counting",
  "data": {
    "count": 59,
    "car": 15,
    "motorbike": 40,
    "truck": 3,
    "bus": 1
  },
  "time": 1722780000.0
}
```

#### Mô tả các trường:
| Trường | Kiểu dữ liệu | Bắt buộc | Mô tả |
| :--- | :--- | :--- | :--- |
| `stream_id` | `UUID string` | **Có** | ID của Camera (trùng với Camera ID trên CMS) |
| `metric_type` | `string` | **Có** | `people_counting` hoặc `vehicle_counting` |
| `data` | `object` | **Có** | Chứa số liệu đếm số lượng (tổng `count`, phân loại `person`, `car`, `motorbike`, `truck`, `bus`...) |
| `time` | `float` | Không | Unix timestamp (giây). Nếu không truyền CMS sẽ lấy thời gian server |

---

### 2.2 API Gửi Batch Nhiều Camera (Asynchronous)
Dùng khi CORE AI thu thập đếm hàng loạt nhiều camera cùng lúc để tối ưu hiệu năng HTTP request.

* **HTTP Method**: `POST`
* **URL**: `/api/camera-statistics/batch`
* **Headers**: `Content-Type: application/json`

#### Payload Mẫu:
```json
{
  "records": [
    {
      "stream_id": "64a8fa75-316d-49f3-8e19-dca09db78e19",
      "metric_type": "people_counting",
      "data": { "count": 20, "person": 20 },
      "time": 1722780000.0
    },
    {
      "stream_id": "8e19dca0-9db7-49f3-64a8-fa75316d8e19",
      "metric_type": "vehicle_counting",
      "data": { "count": 45, "car": 10, "motorbike": 35 },
      "time": 1722780000.0
    }
  ]
}
```

---

## 3. Mã Lỗi & Lưu Ý Cho CORE AI
1. **Response `null` hoặc 200 thành công**:
   - Nếu response chứa record Object: Dữ liệu đã ghi nhận & phát SocketIO thành công.
   - Nếu response trả về `null`: Camera chưa được cấp License `people_counting` hoặc `vehicle_counting`. CORE AI cần kiểm tra cấp phép trên CMS.
2. **Tần suất gửi khuyến nghị**:
   - Gửi theo định kỳ mỗi phút hoặc khi có thay đổi biến động số lượng đáng kể.
