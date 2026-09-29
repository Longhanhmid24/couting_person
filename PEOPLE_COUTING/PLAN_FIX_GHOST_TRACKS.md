# KẾ HOẠCH NÂNG CẤP LÕI TRACKING DỰ ÁN: XỬ LÝ TRIỆT ĐỂ GHOST TRACKS (VẾT MA KHI NGƯỜI ĐI KHUẤT)

> **Mục tiêu**: Nâng cấp trực tiếp vào engine tracking và đếm người của dự án (`PEOPLE_COUTING`), triệt tiêu hiện tượng giữ track ảo khi người đã đi khuất tầm nhìn, tránh đếm sót người tiếp theo, loại bỏ rủi ro đếm sai và tối ưu hóa tài nguyên runtime.

---

## 1. TÁC HẠI CỦA GHOST TRACKS ĐỐI VỚI HỆ THỐNG DỰ ÁN

Ghost tracks (vết ma còn tồn tại sau khi người đã đi khuất) không chỉ là vấn đề hiển thị, mà ảnh hưởng trực tiếp đến logic nghiệp vụ cốt lõi của hệ thống đếm người:

1. **Đếm sót người mới (Under-counting nghiêm trọng):**
   - Khi một người đi qua cửa đã được đếm (`counted = True`), sau đó đi vào trong nhà khuất tầm nhìn.
   - Nếu track cũ vẫn bị giữ 75 frames (7.5 giây) lơ lửng ngay cửa, khi **người tiếp theo bước qua cửa**, thuật toán ByteTrack sẽ ghép detection của người mới vào ID cũ này (do khoảng cách toạ độ gần).
   - Vì ID cũ đã mang cờ `counted = True`, hệ thống sẽ **bỏ qua không đếm người mới**, dẫn đến thiếu hụt số liệu đếm thực tế!
2. **Đếm sai do Kalman Drift & Ghép nhầm nhiễu (False Counting):**
   - Nếu một người chưa vượt vạch (đi đến gần cửa rồi quay ra ngoài khuất camera), track ma vẫn đứng im ở cửa.
   - Khi Kalman Filter tiếp tục dự đoán toạ độ không có detection thực, sai số tích luỹ (drift) hoặc một vệt bóng/vật thể khác ghép nhầm có thể vô tình đẩy toạ độ cắt qua vạch $\rightarrow$ **kích hoạt đếm ma**.
3. **Ô nhiễm bộ nhớ & lãng phí CPU trong xử lý hàng loạt:**
   - Mỗi track ma giữ ma trận hiệp phương sai Kalman, lịch sử toạ độ, FSM state trong `_person_tracks`.
   - Trong giờ cao điểm nhiều người qua lại, ma trận liên kết IoU N x M phải tính toán liên tục với hàng chục track đã biến mất khỏi thực tế.
4. **Sai lệch ảnh chụp bằng chứng (Alarm Snapshots):**
   - Khi có sự kiện kích hoạt chụp ảnh lưu trữ, các box ma có thể bị vẽ đè hoặc crop nhầm vào đối tượng không còn hiện diện trong khung hình.

---

## 2. NGUYÊN NHÂN KỸ THUẬT TRONG MÃ NGUỒN HIỆN TẠI

1. **`TRACK_BUFFER` đặt 75 frames (~7.5 giây):** Quá dài đối với luồng camera ra vào cửa/cổng (nơi người di chuyển liên tục và rời khỏi khung hình nhanh chóng).
2. **Thiếu cơ chế giải phóng sớm cho track đã hoàn thành đếm (`counted == True`):** Khi một người đã được đếm xong và biến mất khỏi tầm nhìn, hệ thống vẫn giữ track trong bộ nhớ ngang bằng thời gian với người chưa đếm.
3. **Không có cơ chế thoát biên (Boundary Exit):** Người bước ra mép khung hình hoặc khuất sau tường mép cửa không được giải phóng ngay.
4. **Không tách bạch vòng đời track trong Pipeline:** Module `PersonProcessor` nhận toàn bộ danh sách track từ `byte_tracker` bao gồm cả những track đang mất dấu (`matched = False`), khiến logic đếm và quản lý đối tượng xử lý cả những đối tượng không có thực trong frame hiện tại.

---

## 3. GIẢI PHÁP CỐT LÕI NÂNG CẤP VÀO PROJECT

```
                 [Frame mới từ Camera]
                           │
                           ▼
                  [Detector YOLO/gRPC]
                           │
                           ▼
                 [ByteTrack Association]
                           │
         ┌─────────────────┴─────────────────┐
         ▼                                   ▼
 [Matched: Có Detection]           [Unmatched: Mất dấu]
         │                                   │
  • Cập nhật Kalman Filter            • Tăng obj.lost += 1
  • Xét vượt vạch đếm người           • KIỂM TRA ĐIỀU KIỆN GIẢI PHÓNG:
  • obj.lost = 0                        - Đã đếm (counted) & lost >= 5  ──► [GIẢI PHÓNG NGAY]
                                        - Thoát biên ảnh & lost >= 2    ──► [GIẢI PHÓNG NGAY]
                                        - lost > TRACK_BUFFER (25)      ──► [HẾT HẠN - EXPIRED]
```

### Chi tiết các cải tiến thuật toán:

### 1. Cơ chế Fast Retirement cho Track đã hoàn thành đếm
- **Nguyên lý:** Một người đã được ghi nhận vượt vạch thành công (`obj.counted == True`), mục tiêu theo vết đối với người này đã hoàn tất.
- **Quy tắc:** Nếu `obj.counted == True` và mất dấu liên tiếp **5 frames** (~0.5 giây) $\rightarrow$ **Xóa bỏ và giải phóng track ngay lập tức**.
- **Hiệu quả:** Cửa ra vào lập tức thông thoáng; khi người tiếp theo bước tới, ByteTrack sẽ khởi tạo ID mới và đếm chính xác 100%, không bị nuốt ID.

### 2. Cơ chế Boundary Exit (Thoát khỏi mép khung hình)
- **Nguyên lý:** Người đi vào mép cửa hoặc góc khuất sát cạnh camera (cách biên ảnh < 20px) và bị mất dấu detection.
- **Quy tắc:** Nếu toạ độ bbox nằm sát biên ($x < 20$ hoặc $y < 20$ hoặc $x+w > W-20$) và `lost >= 2` $\rightarrow$ Đánh dấu hết hạn ngay lập tức, không chờ hết buffer.

### 3. Rút ngắn `TRACK_BUFFER` tối ưu cho nghiệp vụ đếm người
- Giảm `TRACK_BUFFER` từ **75 frames (7.5s)** xuống **25 frames (2.5s)**.
- Đủ thời gian xử lý che khuất ngắn (2 người đi chéo qua nhau trong 1 - 2 giây), nhưng loại bỏ hoàn toàn tình trạng giữ vết trơ trơ suốt 7 - 10 giây khi người đã rời đi.

### 4. Khóa Kalman Drift không cho đếm ma
- Chỉ kích hoạt hàm `_check_line_crossing()` khi track có **detection thực tế trong frame hiện tại** (`matched == True`).
- Tuyệt đối không cho phép toạ độ dự đoán thuần tuý của Kalman Filter tự ý kích hoạt sự kiện vượt vạch khi không có người thực tế.

---

## 4. CHI TIẾT CÁC FILE CẦN CẬP NHẬT TRONG DỰ ÁN

| File | Nội dung cập nhật |
| :--- | :--- |
| `app/core/settings.py` | • `TRACK_BUFFER = 25` (giảm từ 75)<br>• `COUNTED_RETIRE_FRAMES = 5`<br>• `BOUNDARY_MARGIN = 20` |
| `app/tracker/byte_tracker.py` | • Hỗ trợ cờ giải phóng nhanh các track đã mất dấu lâu hoặc thoát biên.<br>• Phân tách rõ track active (có detection) và track lost dự đoán ngầm. |
| `app/services/person_process.py` | • Áp dụng logic Fast Retirement: xoá track khỏi `_person_tracks` khi `counted=True` và `lost >= COUNTED_RETIRE_FRAMES`.<br>• Chỉ gọi `_check_line_crossing` khi `result.matched == True`.<br>• Xử lý dọn dẹp bộ nhớ triệt để, không để rò rỉ object tracking. |

---

## 5. KẾ HOẠCH TRIỂN KHAI VÀ KIỂM THỬ

1. **Bước 1: Triển khai mã nguồn:**
   - Sửa `settings.py`, `byte_tracker.py`, `person_process.py` theo đúng 4 nguyên lý trên.
2. **Bước 2: Đóng gói và Build Docker Image:**
   - Build image Docker production: `harbor.tado.vn/long_dev/people_counting:v2.0.0`.
   - Push lên Harbor registry.
3. **Bước 3: Deploy lên Server Test (`100.115.98.97`):**
   - Pull image mới và restart container `grpc-main-people-counting`.
4. **Bước 4: Kiểm tra đánh giá thực tế:**
   - Quan sát log backend: theo dõi `track_sống` giảm về đúng số lượng người thực tế, không bị tích luỹ vết ma.
   - Kiểm tra đếm thực tế khi nhiều người lần lượt đi qua cửa: xác nhận đếm đủ từng người, không bị nuốt đếm do ghép vào track cũ.
5. **Bước 5: Dọn dẹp sạch mã nguồn Live Stream tạm thời:**
   - Xóa `live_streamer.py`, gỡ port `8899`, revert `main.py` về nguyên bản production khi hoàn tất kiểm tra.
