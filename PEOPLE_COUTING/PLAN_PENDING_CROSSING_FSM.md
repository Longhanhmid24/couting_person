# Kế hoạch tối ưu hóa bộ đếm người: Cơ chế Pending Crossing FSM (Thực hiện ngày mai)

## 1. Bối cảnh & Nguyên nhân sự cố (Phân tích ngày 30/09/2026)
Vào lúc **18:46:50 – 18:47:55**, hệ thống ghi nhận một cô gái đi từ ngoài sân vào sảnh tòa nhà nhưng không được đếm (`ĐẾM_NGƯỜI = 0`) dù detector nhận diện liên tục 217 lượt.

### Các nguyên nhân kỹ thuật đã xác định:
1. **FSM nuốt sự kiện một chiều (Single-Frame Premature State Switch)**:
   - Điểm sinh ra (`first_point`) của người đi bộ xuất hiện cách vạch khoảng $20 - 30\text{px}$.
   - Khi vừa bước qua vạch, dịch chuyển mới chỉ đạt $25 - 30\text{px} < 40\text{px}$ (ngưỡng dịch chuyển).
   - `LineZoneCrossingFSM` trả về `None`, nhưng lại lập tức chuyển `self.state = side` (bên kia vạch).
   - Ở các frame tiếp theo, người đó tiếp tục đi sâu vào trong sảnh ($net\_disp > 60\text{px}$), nhưng vì `side == previous` nên FSM không bao giờ sinh lại sự kiện nữa.
2. **Khóa chết `obj.counted = True` trước khi hoàn tất hành trình**:
   - Tại `_check_line_crossing` trong `person_process.py`, việc gán `obj.counted = True` khi chưa đủ `net_disp` hay `min_frames` khiến track bị loại bỏ vĩnh viễn ở các frame sau.
3. **Biên hình chiếu vạch bị hẹp (`projection < 0.02 or > 0.98`)**:
   - Vạch `[(387, 632), (895, 513)]` có đầu phải sát tường. Người đi bộ nép bên phải cầu thang có hình chiếu vượt $0.98$ bị coi là ngoài vạch và bị reset track.

---

## 2. Giải pháp kỹ thuật: Pending Crossing FSM

### 2.1. Luồng xử lý Pending Crossing
- Khi track cắt qua vạch đếm (chuyển vế từ `side = +1` sang `-1` hoặc ngược lại):
  - FSM ghi nhận và lưu hướng di chuyển vào `self.pending_event = "person_in"` (hoặc `"person_out"`).
  - Xác định hướng dựa trên vector quỹ đạo tổng thể $(point - first\_point)$ kết hợp chuyển vế hình học, loại bỏ hoàn toàn nhiễu giật hộp 1-frame.
- **Nếu tại frame cắt vạch chưa đạt $net\_disp \ge 35\text{px}$**:
  - FSM **giữ nguyên `pending_event`**, không hủy bỏ và không khóa track.
  - Ở các frame tiếp theo, người đi bộ bước thêm 1–2 bước nữa $\rightarrow net\_disp \ge 35\text{px} \rightarrow$ FSM **kích hoạt sự kiện đếm và lưu ảnh ngay lập tức**.
- **Đối với xe máy đỗ yên tại bậc thang**:
  - Hộp nhận diện của xe máy chỉ dao động tại chỗ $\approx 5 - 15\text{px}$.
  - Do không bao giờ đạt ngưỡng $35\text{px}$, `pending_event` sẽ không bao giờ được kích hoạt và tự động giải phóng khi track hết hạn $\rightarrow$ **0 false count**.

---

## 3. Các bước triển khai cụ thể cho ngày mai

### Bước 1: Cập nhật [`counting/app/utils/line_crossing.py`](file:///home/tado/Long-Ana/PEOPLE_COUTING/counting/app/utils/line_crossing.py)
- Nâng cấp `LineZoneCrossingFSM`:
  - Thêm thuộc tính `self.pending_event = None` và `self.counted = False`.
  - Nới rộng biên hình chiếu đoạn thẳng trong `_signed_distance`: `if projection < -0.10 or projection > 1.10: return None`.
  - Cập nhật hàm `_determine_direction(origin_side, dest_side, current_point)` dùng quỹ đạo tổng thể $(point - first\_point)$.
  - Triển khai logic kiểm tra `pending_event`: chỉ phát sự kiện khi `net_disp >= self.stationary_displacement` và không đứng yên.

### Bước 2: Cập nhật [`counting/app/services/person_process.py`](file:///home/tado/Long-Ana/PEOPLE_COUTING/counting/app/services/person_process.py)
- Trong `_check_line_crossing`:
  - Không gán `obj.counted = True` khi chưa đủ `min_frames` hoặc chưa đủ `min_movement`.
  - Chỉ gán `obj.counted = True` khi đã phát sự kiện đếm thành công hoặc khi vi phạm cổng dáng xe máy bè ngang / spatial cooldown.

### Bước 3: Cập nhật [`counting/app/core/settings.py`](file:///home/tado/Long-Ana/PEOPLE_COUTING/counting/app/core/settings.py)
- `MIN_FRAMES_BEFORE_COUNT = 5` (thay vì 8).
- `MIN_PATH_MOVEMENT_PIXELS = 35.0` (thay vì 40.0).
- `MIN_ENTRY_DISTANCE_LINE = 25.0` (thay vì 35.0).

### Bước 4: Đóng gói Docker & Triển khai lên Server `100.115.98.97`
```bash
# 1. Build image mới
docker build -t harbor.tado.vn/long_dev/people_counting:v2.1.1 -f docker/Dockerfile .
docker push harbor.tado.vn/long_dev/people_counting:v2.1.1

# 2. Deploy lên server 100.115.98.97
ssh kb@100.115.98.97
sudo docker stop grpc-main-people-counting && sudo docker rm grpc-main-people-counting
sudo docker pull harbor.tado.vn/long_dev/people_counting:v2.1.1
# Khởi chạy lại container với image v2.1.1
```

### Bước 5: Kiểm thử thực tế (Live Testing)
1. Quan sát xe máy đỗ yên tại bậc thang: Đảm bảo số đếm giữ nguyên 0 qua nhiều chu kỳ 60s.
2. Kiểm tra người đi bộ (IN và OUT): Đảm bảo đếm chính xác cả người đi sát tường và người xuất hiện gần vạch.
3. Tải và xem ảnh chụp (`view_file`) kiểm chứng chất lượng crop.
