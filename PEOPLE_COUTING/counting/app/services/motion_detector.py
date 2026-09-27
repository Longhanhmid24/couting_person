"""
motion_detector.py — Unified motion detection module (identical across ALL services).

Contains:
  - GPUMotionDetector       : single-camera, torch CUDA tensor (Y plane)
  - CPUMotionDetector       : single-camera, numpy grayscale (fallback / CPU decode mode)
  - BatchedGPUMotionDetector: N cameras in one vectorized forward (standalone motion engines)

Ghi chú tối ưu (quan trọng khi chạy nhiều camera):
  * min_motion_pixels được CHUẨN HOÁ THEO DIỆN TÍCH so với khung tham chiếu
    640x360. Trước đây đường GPU lọc ở 640x360 còn đường CPU lọc ở 320x180
    nhưng dùng chung một ngưỡng tuyệt đối => camera decode bằng CPU kém nhạy
    gấp 4 lần với cùng một chuyển động. Nay cùng config cho cùng độ nhạy.
  * Morphology mặc định là OPEN rồi CLOSE (denoise=True). Bản cũ chỉ CLOSE
    (dilate→erode) nên KHÔNG hề loại được nhiễu hạt: một pixel nhiễu nở ra 3x3
    rồi co lại đúng 1 pixel, vẫn được đếm. Ban đêm nhiễu cảm biến dễ vượt
    ngưỡng và làm mọi frame bị coi là có chuyển động — tức là mất trắng lợi ích
    của motion gating. OPEN xoá hạt nhiễu trước, CLOSE vá lỗ trong khối thật.
  * Giảm số lần đồng bộ GPU→CPU và số tensor cấp phát mỗi frame (đếm bằng
    count_nonzero trên mask fp16 thay vì .float().sum(), tái dùng buffer diff).
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Khung tham chiếu để chuẩn hoá min_motion_pixels theo diện tích.
REF_W, REF_H = 640, 360


def _scale_min_pixels(min_motion_pixels, width, height):
    """Quy đổi min_motion_pixels sang độ phân giải thực tế của mask."""
    if not width or not height:
        return max(1, int(min_motion_pixels))
    ratio = (float(width) * float(height)) / float(REF_W * REF_H)
    return max(1, int(round(min_motion_pixels * ratio)))


class GPUMotionDetector:
    """
    Background subtraction + morphology on a single camera frame on the GPU.
    Supports float16 and float32 CUDA tensors of shape (Height, Width) or (1, Height, Width).
    """
    def __init__(self, width=640, height=360, threshold=12.0, alpha=0.005,
                 min_motion_pixels=400, dtype=torch.float16, denoise=True):
        self.width = width
        self.height = height
        self.threshold = threshold
        self.alpha = alpha
        self.min_motion_pixels = min_motion_pixels
        self.dtype = dtype
        self.denoise = denoise
        self.bg_model = None       # CUDA tensor
        self._diff = None          # buffer tái dùng, tránh cấp phát mỗi frame
        self._min_px = None        # ngưỡng đã chuẩn hoá theo diện tích mask thật

    @torch.no_grad()
    def detect(self, frame_tensor):
        """
        Processes a single frame tensor on the GPU.
        - frame_tensor: CUDA tensor of shape (H, W) or (1, H, W)

        Returns:
            - motion_detected: boolean
            - motion_pixel_count: int
        """
        if frame_tensor is None:
            return False, 0

        if frame_tensor.dim() == 2:
            frame_tensor = frame_tensor.unsqueeze(0)

        if frame_tensor.dtype != self.dtype:
            frame_tensor = frame_tensor.to(self.dtype)

        # Initialize background model if needed
        if self.bg_model is None or self.bg_model.shape != frame_tensor.shape:
            self.bg_model = frame_tensor.clone()
            self._diff = torch.empty_like(frame_tensor)
            # Chuẩn hoá ngưỡng theo diện tích mask thật (không theo self.width/height
            # khai báo lúc khởi tạo, vì caller có thể resize khác).
            h, w = frame_tensor.shape[-2], frame_tensor.shape[-1]
            self._min_px = _scale_min_pixels(self.min_motion_pixels, w, h)
            return False, 0

        # 1. Update background model: bg = (1 - alpha) * bg + alpha * frame
        self.bg_model.mul_(1.0 - self.alpha).add_(frame_tensor, alpha=self.alpha)

        # 2. Absolute difference (detects both brightness and color changes)
        #    Ghi vào buffer sẵn có để không cấp phát tensor mới mỗi frame.
        diff = self._diff
        torch.sub(frame_tensor, self.bg_model, out=diff)
        diff.abs_()

        # 3. Apply sensitive threshold -> mask 0/1 ngay trên buffer diff
        #    sign(diff - t) cho 1 khi diff > t, 0 khi bằng, -1 khi nhỏ hơn;
        #    clamp_(0) đưa về đúng ngữ nghĩa (diff > threshold).
        diff.sub_(self.threshold).sign_().clamp_(min=0)
        mask4d = diff.unsqueeze(0)

        # 4. Morphology. OPEN (erode→dilate) xoá nhiễu hạt, CLOSE (dilate→erode)
        #    vá lỗ trong khối chuyển động thật.
        if self.denoise:
            # erode(x) = -maxpool(-x). Dùng negate nên đúng với mọi giá trị và
            # rẻ hơn dạng 1 - maxpool(1 - x) của bản cũ (bớt 2 lần cấp phát).
            eroded = F.max_pool2d(torch.neg(mask4d), kernel_size=3, stride=1, padding=1)
            eroded.neg_()                                     # = erode(mask)
            opened = F.max_pool2d(eroded, kernel_size=3, stride=1, padding=1)
        else:
            opened = mask4d

        dilated = F.max_pool2d(opened, kernel_size=3, stride=1, padding=1)
        closed = F.max_pool2d(torch.neg(dilated), kernel_size=3, stride=1, padding=1)
        closed.neg_()                                         # = erode(dilate(x))

        # 5. Count motion pixels. count_nonzero chạy trực tiếp trên fp16 nên
        #    không phải cấp phát bản fp32 của cả mask như bản cũ.
        motion_pixel_count = int(torch.count_nonzero(closed))
        min_px = self._min_px if self._min_px is not None else self.min_motion_pixels
        motion_detected = motion_pixel_count > min_px

        return motion_detected, motion_pixel_count

    def reset(self):
        """Reset the background model (e.g. on reconnection/restart)"""
        self.bg_model = None
        self._diff = None


class CPUMotionDetector:
    """
    Background subtraction + morphology on a CPU numpy grayscale array.
    """
    def __init__(self, threshold=12.0, alpha=0.005, min_motion_pixels=400,
                 denoise=True):
        self.threshold = threshold
        self.alpha = alpha
        self.min_motion_pixels = min_motion_pixels
        self.denoise = denoise
        self.bg_model = None
        # Kernel tạo một lần, bản cũ gọi getStructuringElement mỗi frame.
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self._bg_u8 = None      # buffer tái dùng
        self._diff = None
        self._mask = None
        self._min_px = None

    def detect(self, gray_frame):
        """
        gray_frame: Grayscale numpy array

        Returns:
            - motion_detected: boolean
            - motion_pixel_count: int
        """
        if gray_frame is None:
            return False, 0

        if self.bg_model is None or self.bg_model.shape != gray_frame.shape:
            self.bg_model = gray_frame.astype(np.float32)
            self._bg_u8 = np.empty_like(gray_frame)
            self._diff = np.empty_like(gray_frame)
            self._mask = np.empty_like(gray_frame)
            h, w = gray_frame.shape[:2]
            self._min_px = _scale_min_pixels(self.min_motion_pixels, w, h)
            return False, 0

        # 1. Update background model: bg = (1 - alpha) * bg + alpha * frame
        cv2.accumulateWeighted(gray_frame, self.bg_model, self.alpha)
        cv2.convertScaleAbs(self.bg_model, dst=self._bg_u8)

        # 2. Absolute difference
        cv2.absdiff(gray_frame, self._bg_u8, dst=self._diff)

        # 3. Apply threshold
        cv2.threshold(self._diff, int(self.threshold), 255, cv2.THRESH_BINARY,
                      dst=self._mask)

        # 4. Morphology: OPEN loại nhiễu hạt rồi CLOSE vá lỗ (xem docstring module).
        mask = self._mask
        if self.denoise:
            cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel, dst=mask)
        cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, dst=mask)

        # 5. Count motion pixels
        motion_pixel_count = cv2.countNonZero(mask)
        min_px = self._min_px if self._min_px is not None else self.min_motion_pixels
        motion_detected = motion_pixel_count > min_px

        return motion_detected, motion_pixel_count

    def reset(self):
        """Reset the background model"""
        self.bg_model = None
        self._bg_u8 = None
        self._diff = None
        self._mask = None


class BatchedGPUMotionDetector:
    """
    Vectorized background subtraction + morphology on a batch of camera frames on the GPU.

    Bản cũ mỗi lượt gọi đều torch.stack() lại toàn bộ background model rồi ghi
    NGƯỢC VÀO dict bằng VIEW của tensor stack (self.bg_models[cam] = bg_batch[i]).
    Việc đó giữ sống cả tensor batch cho mỗi camera (không giải phóng được) và
    nếu camera_ids có phần tử trùng thì các view cùng trỏ về một chỗ.
    Nay giữ MỘT buffer batch bền vững, chỉ dựng lại khi tập/thứ tự camera đổi.
    """
    def __init__(self, width=640, height=360, threshold=25, alpha=0.05,
                 min_motion_pixels=1000, dtype=torch.float16, denoise=True):
        self.width = width
        self.height = height
        self.threshold = threshold
        self.alpha = alpha
        self.min_motion_pixels = min_motion_pixels
        self.dtype = dtype
        self.denoise = denoise
        self.bg_models = {}      # camera_id -> CUDA tensor (shape: 1 x height x width)
        self._batch_key = None   # tuple(camera_ids) của buffer đang giữ
        self._bg_batch = None    # buffer batch bền vững (N,1,H,W)
        self._diff = None
        self._min_px = None

    def _flush_batch(self):
        """Đẩy background trong buffer batch về dict dạng bản sao độc lập."""
        if self._bg_batch is None or not self._batch_key:
            return
        for i, cam_id in enumerate(self._batch_key):
            if i < self._bg_batch.shape[0]:
                self.bg_models[cam_id] = self._bg_batch[i].clone()
        self._bg_batch = None
        self._batch_key = None

    def _build_batch(self, camera_ids, batch_tensor):
        """Dựng buffer batch từ dict, khởi tạo camera mới bằng frame hiện tại."""
        shape = batch_tensor.shape[1:]
        rows = []
        for i, cam_id in enumerate(camera_ids):
            bg = self.bg_models.get(cam_id)
            if bg is None or tuple(bg.shape) != tuple(shape):
                # Camera mới, hoặc camera reconnect với độ phân giải khác.
                bg = batch_tensor[i].clone()
                self.bg_models[cam_id] = bg
            rows.append(bg)
        self._bg_batch = torch.stack(rows).contiguous()
        self._batch_key = tuple(camera_ids)
        self._diff = torch.empty_like(self._bg_batch)
        h, w = batch_tensor.shape[-2], batch_tensor.shape[-1]
        self._min_px = _scale_min_pixels(self.min_motion_pixels, w, h)

    @torch.no_grad()
    def process_batch(self, camera_ids, batch_tensor):
        """
        Processes a batch of frames from a list of cameras.
        - camera_ids: list of strings (length N)
        - batch_tensor: float16/float32 CUDA tensor of shape (N, 1, Height, Width)

        Returns:
        - motion_detected: list of booleans (length N)
        - motion_pixel_counts: list of integers (length N)
        """
        if not camera_ids or batch_tensor is None or batch_tensor.size(0) == 0:
            return [], []

        if batch_tensor.dtype != self.dtype:
            batch_tensor = batch_tensor.to(self.dtype)

        key = tuple(camera_ids)
        if self._bg_batch is None or self._batch_key != key or \
                self._bg_batch.shape != batch_tensor.shape:
            # Tập camera (hoặc thứ tự / độ phân giải) đổi: lưu lại rồi dựng buffer mới.
            self._flush_batch()
            self._build_batch(camera_ids, batch_tensor)

        bg_batch = self._bg_batch

        # 1. Update background model: bg = (1 - alpha) * bg + alpha * frame
        #    Cập nhật TRỰC TIẾP trên buffer bền vững nên không cần ghi ngược dict.
        bg_batch.mul_(1.0 - self.alpha).add_(batch_tensor, alpha=self.alpha)

        # 2. Absolute difference (ghi vào buffer sẵn có)
        diff = self._diff
        torch.sub(batch_tensor, bg_batch, out=diff)
        diff.abs_()

        # 3. Apply threshold -> mask 0/1
        diff.sub_(self.threshold).sign_().clamp_(min=0)
        mask = diff

        # 4. Morphology: OPEN (chống nhiễu) rồi CLOSE (vá lỗ)
        if self.denoise:
            # erode(x) = -maxpool(-x)
            eroded = F.max_pool2d(torch.neg(mask), kernel_size=3, stride=1, padding=1)
            eroded.neg_()
            opened = F.max_pool2d(eroded, kernel_size=3, stride=1, padding=1)
        else:
            opened = mask

        dilated = F.max_pool2d(opened, kernel_size=3, stride=1, padding=1)
        closed = F.max_pool2d(torch.neg(dilated), kernel_size=3, stride=1, padding=1)
        closed.neg_()

        # 5. Count motion pixels. Đếm trên fp16 (count_nonzero) rồi CHỈ MỘT lần
        #    chuyển GPU->CPU cho cả batch, thay vì hai lần như bản cũ.
        motion_pixel_counts = torch.count_nonzero(closed, dim=(1, 2, 3))
        min_px = self._min_px if self._min_px is not None else self.min_motion_pixels
        counts = motion_pixel_counts.cpu()
        counts_list = counts.tolist()
        motion_detected = [c > min_px for c in counts_list]

        return motion_detected, counts_list

    def remove_camera(self, camera_id):
        """Clean up background model when camera is disabled/removed"""
        if self._batch_key and camera_id in self._batch_key:
            # Buffer batch đang chứa camera này -> lưu lại phần còn dùng được rồi bỏ.
            self._flush_batch()
        if camera_id in self.bg_models:
            del self.bg_models[camera_id]
