"""
camera_reader.py — Centralized Camera Reader Module for License Plate Recognition System.
Supports RTSP (GPU/CPU), Dahua SDK (GPU/CPU), and Hikvision SDK (CPU).
All readers push standardized FramePackets to a shared output queue.
"""
import threading
import time
import uuid
import queue
import logging
import os
from dataclasses import dataclass
from typing import Optional
import numpy as np
import cv2
import torch
from core.settings import settings

LOGGER = logging.getLogger(__name__)


@dataclass
class FramePacket:
    """Standardized frame output from any camera reader."""
    cam_id: str
    cam_name: str
    timestamp: float
    frame_uuid: str
    frame: Optional[np.ndarray]  # BGR frame; None when motion=False
    frame_count: int
    motion_detected: bool


# ---------------------------------------------------------------------------
# Base reader
# ---------------------------------------------------------------------------
class BaseCameraReader:
    """Abstract base for all camera readers."""

    def __init__(self, cam_id: str, output_queue: queue.Queue,
                 shutdown_event: threading.Event,
                 name: Optional[str] = None,
                 motion_threshold: float = 12.0,
                 motion_alpha: float = 0.005,
                 min_motion_pixels: int = 400,
                 max_connect_attempts: int = 0,
                 **kwargs):
        self.cam_id = cam_id
        self.cam_name = name or cam_id
        self.output_queue = output_queue
        self.shutdown = shutdown_event
        self.frame_count = 0
        # Số frame bị bỏ vì shared queue đầy — chỉ số trực tiếp cho biết engine
        # đã quá tải (đang chạy nhiều camera hơn khả năng xử lý của pipeline).
        self.dropped_frames = 0
        self._node = uuid.UUID(cam_id).node if self._is_valid_uuid(cam_id) else uuid.getnode()
        self._thread: Optional[threading.Thread] = None
        self.is_alive_flag = False
        self.is_connected = False
        self.stopped = False

        # Motion detection params
        self.motion_threshold = motion_threshold
        self.motion_alpha = motion_alpha
        self.min_motion_pixels = min_motion_pixels

        # Bookkeeping tái kết nối: 0 = thử lại vô hạn. Chỉ các Auto reader mới
        # giới hạn số lần thử của reader chính để kịp fallback sang reader phụ.
        self.max_connect_attempts = int(max_connect_attempts or 0)
        self._connect_failures = 0
        self._ever_connected = False

    def _is_valid_uuid(self, val_str: str) -> bool:
        try:
            uuid.UUID(val_str)
            return True
        except ValueError:
            return False

    # -- lifecycle ----------------------------------------------------------
    def start(self):
        self._thread = threading.Thread(
            target=self._safe_loop, name=f"rd-{self.cam_name}", daemon=True)
        self._thread.start()

    def stop(self):
        self.stopped = True
        self.is_alive_flag = False
        self._cleanup()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def is_stopped(self) -> bool:
        return self.stopped or self.shutdown.is_set()

    def alive(self) -> bool:
        return self.is_alive_flag and self._thread is not None and self._thread.is_alive()

    # -- retry bookkeeping --------------------------------------------------
    def _register_connect_success(self):
        """Kết nối thành công: xoá bộ đếm lỗi, mở lại retry vô hạn."""
        self._connect_failures = 0
        self._ever_connected = True

    def _register_connect_failure(self):
        self._connect_failures += 1

    def _should_give_up(self) -> bool:
        """
        Chỉ bỏ cuộc khi CHƯA từng kết nối được lần nào và đã vượt hạn mức thử.
        Camera đã từng chạy thì luôn thử lại vô hạn (không bao giờ chết thread).
        """
        if self.max_connect_attempts <= 0 or self._ever_connected:
            return False
        return self._connect_failures >= self.max_connect_attempts

    def _backoff_sleep(self, base: float = 2.0, cap: float = 30.0):
        """
        Ngủ có giới hạn tăng dần nhưng vẫn phản hồi shutdown ngay lập tức
        (ngủ từng nhịp 0.25s thay vì block cả chục giây).
        """
        delay = min(cap, base * (2 ** max(0, self._connect_failures - 1)))
        deadline = time.monotonic() + delay
        while time.monotonic() < deadline:
            if self.is_stopped():
                return
            time.sleep(0.25)

    # -- internal -----------------------------------------------------------
    def _safe_loop(self):
        self.is_alive_flag = True
        try:
            self._read_loop()
        except Exception as exc:
            LOGGER.error(f"[{self.cam_name}] Reader crashed: {exc}")
        finally:
            self.is_alive_flag = False
            self.is_connected = False

    def _read_loop(self):
        raise NotImplementedError

    def _cleanup(self):
        """Override in subclasses for resource cleanup."""
        pass

    # -- helpers ------------------------------------------------------------
    def _enqueue(self, frame: Optional[np.ndarray], motion: bool):
        self.frame_count += 1
        pkt = FramePacket(
            cam_id=self.cam_id,
            cam_name=self.cam_name,
            timestamp=time.time(),
            frame_uuid=str(uuid.uuid1(node=self._node)),
            frame=frame,
            frame_count=self.frame_count,
            motion_detected=motion,
        )
        try:
            self.output_queue.put(pkt, block=False)
        except queue.Full:
            # Drop-oldest: ưu tiên frame mới, nhưng ĐẾM lại để engine báo quá tải.
            self.dropped_frames += 1
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.output_queue.put(pkt, block=False)
            except queue.Full:
                pass

        if self.frame_count % 500 == 0:
            LOGGER.info(
                f"[{self.cam_name}] [HEARTBEAT] frame={self.frame_count} "
                f"motion={motion} dropped={self.dropped_frames}")


def nv12_to_bgr_torch(nv12_tensor, H, W, max_dim=1280):
    """
    Optimized NV12 to BGR conversion running entirely on GPU using PyTorch.
    Downscales on GPU to max_dim if higher resolution to minimize PCIe GPU->CPU memory transfers.
    """
    import torch.nn.functional as F

    # Chỉ ép kiểu float TRÊN TỪNG MẶT PHẲNG cần dùng, không .float() cả buffer
    # NV12 (tiết kiệm ~1.5x VRAM và một lần ghi toàn khung mỗi frame).
    # fp16 đủ chính xác cho khoảng giá trị 0..255 và nhanh hơn trên tensor core.
    dt = torch.float16 if nv12_tensor.is_cuda else torch.float32
    Y = nv12_tensor[:H, :].to(dt)
    UV = nv12_tensor[H:, :]
    U = UV[:, 0::2].to(dt)
    V = UV[:, 1::2].to(dt)

    if max_dim is not None and (W > max_dim or H > max_dim):
        scale = float(max_dim) / float(max(H, W))
        # Giữ kích thước chẵn để JPEG/YOLO downstream không phải pad.
        new_H = max(2, int(H * scale) & ~1)
        new_W = max(2, int(W * scale) & ~1)
        Y = F.interpolate(Y[None, None], size=(new_H, new_W),
                          mode='bilinear', align_corners=False)[0, 0]
        H, W = new_H, new_W

    # Upsample chroma trực tiếp về (H, W) — cỡ đích, không phải cỡ gốc.
    U_up = F.interpolate(U[None, None], size=(H, W),
                         mode='bilinear', align_corners=False)[0, 0].sub_(128.0)
    V_up = F.interpolate(V[None, None], size=(H, W),
                         mode='bilinear', align_corners=False)[0, 0].sub_(128.0)

    # Ghi thẳng 3 kênh vào một buffer (3, H, W) liền mạch: bỏ được torch.stack
    # (vốn cấp phát thêm một bản sao toàn khung) và 3 tensor trung gian B/G/R.
    out = torch.empty((3, H, W), dtype=dt, device=Y.device)
    torch.add(Y, U_up, alpha=1.772, out=out[0])                  # B
    torch.add(Y, V_up, alpha=1.402, out=out[2])                  # R
    g = out[1]
    torch.mul(U_up, -0.344136, out=g)
    g.add_(V_up, alpha=-0.714136).add_(Y)                        # G

    # (3,H,W) -> (H,W,3); .contiguous() ngay trong .byte() bằng một lần copy.
    return out.clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).contiguous()


# Lock used only when creating GPU decoders (thread-safe init)
_decoder_creation_lock = threading.Lock()


class NVDECNotSupportedError(Exception):
    """Raised when NVDEC hardware video decoder is not available or supported by GPU."""
    pass


# ---------------------------------------------------------------------------
# RTSP GPU Reader
# ---------------------------------------------------------------------------
class RTSPGPUReader(BaseCameraReader):
    """Reads RTSP stream, decodes on GPU (NVDEC), motion-detects on GPU."""

    def __init__(self, cam_id, rtsp_url, output_queue, shutdown_event, gpuid=0, **kwargs):
        super().__init__(cam_id, output_queue, shutdown_event, **kwargs)
        self.rtsp_url = rtsp_url
        self.gpuid = gpuid

    @torch.no_grad()
    def _read_loop(self):
        import av
        import torch.nn.functional as F
        import PyNvVideoCodec as nvc
        from services.motion_detector import GPUMotionDetector

        torch.cuda.set_device(self.gpuid)
        gpu_detector = GPUMotionDetector(
            width=640, height=360,
            threshold=self.motion_threshold,
            alpha=self.motion_alpha,
            min_motion_pixels=self.min_motion_pixels,
            dtype=torch.float16,
        )

        while not self.is_stopped():
            container = None
            gpu_decoder = None
            try:
                container = av.open(
                    self.rtsp_url,
                    options={
                        'rtsp_transport': 'tcp',
                        'fflags': 'nobuffer+discardcorrupt',
                        'flags': 'low_delay',
                        'err_detect': 'ignore_err',
                        'max_delay': '500000',
                        'reorder_queue_size': '1',
                        'stimeout': '15000000',
                        'rw_timeout': '15000000'
                    },
                    timeout=15,
                )
                vstream = container.streams.video[0]
                codec_name = vstream.name
                codec_id = nvc.cudaVideoCodec.H264
                filt = "h264_mp4toannexb"
                if codec_name in ("hevc", "h265"):
                    codec_id = nvc.cudaVideoCodec.HEVC
                    filt = "hevc_mp4toannexb"

                bsf = av.BitStreamFilterContext(filt, vstream)
                with _decoder_creation_lock:
                    try:
                        gpu_decoder = nvc.CreateDecoder(
                            gpuid=self.gpuid, codec=codec_id, usedevicememory=True)
                    except Exception as exc:
                        err_str = str(exc)
                        if "Decoder not initialized" in err_str or "Error code : 3" in err_str or "NvDecoder" in err_str:
                            LOGGER.error(f"[{self.cam_name}] NVDEC hardware decoder creation failed: {err_str}")
                            raise NVDECNotSupportedError(err_str)
                        raise

                LOGGER.info(
                    f"[{self.cam_name}] RTSP GPU connected ({codec_name} "
                    f"{vstream.width}x{vstream.height})")

                self.is_connected = True
                self._register_connect_success()
                last_decode_t = time.time()
                last_frame_t = time.time()
                target_frame_interval = 1.0 / float(getattr(settings, 'TARGET_FPS', 25))
                stall_timeout = float(getattr(settings, 'DECODE_STALL_TIMEOUT', 10.0))

                for packet in container.demux(vstream):
                    if self.is_stopped():
                        break
                    # Watchdog ở CẤP DEMUX: bắt được cả trường hợp packet vẫn về
                    # nhưng decoder không sinh frame nào (trước đây check nằm bên
                    # trong vòng frame nên không bao giờ chạy khi decoder treo).
                    if time.time() - last_decode_t > stall_timeout:
                        raise RuntimeError(
                            f"No decoded frames for {stall_timeout:.0f}s (decoder stalled)")
                    if packet.size == 0 or packet.dts is None:
                        continue
                    try:
                        filtered = bsf.filter(packet)
                    except Exception:
                        continue

                    for fp in filtered:
                        if fp.size == 0:
                            continue
                        p = nvc.PacketData()
                        p.bsl_data = fp.buffer_ptr
                        p.bsl = fp.buffer_size
                        p.pts = fp.pts or 0
                        p.dts = fp.dts or 0
                        p.key = fp.is_keyframe

                        try:
                            frames = gpu_decoder.Decode(p)
                            if len(frames) > 0:
                                last_decode_t = time.time()
                                now = time.time()
                                elapsed = now - last_frame_t
                                if elapsed < target_frame_interval:
                                    del frames
                                    continue  # Bỏ qua frame trùng FPS mà KHÔNG SLEEP để không bao giờ bị trễ buffer RTSP
                                last_frame_t = now

                                raw = torch.from_dlpack(frames[-1])

                                if getattr(settings, 'USE_MOTION_DETECTOR', False):
                                    y_ch = raw[:vstream.height, :]
                                    y4d = y_ch.float().unsqueeze(0).unsqueeze(0)
                                    resized = F.interpolate(
                                        y4d, size=(360, 640),
                                        mode='bilinear', align_corners=False
                                    ).squeeze()
                                    motion, _ = gpu_detector.detect(resized)

                                    if motion:
                                        gpu_bgr = nv12_to_bgr_torch(raw, vstream.height, vstream.width)
                                        cpu_bgr = gpu_bgr.cpu().numpy()
                                        self._enqueue(cpu_bgr, True)
                                    else:
                                        self._enqueue(None, False)
                                else:
                                    gpu_bgr = nv12_to_bgr_torch(raw, vstream.height, vstream.width)
                                    cpu_bgr = gpu_bgr.cpu().numpy()
                                    self._enqueue(cpu_bgr, True)

                                del raw
                            del frames
                        except Exception as de:
                            err_de_str = str(de)
                            if "Decoder not initialized" in err_de_str or "Error code : 3" in err_de_str or "NvDecoder" in err_de_str:
                                LOGGER.error(f"[{self.cam_name}] NVDEC decode failure: {err_de_str}")
                                raise NVDECNotSupportedError(err_de_str)
                            LOGGER.error(f"[{self.cam_name}] decode err: {de}")
                            gpu_detector.reset()
                            break

            except NVDECNotSupportedError:
                raise
            except Exception as exc:
                err_str = str(exc)
                if "Decoder not initialized" in err_str or "Error code : 3" in err_str or "NvDecoder" in err_str:
                    LOGGER.error(f"[{self.cam_name}] NVDEC error: {err_str}")
                    raise NVDECNotSupportedError(err_str)
                if "1414092869" in err_str or "Immediate exit" in err_str or "AVError" in err_str:
                    LOGGER.warning(f"[{self.cam_name}] RTSP GPU stream reconnecting (network timeout/glitch)...")
                else:
                    LOGGER.error(f"[{self.cam_name}] RTSP GPU error: {exc}")
            finally:
                if container:
                    try:
                        container.close()
                    except Exception:
                        pass
                gpu_decoder = None
                self.is_connected = False

            self._register_connect_failure()
            if self._should_give_up():
                LOGGER.warning(
                    f"[{self.cam_name}] RTSP GPU: bỏ cuộc sau "
                    f"{self._connect_failures} lần thử, nhường cho reader dự phòng")
                return
            if not self.is_stopped():
                self._backoff_sleep()

# ---------------------------------------------------------------------------
# RTSP CPU Reader
# ---------------------------------------------------------------------------
class RTSPCPUReader(BaseCameraReader):
    """Reads RTSP stream, decodes on CPU, motion-detects on CPU."""

    def __init__(self, cam_id, rtsp_url, output_queue, shutdown_event, **kwargs):
        super().__init__(cam_id, output_queue, shutdown_event, **kwargs)
        self.rtsp_url = rtsp_url

    def _read_loop(self):
        import av
        from services.motion_detector import CPUMotionDetector

        cpu_det = CPUMotionDetector(
            threshold=self.motion_threshold,
            alpha=self.motion_alpha,
            min_motion_pixels=self.min_motion_pixels,
        )

        while not self.is_stopped():
            container = None
            try:
                container = av.open(
                    self.rtsp_url,
                    options={
                        'rtsp_transport': 'tcp',
                        'fflags': 'nobuffer+discardcorrupt',
                        'err_detect': 'ignore_err',
                        'max_delay': '500000',
                        'reorder_queue_size': '1',
                        'stimeout': '15000000',
                        'rw_timeout': '15000000'
                    },
                    timeout=15,
                )
                stream = container.streams.video[0]
                LOGGER.info(f"[{self.cam_name}] RTSP CPU connected")
                self.is_connected = True
                self._register_connect_success()
                last_frame_t = time.time()
                target_frame_interval = 1.0 / float(getattr(settings, 'TARGET_FPS', 25))

                for packet in container.demux(stream):
                    if self.is_stopped():
                        break
                    for frame in packet.decode():
                        if self.is_stopped():
                            break

                        now = time.time()
                        elapsed = now - last_frame_t
                        if elapsed < target_frame_interval:
                            continue  # Bỏ qua frame trùng FPS mà KHÔNG SLEEP để không bao giờ bị trễ buffer RTSP
                        last_frame_t = now

                        img = frame.to_ndarray(format='bgr24')
                        if getattr(settings, 'USE_MOTION_DETECTOR', False):
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            gray_sm = cv2.resize(gray, (320, 180))
                            motion, _ = cpu_det.detect(gray_sm)
                            self._enqueue(img if motion else None, motion)
                        else:
                            self._enqueue(img, True)

            except Exception as exc:
                err_str = str(exc)
                if "1414092869" in err_str or "Immediate exit" in err_str or "AVError" in err_str:
                    LOGGER.warning(f"[{self.cam_name}] RTSP CPU stream reconnecting (network timeout/glitch)...")
                else:
                    LOGGER.error(f"[{self.cam_name}] RTSP CPU error: {exc}")
            finally:
                self.is_connected = False
                if container:
                    try:
                        container.close()
                    except Exception:
                        pass

            self._register_connect_failure()
            if self._should_give_up():
                LOGGER.warning(
                    f"[{self.cam_name}] RTSP CPU: bỏ cuộc sau "
                    f"{self._connect_failures} lần thử")
                return
            if not self.is_stopped():
                self._backoff_sleep()

# ---------------------------------------------------------------------------
# Dahua SDK GPU Reader
# ---------------------------------------------------------------------------
class DahuaGPUReader(BaseCameraReader):
    """Reads Dahua camera via SDK, decodes on GPU with motion detection."""

    def __init__(self, cam_id, output_queue, shutdown_event,
                 url, port, user, pwd, channel, gpuid=0, codec='h264', **kwargs):
        super().__init__(cam_id, output_queue, shutdown_event, **kwargs)
        self.url = url
        self.port = port
        self.user = user
        self.pwd = pwd
        self.channel = channel
        self.gpuid = gpuid
        self.codec = codec
        self._viewer = None

    def _read_loop(self):
        from services.sdk.dha_sdk_realplay import DahuaGPUCameraViewer

        port = int(self.port) if self.port else 37777
        ch = (int(self.channel) - 1) if self.channel else 0
        stall_timeout = float(getattr(settings, 'SDK_STALL_TIMEOUT', 30.0))

        # Vòng ngoài: login/start_stream thất bại KHÔNG còn làm chết thread reader.
        # Trước đây `return` ở đây khiến camera Dahua offline vĩnh viễn cho tới khi
        # engine restart; giờ luôn thử lại với backoff cho đến khi stop().
        while not self.is_stopped():
            self._viewer = DahuaGPUCameraViewer(
                codec=self.codec, gpuid=self.gpuid, motion_detect=True,
                motion_threshold=self.motion_threshold,
                motion_alpha=self.motion_alpha,
                min_motion_pixels=self.min_motion_pixels,
            )
            try:
                connected = self._viewer.connect(self.url, port, self.user, self.pwd)
                if not connected or not self._viewer.is_connected:
                    raise RuntimeError(f"Dahua GPU SDK login failed to {self.url}:{port}")
                if not self._viewer.start_stream(ch, stream_type=0) or not self._viewer.is_playing:
                    raise RuntimeError(f"Dahua GPU SDK start_stream failed on channel {ch}")

                LOGGER.info(f"[{self.cam_name}] Dahua GPU SDK connected")
                self.is_connected = True
                self._register_connect_success()

                last_progress_t = time.monotonic()
                last_seen_count = -1
                while not self.is_stopped() and self._viewer.is_playing:
                    frame, motion = self._viewer.decode_next_packet()
                    if not getattr(settings, 'USE_MOTION_DETECTOR', False):
                        motion = True
                    if motion and frame is not None:
                        self._enqueue(frame, True)
                    elif not motion:
                        self._enqueue(None, False)
                    else:
                        time.sleep(0.005)

                    # Watchdog: SDK báo is_playing nhưng bộ đếm frame đứng im
                    # (mất stream ngầm) -> ép reconnect thay vì spin vô hạn.
                    seq = getattr(self._viewer, 'frame_count', 0)
                    if seq != last_seen_count:
                        last_seen_count = seq
                        last_progress_t = time.monotonic()
                    elif time.monotonic() - last_progress_t > stall_timeout:
                        raise RuntimeError(
                            f"Dahua GPU SDK stalled: no new frame for {stall_timeout:.0f}s")

            except Exception as exc:
                LOGGER.error(f"[{self.cam_name}] Dahua GPU SDK error: {exc}")
            finally:
                self.is_connected = False
                self._release_viewer()

            self._register_connect_failure()
            if self._should_give_up():
                LOGGER.warning(
                    f"[{self.cam_name}] Dahua GPU SDK: bỏ cuộc sau "
                    f"{self._connect_failures} lần thử, nhường cho reader dự phòng")
                return
            if not self.is_stopped():
                self._backoff_sleep()

    def _release_viewer(self):
        if self._viewer:
            try:
                self._viewer.cleanup()
            except Exception:
                pass
            self._viewer = None

    def _cleanup(self):
        self._release_viewer()


# ---------------------------------------------------------------------------
# Dahua SDK CPU Reader
# ---------------------------------------------------------------------------
class DahuaCPUReader(BaseCameraReader):
    """Reads Dahua camera via SDK, decodes on CPU."""

    def __init__(self, cam_id, output_queue, shutdown_event,
                 url, port, user, pwd, channel, codec='h264', **kwargs):
        super().__init__(cam_id, output_queue, shutdown_event, **kwargs)
        self.url = url
        self.port = port
        self.user = user
        self.pwd = pwd
        self.channel = channel
        self.codec = codec
        self._viewer = None

    def _read_loop(self):
        from services.sdk.dha_sdk_realplay import DahuaCameraViewer
        from services.motion_detector import CPUMotionDetector

        cpu_det = CPUMotionDetector(
            threshold=self.motion_threshold,
            alpha=self.motion_alpha,
            min_motion_pixels=self.min_motion_pixels,
        )
        port = int(self.port) if self.port else 37777
        ch = (int(self.channel) - 1) if self.channel else 0
        target_fps = max(0.1, float(getattr(settings, 'TARGET_FPS', 15)))
        target_interval = 1.0 / target_fps
        stall_timeout = float(getattr(settings, 'SDK_STALL_TIMEOUT', 30.0))

        # Vòng ngoài tái kết nối: login thất bại không còn giết thread reader.
        while not self.is_stopped():
            self._viewer = DahuaCameraViewer(codec=self.codec)
            cpu_det.reset()
            try:
                connected = self._viewer.connect(self.url, port, self.user, self.pwd)
                if not connected or not self._viewer.is_connected:
                    raise RuntimeError(f"Dahua CPU SDK login failed to {self.url}:{port}")
                if not self._viewer.start_stream(ch, stream_type=0) or not self._viewer.is_playing:
                    raise RuntimeError(f"Dahua CPU SDK start_stream failed on channel {ch}")

                LOGGER.info(f"[{self.cam_name}] Dahua CPU SDK connected")
                self.is_connected = True
                self._register_connect_success()

                last_enqueue_at = float('-inf')
                last_viewer_seq = -1
                last_progress_t = time.monotonic()
                while not self.is_stopped() and self._viewer.is_playing:
                    viewer_seq = self._viewer.get_frame_count()
                    now = time.monotonic()

                    if viewer_seq != last_viewer_seq:
                        last_progress_t = now
                    elif now - last_progress_t > stall_timeout:
                        raise RuntimeError(
                            f"Dahua CPU SDK stalled: no new frame for {stall_timeout:.0f}s")

                    # Không copy/enqueue lại cùng một frame SDK; đồng thời giới hạn TARGET_FPS.
                    if viewer_seq == last_viewer_seq or now - last_enqueue_at < target_interval:
                        time.sleep(0.002)
                        continue
                    frame = self._viewer.get_frame()
                    if frame is not None:
                        last_viewer_seq = viewer_seq
                        last_enqueue_at = now
                        if getattr(settings, 'USE_MOTION_DETECTOR', False):
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            gray_sm = cv2.resize(gray, (320, 180))
                            motion, _ = cpu_det.detect(gray_sm)
                            self._enqueue(frame if motion else None, motion)
                        else:
                            self._enqueue(frame, True)
                    else:
                        time.sleep(0.005)

            except Exception as exc:
                LOGGER.error(f"[{self.cam_name}] Dahua CPU SDK error: {exc}")
            finally:
                self.is_connected = False
                self._release_viewer()

            self._register_connect_failure()
            if self._should_give_up():
                LOGGER.warning(
                    f"[{self.cam_name}] Dahua CPU SDK: bỏ cuộc sau "
                    f"{self._connect_failures} lần thử")
                return
            if not self.is_stopped():
                self._backoff_sleep()

    def _release_viewer(self):
        if self._viewer:
            try:
                self._viewer.cleanup()
            except Exception:
                pass
            self._viewer = None

    def _cleanup(self):
        self._release_viewer()


# ---------------------------------------------------------------------------
# Dahua Auto Reader (Dahua GPU NVDEC Primary -> Dahua CPU Fallback)
# ---------------------------------------------------------------------------
class DahuaAutoReader(BaseCameraReader):
    """
    Dahua Auto Reader:
    1. Primary: Attempts Dahua GPU Reader (DahuaGPUReader - PyNvVideoCodec NVDEC).
    2. Fallback: If GPU decode fails or produces 0 frames / format error,
       automatically falls back to Dahua CPU Reader (FFmpeg pipe auto-probing).
    """

    def __init__(self, cam_id, output_queue, shutdown_event,
                 url, port, user, pwd, channel=1, codec='h264', gpuid=0, **kwargs):
        self._active_reader = None
        super().__init__(cam_id, output_queue, shutdown_event, **kwargs)
        self.url = url
        self.port = port
        self.user = user
        self.pwd = pwd
        self.channel = channel
        self.codec = codec
        self.gpuid = gpuid
        self.reader_kwargs = kwargs

    @property
    def is_connected(self):
        active = getattr(self, '_active_reader', None)
        if active:
            return getattr(active, 'is_connected', False)
        return getattr(self, '_is_connected_fallback', False)

    @is_connected.setter
    def is_connected(self, val):
        self._is_connected_fallback = bool(val)
        active = getattr(self, '_active_reader', None)
        if active:
            active.is_connected = val

    @property
    def frame_count(self):
        active = getattr(self, '_active_reader', None)
        if active:
            return getattr(active, 'frame_count', 0)
        return getattr(self, '_frame_count_fallback', 0)

    @frame_count.setter
    def frame_count(self, val):
        self._frame_count_fallback = int(val)
        active = getattr(self, '_active_reader', None)
        if active:
            active.frame_count = val

    def _read_loop(self):
        gpu_available = False
        try:
            gpu_available = torch.cuda.is_available()
        except Exception:
            pass

        force_cpu = getattr(settings, 'FORCE_CPU_DECODE', False)
        primary_attempts = int(getattr(settings, 'PRIMARY_CONNECT_ATTEMPTS', 2))

        # 1. Primary Attempt: Dahua GPU Reader (PyNvVideoCodec NVDEC)
        if gpu_available and not force_cpu:
            LOGGER.info(f"[{self.cam_name}] Dahua Primary Attempt: Dahua GPU Reader ({self.url}:{self.port})...")
            gpu_reader = DahuaGPUReader(
                self.cam_id, self.output_queue, self.shutdown,
                url=self.url, port=self.port, user=self.user, pwd=self.pwd,
                channel=self.channel, codec=self.codec, gpuid=self.gpuid,
                name=self.cam_name,
                motion_threshold=self.motion_threshold,
                motion_alpha=self.motion_alpha,
                min_motion_pixels=self.min_motion_pixels,
                max_connect_attempts=primary_attempts,
            )
            self._active_reader = gpu_reader
            try:
                gpu_reader._read_loop()
                if self.is_stopped():
                    return
                if gpu_reader._ever_connected:
                    # Đã từng chạy được bằng GPU: reader GPU tự retry vô hạn, việc
                    # nó trả về ở đây nghĩa là stop() — không cần fallback.
                    return
            except Exception as e:
                LOGGER.warning(f"[{self.cam_name}] Dahua GPU reader failed: {e}. Falling back to Dahua CPU Reader...")

        # 2. Fallback Attempt: Dahua CPU Reader (FFmpeg pipe)
        LOGGER.info(f"[{self.cam_name}] Dahua Fallback Attempt: Dahua CPU Reader ({self.url}:{self.port})...")
        cpu_reader = DahuaCPUReader(
            self.cam_id, self.output_queue, self.shutdown,
            url=self.url, port=self.port, user=self.user, pwd=self.pwd,
            channel=self.channel, codec=self.codec,
            name=self.cam_name,
            motion_threshold=self.motion_threshold,
            motion_alpha=self.motion_alpha,
            min_motion_pixels=self.min_motion_pixels,
        )
        self._active_reader = cpu_reader
        cpu_reader._read_loop()

    def _cleanup(self):
        active = getattr(self, '_active_reader', None)
        if active:
            try:
                active.stop()
                active._cleanup()
            except Exception:
                pass
            self._active_reader = None


# ---------------------------------------------------------------------------
# HIK SDK Reader (CPU Decoding)
# ---------------------------------------------------------------------------
class HIKReader(BaseCameraReader):
    """Reads Hikvision camera via SDK, decodes on CPU."""

    def __init__(self, cam_id, output_queue, shutdown_event,
                 url, port, user, pwd, channel, codec='h264', **kwargs):
        super().__init__(cam_id, output_queue, shutdown_event, **kwargs)
        self.codec = codec
        self.url = url
        self.port = port
        self.user = user
        self.pwd = pwd
        self.channel = channel
        self._viewer = None

    def _read_loop(self):
        from services.sdk.hik_sdk_realplay import HIKCameraViewer
        from services.motion_detector import CPUMotionDetector

        cpu_det = CPUMotionDetector(
            threshold=self.motion_threshold,
            alpha=self.motion_alpha,
            min_motion_pixels=self.min_motion_pixels,
        )
        target_fps = max(0.1, float(getattr(settings, 'TARGET_FPS', 15)))
        target_interval = 1.0 / target_fps
        stall_timeout = float(getattr(settings, 'SDK_STALL_TIMEOUT', 30.0))

        # Vòng ngoài tái kết nối: connect/stream thất bại không còn giết thread.
        while not self.is_stopped():
            self._viewer = HIKCameraViewer()
            cpu_det.reset()
            try:
                if not self._viewer.connect(self.url, self.port, self.user, self.pwd):
                    raise RuntimeError("HIK connect failed")
                if not self._viewer.start_stream(self.channel, 0):
                    raise RuntimeError("HIK stream failed")
                time.sleep(2)
                LOGGER.info(f"[{self.cam_name}] HIK SDK connected")
                self.is_connected = True
                self._register_connect_success()

                last_enqueue_at = float('-inf')
                last_viewer_seq = -1
                last_progress_t = time.monotonic()
                while not self.is_stopped():
                    viewer_seq = self._viewer.get_frame_count()
                    now = time.monotonic()

                    if viewer_seq != last_viewer_seq:
                        last_progress_t = now
                    elif now - last_progress_t > stall_timeout:
                        raise RuntimeError(
                            f"HIK SDK stalled: no new frame for {stall_timeout:.0f}s")

                    if viewer_seq == last_viewer_seq or now - last_enqueue_at < target_interval:
                        time.sleep(0.002)
                        continue
                    frame = self._viewer.get_frame()
                    if frame is not None:
                        last_viewer_seq = viewer_seq
                        last_enqueue_at = now
                        if getattr(settings, 'USE_MOTION_DETECTOR', False):
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            gray_sm = cv2.resize(gray, (320, 180))
                            motion, _ = cpu_det.detect(gray_sm)
                            self._enqueue(frame if motion else None, motion)
                        else:
                            self._enqueue(frame, True)
                    else:
                        time.sleep(0.005)

            except Exception as exc:
                LOGGER.error(f"[{self.cam_name}] HIK SDK error: {exc}")
            finally:
                self.is_connected = False
                self._release_viewer()

            self._register_connect_failure()
            if self._should_give_up():
                LOGGER.warning(
                    f"[{self.cam_name}] HIK SDK: bỏ cuộc sau "
                    f"{self._connect_failures} lần thử")
                return
            if not self.is_stopped():
                self._backoff_sleep()

    def _release_viewer(self):
        if self._viewer:
            try:
                self._viewer.stop_stream()
                self._viewer.cleanup()
            except Exception:
                pass
            self._viewer = None

    def _cleanup(self):
        self._release_viewer()


# ---------------------------------------------------------------------------
# HIK Hybrid Reader (RTSP GPU Primary -> HIK CPU SDK Fallback)
# ---------------------------------------------------------------------------
class HIKAutoReader(BaseCameraReader):
    """
    Hikvision Hybrid Reader:
    1. Primary: Attempts RTSP GPU Reader (RTSPGPUReader - PyNvVideoCodec NVDEC) for high performance & zero CPU overhead.
    2. Fallback: If RTSP GPU fails to connect, automatically falls back to HIK CPU SDK (HIKReader).
    """

    def __init__(self, cam_id, output_queue, shutdown_event,
                 url, port, user, pwd, channel, rtsp_url="", gpuid=0, codec='h264', **kwargs):
        super().__init__(cam_id, output_queue, shutdown_event, **kwargs)
        self.url = url
        self.port = port
        self.user = user
        self.pwd = pwd
        self.channel = channel
        self.gpuid = gpuid
        self.codec = codec

        # Auto-construct RTSP URL if not provided
        if not rtsp_url:
            chan_str = f"{channel}01" if channel else "101"
            rtsp_url = f"rtsp://{user}:{pwd}@{url}:554/Streaming/Channels/{chan_str}"
        self.rtsp_url = rtsp_url
        self._active_reader = None

    def _read_loop(self):
        gpu_available = False
        try:
            gpu_available = torch.cuda.is_available()
        except Exception:
            pass

        primary_attempts = int(getattr(settings, 'PRIMARY_CONNECT_ATTEMPTS', 2))

        # 1. Primary Attempt: RTSP GPU Mode (PyNvVideoCodec NVDEC)
        if gpu_available:
            LOGGER.info(f"[{self.cam_name}] HIK Primary Attempt: RTSP GPU Reader ({self.rtsp_url})...")
            gpu_reader = RTSPGPUReader(
                self.cam_id, self.rtsp_url, self.output_queue, self.shutdown,
                gpuid=self.gpuid, name=self.cam_name,
                motion_threshold=self.motion_threshold,
                motion_alpha=self.motion_alpha,
                min_motion_pixels=self.min_motion_pixels,
                max_connect_attempts=primary_attempts,
            )
            self._active_reader = gpu_reader
            try:
                gpu_reader._read_loop()
                if self.is_stopped():
                    return
                if gpu_reader._ever_connected:
                    return
            except Exception as e:
                LOGGER.warning(f"[{self.cam_name}] RTSP GPU failed: {e}. Falling back to HIK CPU SDK...")

        # 2. Fallback: HIK CPU SDK Reader
        LOGGER.info(f"[{self.cam_name}] HIK Fallback Attempt: HIK CPU SDK Reader...")
        cpu_reader = HIKReader(
            self.cam_id, self.output_queue, self.shutdown,
            url=self.url, port=self.port, user=self.user, pwd=self.pwd,
            channel=self.channel, codec=self.codec, name=self.cam_name,
            motion_threshold=self.motion_threshold,
            motion_alpha=self.motion_alpha,
            min_motion_pixels=self.min_motion_pixels,
        )
        self._active_reader = cpu_reader
        cpu_reader._read_loop()

    @property
    def is_connected(self):
        active = getattr(self, '_active_reader', None)
        if active:
            return getattr(active, 'is_connected', False)
        return getattr(self, '_is_connected_fallback', False)

    @is_connected.setter
    def is_connected(self, val):
        self._is_connected_fallback = bool(val)
        active = getattr(self, '_active_reader', None)
        if active:
            active.is_connected = val

    @property
    def frame_count(self):
        active = getattr(self, '_active_reader', None)
        if active:
            return getattr(active, 'frame_count', 0)
        return getattr(self, '_frame_count_fallback', 0)

    @frame_count.setter
    def frame_count(self, val):
        self._frame_count_fallback = int(val)
        active = getattr(self, '_active_reader', None)
        if active:
            active.frame_count = val

    def _cleanup(self):
        active = getattr(self, '_active_reader', None)
        if active:
            try:
                active.stop()
                active._cleanup()
            except Exception:
                pass
            self._active_reader = None


# ---------------------------------------------------------------------------
# RTSP Auto Reader (RTSP GPU NVDEC Primary -> RTSP CPU Fallback)
# ---------------------------------------------------------------------------
class RTSPAutoReader(BaseCameraReader):
    """
    RTSP Auto Reader:
    1. Primary: Attempts RTSP GPU Reader (RTSPGPUReader - PyNvVideoCodec NVDEC) for high performance.
    2. Fallback: If NVDEC hardware decoder is unavailable/unsupported (e.g. P106-100 GPU or session limit),
       automatically falls back to RTSP CPU Reader (PyAV CPU Decoding).
    """

    def __init__(self, cam_id, rtsp_url, output_queue, shutdown_event, gpuid=0, **kwargs):
        self._active_reader = None
        super().__init__(cam_id, output_queue, shutdown_event, **kwargs)
        self.rtsp_url = rtsp_url
        self.gpuid = gpuid
        self.motion_kw = kwargs

    @property
    def is_connected(self):
        active = getattr(self, '_active_reader', None)
        if active:
            return getattr(active, 'is_connected', False)
        return getattr(self, '_is_connected_fallback', False)

    @is_connected.setter
    def is_connected(self, val):
        self._is_connected_fallback = bool(val)
        active = getattr(self, '_active_reader', None)
        if active:
            active.is_connected = val

    @property
    def frame_count(self):
        active = getattr(self, '_active_reader', None)
        if active:
            return getattr(active, 'frame_count', 0)
        return getattr(self, '_frame_count_fallback', 0)

    @frame_count.setter
    def frame_count(self, val):
        self._frame_count_fallback = int(val)
        active = getattr(self, '_active_reader', None)
        if active:
            active.frame_count = val

    def _read_loop(self):
        gpu_available = False
        try:
            gpu_available = torch.cuda.is_available()
        except Exception:
            pass

        force_cpu = getattr(settings, 'FORCE_CPU_DECODE', False)
        primary_attempts = int(getattr(settings, 'PRIMARY_CONNECT_ATTEMPTS', 2))

        # 1. Primary Attempt: RTSP GPU Mode (PyNvVideoCodec NVDEC)
        if gpu_available and not force_cpu:
            LOGGER.info(f"[{self.cam_name}] RTSP Primary Attempt: RTSP GPU Reader ({self.rtsp_url})...")
            gpu_reader = RTSPGPUReader(
                self.cam_id, self.rtsp_url, self.output_queue, self.shutdown,
                gpuid=self.gpuid, name=self.cam_name,
                motion_threshold=self.motion_threshold,
                motion_alpha=self.motion_alpha,
                min_motion_pixels=self.min_motion_pixels,
                max_connect_attempts=primary_attempts,
            )
            self._active_reader = gpu_reader
            try:
                gpu_reader._read_loop()
                if self.is_stopped():
                    return
                if gpu_reader._ever_connected:
                    return
            except NVDECNotSupportedError as e:
                LOGGER.warning(f"[{self.cam_name}] ⚠️ NVDEC Hardware Decoding unavailable on GPU ({e}). Falling back permanently to RTSP CPU Reader (PyAV)...")
            except Exception as e:
                LOGGER.warning(f"[{self.cam_name}] RTSP GPU failed: {e}. Falling back to RTSP CPU Reader...")

        # 2. Fallback: RTSP CPU Reader
        LOGGER.info(f"[{self.cam_name}] RTSP Fallback Attempt: RTSP CPU Reader (PyAV CPU Decoding)...")
        cpu_reader = RTSPCPUReader(
            self.cam_id, self.rtsp_url, self.output_queue, self.shutdown,
            name=self.cam_name,
            motion_threshold=self.motion_threshold,
            motion_alpha=self.motion_alpha,
            min_motion_pixels=self.min_motion_pixels,
        )
        self._active_reader = cpu_reader
        cpu_reader._read_loop()

    def _cleanup(self):
        if self._active_reader:
            try:
                self._active_reader.stop()
                self._active_reader._cleanup()
            except Exception:
                pass
            self._active_reader = None



# ---------------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------------
def create_reader(cam_id: str, cam_config: dict,
                  output_queue: queue.Queue,
                  shutdown_event: threading.Event,
                  gpuid: int = 0) -> BaseCameraReader:
    """
    Create the appropriate reader based on camera config.

    cam_config keys:
        use_sdk (int), manufacturer (str),
        url (str, RTSP URL), storage_url, storage_port,
        storage_username, storage_password, storage_channel
    """
    use_sdk = cam_config.get('use_sdk', 0)
    manufacturer = (cam_config.get('manufacturer') or '').lower()
    gpu_available = False
    try:
        gpu_available = torch.cuda.is_available()
    except Exception:
        # torch.cuda.is_available() có thể ném RuntimeError/OSError khi driver
        # NVIDIA lỗi, không chỉ ImportError -> bắt rộng để rơi về CPU an toàn.
        pass

    force_cpu = getattr(settings, 'FORCE_CPU_DECODE', False)

    motion_kw = {
        'motion_threshold': cam_config.get('motion_threshold', 12.0),
        'motion_alpha': cam_config.get('motion_alpha', 0.005),
        'min_motion_pixels': cam_config.get('min_motion_pixels', 400),
    }

    cam_name = cam_config.get('name') or cam_id

    if use_sdk == 1:
        codec = cam_config.get('codec') or cam_config.get('stream_codec') or 'h264'
        sdk_kw = dict(
            url=cam_config.get('storage_url') or cam_config.get('url'),
            port=int(cam_config['storage_port']) if cam_config.get('storage_port') else 37777,
            user=cam_config.get('storage_username') or 'admin',
            pwd=cam_config.get('storage_password') or '',
            channel=int(cam_config['storage_channel']) if cam_config.get('storage_channel') else 1,
            codec=codec,
        )
        if manufacturer in ('hik', 'hikvision'):
            rtsp_url = cam_config.get('url') or ''
            return HIKAutoReader(
                cam_id, output_queue, shutdown_event,
                rtsp_url=rtsp_url, name=cam_name, gpuid=gpuid,
                **sdk_kw, **motion_kw)
        else:
            # Dahua, KBVision, Kabevision, Kabe và các hãng OEM Dahua đều dùng Dahua SDK
            return DahuaAutoReader(
                cam_id, output_queue, shutdown_event,
                name=cam_name, gpuid=gpuid, **sdk_kw, **motion_kw)

    # RTSP mode
    rtsp_url = cam_config.get('url', '')
    if gpu_available and not force_cpu:
        return RTSPAutoReader(
            cam_id, rtsp_url, output_queue, shutdown_event,
            name=cam_name, gpuid=gpuid, **motion_kw)
    else:
        return RTSPCPUReader(
            cam_id, rtsp_url, output_queue, shutdown_event,
            name=cam_name, **motion_kw)

