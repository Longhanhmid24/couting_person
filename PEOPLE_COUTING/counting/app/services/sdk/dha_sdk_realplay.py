# coding=utf-8
import queue
import cv2
import numpy as np
import threading
import time
import subprocess
import os
import re
from ctypes import *
import torch
from NetSDK.NetSDK import NetClient
from NetSDK.SDK_Callback import fDisConnect, fHaveReConnect, fRealDataCallBackEx2
from NetSDK.SDK_Enum import SDK_RealPlayType, EM_LOGIN_SPAC_CAP_TYPE, EM_REALDATA_FLAG
from NetSDK.SDK_Struct import (C_LLONG, NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY,
                              NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY, LOG_SET_PRINT_INFO)

decoding_lock = threading.Lock()
sdk_login_lock = threading.Lock()

def nv12_to_bgr_torch(nv12_tensor, H, W, max_dim=1280):
    """
    Optimized NV12 to BGR conversion running entirely on GPU using PyTorch.
    Downscales on GPU to max_dim if higher resolution to minimize PCIe GPU->CPU memory transfers.
    """
    import torch
    import torch.nn.functional as F
    nv12 = nv12_tensor.float()
    Y = nv12[:H, :]
    UV = nv12[H:, :]
    U = UV[:, 0::2]
    V = UV[:, 1::2]

    if max_dim is not None and (W > max_dim or H > max_dim):
        scale = float(max_dim) / float(max(H, W))
        new_H = int(H * scale)
        new_W = int(W * scale)
        Y = F.interpolate(Y.unsqueeze(0).unsqueeze(0), size=(new_H, new_W), mode='bilinear', align_corners=False).squeeze()
        H, W = new_H, new_W
    
    # Bilinear upsample to H x W
    U_up = F.interpolate(U.unsqueeze(0).unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze()
    V_up = F.interpolate(V.unsqueeze(0).unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze()
    
    U_val = U_up - 128.0
    V_val = V_up - 128.0
    
    B = Y + 1.772 * U_val
    G = Y - 0.344136 * U_val - 0.714136 * V_val
    R = Y + 1.402 * V_val
    
    bgr = torch.stack([B, G, R], dim=-1)
    return torch.clamp(bgr, 0, 255).byte()


def detect_nal_codec(raw_bytes: bytes):
    """
    Phát hiện chuẩn xác định dạng codec (H.264 hoặc H.265 / HEVC) từ NAL bitstream header.
    
    Quy tắc phân định chuẩn ISO/IEC 14496-10 (AVC) và ISO/IEC 23008-2 (HEVC):
    1. H.265 (HEVC) Parameter Sets:
       - VPS (Video Parameter Set, nal_type=32): byte0 = 0x40/0x41, byte1 có temporal_id != 0.
         (VPS CHỈ TỒN TẠI trong HEVC, H.264 KHÔNG CÓ VPS).
       - SPS (nal_type=33): byte0 = 0x42/0x43, byte1 có temporal_id != 0.
       - PPS (nal_type=34): byte0 = 0x44/0x45, byte1 có temporal_id != 0.
    2. H.264 (AVC) Parameter Sets:
       - SPS (nal_type=7): (byte0 & 0x1F) == 7 (thường byte0 = 0x27, 0x47, 0x67).
       - PPS (nal_type=8): (byte0 & 0x1F) == 8 (thường byte0 = 0x28, 0x48, 0x68).
    3. IDR frames:
       - H.265 IDR (nal_type=19, 20): byte0 chẵn (0x26 hoặc 0x28) và byte1 = 0x01.
       - H.264 IDR (nal_type=5): (byte0 & 0x1F) == 5 (0x25, 0x45, 0x65).
    """
    if not raw_bytes or len(raw_bytes) < 6:
        return None

    # Quét MỌI NAL trong gói (tối đa 64 NAL). Camera Dahua hay đặt AUD/SEI ở đầu
    # gói, SPS/PPS/VPS nằm phía sau — bản cũ chỉ xem NAL đầu nên không bao giờ
    # thấy parameter set ("Inconclusive NALs after 50 packets").
    pos = 4 if (raw_bytes.startswith(b'DHAV') or raw_bytes.startswith(b'dhav')) else 0
    idr_guess = None
    h264_ps = False
    n = len(raw_bytes)
    for _ in range(64):
        idx = raw_bytes.find(b'\x00\x00\x01', pos)
        if idx == -1 or idx + 4 >= n:
            break
        b0 = raw_bytes[idx + 3]
        b1 = raw_bytes[idx + 4]
        pos = idx + 3

        h264_type = b0 & 0x1F
        h265_type = (b0 >> 1) & 0x3F

        # 1. H.265 Parameter Sets (VPS=32, SPS=33, PPS=34)
        # Chuẩn HEVC: parameter set luôn có nuh_layer_id=0 và TemporalId=0, tức byte0
        # bit0 = 0 và byte1 == 0x01 (nuh_temporal_id_plus1 = 1). Bản cũ chỉ đòi
        # (b1 & 7) != 0 nên P-slice H.264 (0x41 0x9a...) bị nhận nhầm là VPS H.265.
        if h265_type in (32, 33, 34) and (b0 & 0x81) == 0 and b1 == 0x01:
            return 'h265'

        # 2. H.264 Parameter Sets (SPS=7, PPS=8); forbidden_zero_bit = 0. Ghi nhận
        # rồi quét tiếp: 0x28 0x01 cũng là IDR_N_LP của H.265, VPS có thể ở sau.
        if (b0 & 0x80) == 0 and h264_type in (7, 8) and not (b0 == 0x28 and b1 == 0x01):
            h264_ps = True

        # 3./4. IDR slice — chỉ dùng khi cả gói không có parameter set nào
        if idr_guess is None:
            if h265_type in (19, 20) and (b0 & 0x81) == 0 and b1 == 0x01:
                idr_guess = 'h265'
            elif (b0 & 0x80) == 0 and h264_type == 5:
                idr_guess = 'h264'

    if h264_ps:
        return 'h264'
    return idr_guess


class DahuaCameraViewer:
    def __init__(self, codec='h264'):
        self.loginID = C_LLONG()
        self.playID = C_LLONG()
        self.sdk = NetClient()
        self.codec = (codec or 'h264').lower() if codec else 'h264'
        self.m_DisConnectCallBack = fDisConnect(self.on_disconnect)
        self.m_ReConnectCallBack = fHaveReConnect(self.on_reconnect)
        self.m_RealDataCallBack = fRealDataCallBackEx2(self.on_frame_data)
        self.raw_buffer = bytearray()
        self.frame_queue = []
        self.frame_lock = threading.Lock()
        self.is_connected = False
        self.is_playing = False
        self.current_frame = None
        self.frame_count = 0
        self.ffmpeg_process = None
        self.frame_width = None
        self.frame_height = None
        self.resolution_event = threading.Event()
        self.pending_packets = []
        self.decoder_ready = False
        self.device_info = None
        self.sdk.InitEx(self.m_DisConnectCallBack)
        self.sdk.SetAutoReconnect(self.m_ReConnectCallBack)

    def connect(self, ip, port, username, password):
        print(f"Connecting to {ip}:{port}...")
        if self.loginID:
            print("Already connected!")
            return True
        stuInParam = NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY()
        stuInParam.dwSize = sizeof(NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY)
        stuInParam.szIP = ip.encode()
        stuInParam.nPort = int(port) if port else 37777
        stuInParam.szUserName = username.encode()
        stuInParam.szPassword = password.encode()
        stuInParam.emSpecCap = EM_LOGIN_SPAC_CAP_TYPE.TCP
        stuOutParam = NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY()
        stuOutParam.dwSize = sizeof(NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY)
        with sdk_login_lock:
            self.loginID, device_info, error_msg = self.sdk.LoginWithHighLevelSecurity(stuInParam, stuOutParam)
        if self.loginID != 0:
            self.device_info = device_info
            self.is_connected = True
            print(f"Connected successfully! Channels: {device_info.nChanNum}")
            return True
        else:
            print(f"Connection failed: {error_msg}")
            return False

    def start_stream(self, channel=0, stream_type=0):
        if not self.is_connected:
            print("Not connected to camera!")
            return False
        if self.playID:
            print("Stream already started!")
            return True

        # Tự động clamp channel về 0 nếu camera đơn kênh hoặc channel vượt quá nChanNum
        if hasattr(self, 'device_info') and self.device_info and self.device_info.nChanNum > 0:
            if channel >= self.device_info.nChanNum:
                print(f"[Dahua CPU] Channel {channel} exceeds device channel count {self.device_info.nChanNum}. Clamping channel to 0.")
                channel = 0

        print(f"Starting stream on channel {channel}...")
        play_type = SDK_RealPlayType.Realplay if stream_type == 0 else SDK_RealPlayType.Realplay_1
        self.playID = self.sdk.RealPlayEx(self.loginID, channel, 0, play_type)
        if self.playID != 0:
            self.sdk.SetRealDataCallBackEx2(self.playID, self.m_RealDataCallBack, None, EM_REALDATA_FLAG.RAW_DATA)
            self.is_playing = True
            print("Stream started successfully!")
            self.pending_packets = []
            if self.codec != 'auto':
                self.setup_decoder(codec_hint=self.codec)
                self.decoder_ready = True
            else:
                self.decoder_ready = False
                print("Waiting for first video packet to auto-detect codec for FFmpeg...")
            return True
        else:
            print(f"Failed to start stream: {self.sdk.GetLastErrorMessage()}")
            return False

    def setup_decoder(self, codec_hint='h264'):
        try:
            self.resolution_event.clear()
            self.frame_width = None
            self.frame_height = None
            codec_lower = (codec_hint or 'h264').lower()
            if codec_lower in ('hevc', 'h265'):
                input_opts = ['-f', 'hevc',
                              '-fflags', 'nobuffer', '-flags', 'low_delay',
                              '-probesize', '1000000', '-analyzeduration', '1000000']
            else:
                input_opts = ['-f', 'h264',
                              '-fflags', 'nobuffer', '-flags', 'low_delay',
                              '-probesize', '1000000', '-analyzeduration', '1000000']
            cmd = ['ffmpeg', '-hwaccel', 'none'] + input_opts + [
                '-i', 'pipe:0', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                '-an', '-loglevel', 'info', 'pipe:1'
            ]
            self.current_codec = codec_lower
            print(f"Setting up FFmpeg decoder for stream (format={input_opts[1]})...")
            self.ffmpeg_process = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            self.resolution_parser_thread = threading.Thread(target=self._parse_ffmpeg_resolution, daemon=True)
            self.resolution_parser_thread.start()
            self.decoder_thread = threading.Thread(target=self.decode_frames, daemon=True)
            self.decoder_thread.start()
            print("Decoder setup completed. Waiting for resolution...")
        except FileNotFoundError:
            print("❌ FATAL ERROR: 'ffmpeg' command not found.")
            print("Please install FFmpeg and ensure it is in your system's PATH.")
            self.is_playing = False
        except Exception as e:
            print(f"Failed to setup decoder: {e}")
            self.is_playing = False

    def _parse_ffmpeg_resolution(self):
        res_pattern = re.compile(r'Stream #.*: Video:.*, (\d{2,})x(\d{2,})')
        resolution_found = False
        
        for line in iter(self.ffmpeg_process.stderr.readline, b''):
            if not self.is_playing:
                break
                
            line_str = line.decode('utf-8', errors='ignore')

            if not resolution_found:
                match = res_pattern.search(line_str)
                if match:
                    self.frame_width = int(match.group(1))
                    self.frame_height = int(match.group(2))
                    print(f"✅ Resolution detected: {self.frame_width}x{self.frame_height}")
                    self.resolution_event.set()
                    resolution_found = True

        print("FFmpeg stderr reader thread finished.")
        if not resolution_found and not self.resolution_event.is_set():
            print("⚠️ FFmpeg process exited before resolution could be determined.")
            self.resolution_event.set()
            
    def decode_frames(self):
        print("⏳ Decoder thread started, waiting for resolution signal...")
        if not self.resolution_event.wait(timeout=20):
            print("❌ Timed out waiting for resolution. Decoder thread will exit.")
            self.is_playing = False
            return
        if not self.frame_width or not self.frame_height:
            print("⚠️ Resolution could not be determined. Decoder thread will exit.")
            self.is_playing = False
            return
        bytes_per_frame = self.frame_width * self.frame_height * 3
        while self.is_playing and self.ffmpeg_process:
            try:
                frame_data = self.ffmpeg_process.stdout.read(bytes_per_frame)
                if len(frame_data) == bytes_per_frame:
                    frame = np.frombuffer(frame_data, dtype=np.uint8)
                    frame = frame.reshape((self.frame_height, self.frame_width, 3))
                    with self.frame_lock:
                        self.current_frame = frame.copy()
                        self.frame_count += 1
                elif len(frame_data) == 0 and self.ffmpeg_process.poll() is not None:
                    print("FFmpeg process has terminated. Stopping decoder.")
                    break
                else:
                    time.sleep(0.001)
            except Exception as e:
                print(f"Decode error: {e}")
                break
        print("Decoder thread finished.")

    def stop_stream(self):
        if self.playID:
            self.sdk.StopRealPlayEx(self.playID)
            self.playID = 0
        self.is_playing = False
        self.decoder_ready = False
        self.pending_packets.clear()
        if self.ffmpeg_process:
            try:
                if self.ffmpeg_process.stdin: self.ffmpeg_process.stdin.close()
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=2)
            except Exception as e:
                print(f"Error closing ffmpeg process: {e}")
            self.ffmpeg_process = None
        print("Stream stopped")

    def disconnect(self):
        self.stop_stream()
        if self.loginID:
            self.sdk.Logout(self.loginID)
            self.loginID = 0
            self.is_connected = False
            print("Disconnected")

    def cleanup(self):
        self.disconnect()

    def on_disconnect(self, lLoginID, pchDVRIP, nDVRPort, dwUser):
        print("Camera disconnected!")
        self.is_connected = False

    def on_reconnect(self, lLoginID, pchDVRIP, nDVRPort, dwUser):
        print("Camera reconnected!")
        self.is_connected = True

    def on_frame_data(self, lRealHandle, dwDataType, pBuffer, dwBufSize, param, dwUser):
        if lRealHandle == self.playID and dwDataType == 0:
            data = cast(pBuffer, POINTER(c_ubyte * dwBufSize)).contents
            raw_bytes = bytes(data)

            # Lọc bỏ hoàn toàn các gói không phải video (0xf0: audio, 0xf1: metadata/IVS)
            if (raw_bytes.startswith(b'DHAV') or raw_bytes.startswith(b'dhav')) and len(raw_bytes) > 4:
                if raw_bytes[4] not in (0xfc, 0xfd):
                    return

            start_search = 4 if (raw_bytes.startswith(b'DHAV') or raw_bytes.startswith(b'dhav')) else 0
            idx = raw_bytes.find(b'\x00\x00\x00\x01', start_search)
            if idx == -1:
                idx = raw_bytes.find(b'\x00\x00\x01', start_search)
            if idx != -1:
                raw_bytes = raw_bytes[idx:]
                
                # Nếu chưa khởi tạo decoder (do chế độ auto) -> phát hiện codec từ NAL parameter sets
                if not self.decoder_ready:
                    detected = detect_nal_codec(raw_bytes)
                    if detected is not None:
                        detected_codec = 'hevc' if detected == 'h265' else 'h264'
                        print(f"[Dahua CPU] Auto-detected stream codec for FFmpeg: {detected.upper()}")
                    else:
                        self.pending_packets.append(raw_bytes)
                        if len(self.pending_packets) > 50:
                            detected_codec = 'h264'
                            print("[Dahua CPU] Inconclusive NALs after 50 packets, defaulting to H.264 for FFmpeg...")
                        else:
                            return
                    
                    self.setup_decoder(codec_hint=detected_codec)
                    self.decoder_ready = True
                    for p in self.pending_packets:
                        if self.ffmpeg_process and self.ffmpeg_process.stdin and not self.ffmpeg_process.stdin.closed:
                            try:
                                self.ffmpeg_process.stdin.write(p)
                            except Exception:
                                pass
                    self.pending_packets.clear()

                if self.ffmpeg_process and self.ffmpeg_process.stdin and not self.ffmpeg_process.stdin.closed:
                    try:
                        self.ffmpeg_process.stdin.write(raw_bytes)
                        self.ffmpeg_process.stdin.flush()
                    except (BrokenPipeError, OSError):
                        pass
                    except Exception as e:
                        print(f"Error writing to ffmpeg stdin: {e}")

    def get_frame(self):
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def get_frame_count(self):
        with self.frame_lock:
            return self.frame_count


class DahuaGPUCameraViewer:
    def __init__(self, codec='h264', gpuid=0, motion_detect=False,
                 motion_threshold=25, motion_alpha=0.05, min_motion_pixels=1000):
        self.loginID = C_LLONG()
        self.playID = C_LLONG()
        self.sdk = NetClient()
        self.gpuid = gpuid
        
        raw_c = (codec or 'auto').lower()
        if raw_c in ('hevc', 'h265', 'h.265'):
            self.configured_codec = 'h265'
        elif raw_c in ('h264', 'avc', 'h.264'):
            self.configured_codec = 'h264'
        else:
            self.configured_codec = 'auto'
        self.codec = self.configured_codec
        
        self.motion_detect = motion_detect
        self.motion_threshold = motion_threshold
        self.motion_alpha = motion_alpha
        self.min_motion_pixels = min_motion_pixels
        self.motion_detected = False
        self.detector = None
        
        self.m_DisConnectCallBack = fDisConnect(self.on_disconnect)
        self.m_ReConnectCallBack = fHaveReConnect(self.on_reconnect)
        self.m_RealDataCallBack = fRealDataCallBackEx2(self.on_frame_data)
        
        import queue
        self.packet_queue = queue.Queue(maxsize=150)
        self.frame_lock = threading.Lock()
        self.decode_lock = threading.Lock()
        self.latest_frame_bgr = None
        self.frame_count = 0
        self.fps = 0.0
        
        self.is_connected = False
        self.is_playing = False
        self.decoder = None
        self.waiting_for_keyframe = True
        
        self.last_frame_decoded_time = 0.0
        self.t_last_fps = 0.0
        self.frames_fps = 0
        
        self.sdk.InitEx(self.m_DisConnectCallBack)
        self.sdk.SetAutoReconnect(self.m_ReConnectCallBack)

    def connect(self, ip, port, username, password):
        print(f"[Dahua GPU] Connecting to {ip}:{port}...")
        if self.loginID:
            print("[Dahua GPU] Already connected!")
            return True
            
        stuInParam = NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY()
        stuInParam.dwSize = sizeof(NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY)
        stuInParam.szIP = ip.encode()
        stuInParam.nPort = int(port) if port else 37777
        stuInParam.szUserName = username.encode()
        stuInParam.szPassword = password.encode()
        stuInParam.emSpecCap = EM_LOGIN_SPAC_CAP_TYPE.TCP
        
        stuOutParam = NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY()
        stuOutParam.dwSize = sizeof(NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY)
        
        with sdk_login_lock:
            self.loginID, device_info, error_msg = self.sdk.LoginWithHighLevelSecurity(stuInParam, stuOutParam)
        if self.loginID != 0:
            self.device_info = device_info
            self.is_connected = True
            print(f"[Dahua GPU] Connected successfully! Channels: {device_info.nChanNum}")
            return True
        else:
            print(f"[Dahua GPU] Connection failed: {error_msg}")
            return False

    def start_stream(self, channel=0, stream_type=0):
        if not self.is_connected:
            print("[Dahua GPU] Not connected to camera!")
            return False
        if self.playID:
            print("[Dahua GPU] Stream already started!")
            return True

        # Tự động clamp channel về 0 nếu camera đơn kênh hoặc channel vượt quá nChanNum
        if hasattr(self, 'device_info') and self.device_info and self.device_info.nChanNum > 0:
            if channel >= self.device_info.nChanNum:
                print(f"[Dahua GPU] Channel {channel} exceeds device channel count {self.device_info.nChanNum}. Clamping channel to 0.")
                channel = 0
            
        print(f"[Dahua GPU] Starting stream on channel {channel}...")
        play_type = SDK_RealPlayType.Realplay if stream_type == 0 else SDK_RealPlayType.Realplay_1
        
        self.playID = self.sdk.RealPlayEx(self.loginID, channel, 0, play_type)
        if self.playID != 0:
            self.sdk.SetRealDataCallBackEx2(self.playID, self.m_RealDataCallBack, None, EM_REALDATA_FLAG.RAW_DATA)
            self.is_playing = True
            
            self.last_frame_decoded_time = time.time()
            self.t_last_fps = time.time()
            self.frames_fps = 0
            
            print("[Dahua GPU] Stream started successfully!")
            return True
        else:
            self.is_playing = False
            print(f"[Dahua GPU] Failed to start stream: {self.sdk.GetLastErrorMessage()}")
            return False

    def on_disconnect(self, lLoginID, pchDVRIP, nDVRPort, dwUser):
        print("[Dahua GPU] Camera disconnected!")
        self.is_connected = False

    def on_reconnect(self, lLoginID, pchDVRIP, nDVRPort, dwUser):
        print("[Dahua GPU] Camera reconnected!")
        self.is_connected = True

    def on_frame_data(self, lRealHandle, dwDataType, pBuffer, dwBufSize, param, dwUser):
        if lRealHandle == self.playID and dwDataType == 0:
            data = cast(pBuffer, POINTER(c_ubyte * dwBufSize)).contents
            raw_bytes = bytes(data)

            # Lọc bỏ hoàn toàn các gói không phải video (0xf0: audio, 0xf1: metadata/IVS)
            if (raw_bytes.startswith(b'DHAV') or raw_bytes.startswith(b'dhav')) and len(raw_bytes) > 4:
                if raw_bytes[4] not in (0xfc, 0xfd):
                    return

            try:
                self.packet_queue.put_nowait(raw_bytes)
            except queue.Full:
                try:
                    self.packet_queue.get_nowait()
                    self.packet_queue.put_nowait(raw_bytes)
                except Exception:
                    pass

    @torch.no_grad()
    def decode_next_packet(self):
        import queue
        import PyNvVideoCodec as nvc
        import torch
        import torch.nn.functional as F
        import cv2
        
        try:
            raw_bytes = self.packet_queue.get(timeout=0.01)
        except queue.Empty:
            return None, False
            
        if not raw_bytes or len(raw_bytes) < 5:
            return None, False
            
        start_search = 4 if (raw_bytes.startswith(b'DHAV') or raw_bytes.startswith(b'dhav')) else 0
        is_dhav_key = (len(raw_bytes) > 4 and raw_bytes[4] == 0xfd) if start_search == 4 else False
        idx = raw_bytes.find(b'\x00\x00\x00\x01', start_search)
        prefix_len = 4
        if idx == -1:
            idx = raw_bytes.find(b'\x00\x00\x01', start_search)
            prefix_len = 3
            
        # Lọc bỏ gói không có NAL start code (audio, metadata, heartbeat)
        if idx == -1 or idx + prefix_len >= len(raw_bytes):
            return None, False
            
        raw_bytes = raw_bytes[idx:]
        nal_byte = raw_bytes[prefix_len]
        h264_type = nal_byte & 0x1F
        h265_type = (nal_byte >> 1) & 0x3F
        
        # IDR keyframe thực sự mang đầy đủ dữ liệu khung hình Intra:
        # Trong H.264: IDR = 5 hoặc DHAV header type 0xfd
        # Trong H.265: IDR/IRAP (16..21) hoặc DHAV header type 0xfd
        is_idr_h265 = (16 <= h265_type <= 21) or is_dhav_key
        is_idr_h264 = (h264_type == 5) or is_dhav_key
        
        # Khởi tạo Decoder
        if self.decoder is None:
            if self.configured_codec == 'auto':
                # Chế độ AUTO: Tự động phát hiện codec từ NAL parameter sets
                detected = detect_nal_codec(raw_bytes)
                if detected is not None:
                    self.codec = detected
                    print(f"[Dahua GPU] Auto-detected stream codec: {self.codec.upper()}")
                else:
                    # Chưa thấy parameter set (VPS/SPS), đợi tiếp để xác định đúng chuẩn
                    return None, False
            else:
                # CMS chỉ định H264/H265: dùng ngay, nhưng nếu gói này mang parameter
                # set của codec KHÁC thì luồng thật mới đúng (CMS khai sai / camera
                # đổi codec) — tạo decoder sai codec là không bao giờ ra frame.
                # Chờ parameter set (đi kèm mỗi keyframe) tối đa 50 gói để đối chiếu;
                # không thấy thì mới tin hẳn CMS.
                detected = detect_nal_codec(raw_bytes)
                if detected is None:
                    self._codec_probe_packets = getattr(self, '_codec_probe_packets', 0) + 1
                    if self._codec_probe_packets < 50:
                        return None, False
                if detected is not None and detected != self.configured_codec:
                    print(f"[Dahua GPU] WARNING: CMS video_codec={self.configured_codec.upper()} "
                          f"nhưng luồng thực tế là {detected.upper()} — dùng {detected.upper()}")
                    self.codec = detected
                else:
                    self.codec = self.configured_codec
                self._codec_probe_packets = 0

            print("[Dahua GPU] Lazy init: Loading GPUMotionDetector...")
            from services.motion_detector import GPUMotionDetector
            print("[Dahua GPU] Lazy init: Setting CUDA device...")
            torch.cuda.set_device(self.gpuid)
            print("[Dahua GPU] Lazy init: CUDA device set.")
            print(f"[Dahua GPU] Lazy init: Creating PyNvVideoCodec Decoder ({self.codec.upper()})...")
            codec_id = nvc.cudaVideoCodec.HEVC if self.codec.upper() in ('HEVC', 'H265') else nvc.cudaVideoCodec.H264
            with decoding_lock:
                self.decoder = nvc.CreateDecoder(
                    gpuid=self.gpuid,
                    codec=codec_id,
                    usedevicememory=True
                )
            print("[Dahua GPU] Lazy init: Decoder created.")
            self.waiting_for_keyframe = True
            
            if self.motion_detect and self.detector is None:
                print("[Dahua GPU] Lazy init: Creating GPUMotionDetector instance...")
                self.detector = GPUMotionDetector(
                    width=640,
                    height=360,
                    threshold=self.motion_threshold,
                    alpha=self.motion_alpha,
                    min_motion_pixels=self.min_motion_pixels,
                    dtype=torch.float16
                )
                print("[Dahua GPU] Lazy init: GPUMotionDetector created.")

        # Cơ chế chờ Keyframe đầu tiên trước khi decode để triệt tiêu bóng ma / vỡ hình
        if self.waiting_for_keyframe:
            is_key = is_idr_h265 if self.codec in ('h265', 'hevc') else is_idr_h264
            if not is_key:
                # Bỏ qua P-frame này khi chưa có Keyframe tham chiếu
                return None, False
            self.waiting_for_keyframe = False
            print(f"[Dahua GPU] Keyframe acquired for {self.codec.upper()}, starting clean decode!")
        
        c_data = (c_ubyte * len(raw_bytes)).from_buffer_copy(raw_bytes)
        p_data = nvc.PacketData()
        p_data.bsl_data = cast(c_data, c_void_p).value
        p_data.bsl = len(raw_bytes)
        p_data.pts = 0
        p_data.dts = 0
        
        cpu_bgr = None
        motion_detected_flag = True
        
        try:
            with self.decode_lock:
                if self.decoder is None:
                    return None, False
                decoded_frames = self.decoder.Decode(p_data)
                if len(decoded_frames) > 0:
                    self.last_frame_decoded_time = time.time()
                    decoded_frame = decoded_frames[-1]
                    gpu_tensor_raw = torch.from_dlpack(decoded_frame)
                    
                    raw_height, raw_width = gpu_tensor_raw.shape[:2]
                    
                    if self.motion_detect and self.detector is not None:
                        stream_height = raw_height * 2 // 3
                        y_channel = gpu_tensor_raw[:stream_height, :]
                        
                        y_channel_4d = y_channel.float().unsqueeze(0).unsqueeze(0)
                        resized_gpu = F.interpolate(
                            y_channel_4d,
                            size=(360, 640),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze()
                        
                        motion_detected_flag, _ = self.detector.detect(resized_gpu)
                    
                    if motion_detected_flag:
                        gpu_bgr = nv12_to_bgr_torch(gpu_tensor_raw, raw_height * 2 // 3, raw_width)
                        cpu_bgr = gpu_bgr.cpu().numpy()
                    else:
                        cpu_bgr = None
                    
                    del gpu_tensor_raw
                    del decoded_frame
                del decoded_frames
            
            with self.frame_lock:
                self.motion_detected = motion_detected_flag
                if cpu_bgr is not None:
                    self.latest_frame_bgr = cpu_bgr
                    self.frame_count += 1
            
            if motion_detected_flag and cpu_bgr is not None:
                self.frames_fps += 1
                t_now = time.time()
                if t_now - self.t_last_fps >= 2.0:
                    self.fps = self.frames_fps / (t_now - self.t_last_fps)
                    self.frames_fps = 0
                    self.t_last_fps = t_now
            
            if self.frame_count > 0 and time.time() - self.last_frame_decoded_time > 10.0:
                raise RuntimeError("No frames decoded for 10 seconds (decoder out-of-sync)")
                
            if cpu_bgr is not None:
                # cpu_bgr vừa được tạo ở trên và chỉ caller giữ reference; không copy toàn frame lần hai.
                return cpu_bgr, motion_detected_flag
            else:
                return None, motion_detected_flag
                
        except Exception as dec_err:
            print(f"[Dahua GPU] GPU Decode error: {dec_err}. Re-creating decoder for self-recovery...")
            try:
                while not self.packet_queue.empty():
                    try:
                        self.packet_queue.get_nowait()
                    except Exception:
                        pass
                        
                with decoding_lock:
                    self.decoder = None
                    codec_id = nvc.cudaVideoCodec.HEVC if self.codec.upper() in ('HEVC', 'H265') else nvc.cudaVideoCodec.H264
                    self.decoder = nvc.CreateDecoder(
                        gpuid=self.gpuid,
                        codec=codec_id,
                        usedevicememory=True
                    )
                if self.detector is not None:
                    self.detector.reset()
                self.waiting_for_keyframe = True
                self.last_frame_decoded_time = time.time()
                print("[Dahua GPU] Decoder successfully re-created and packet queue cleared (waiting for Keyframe).")
            except Exception as rec_err:
                print(f"[Dahua GPU] Failed to re-create decoder: {rec_err}")
            return None, False

    def get_frame(self):
        with self.frame_lock:
            return self.latest_frame_bgr.copy() if self.latest_frame_bgr is not None else None

    def get_frame_with_motion(self):
        with self.frame_lock:
            frame = self.latest_frame_bgr.copy() if self.latest_frame_bgr is not None else None
            return frame, self.motion_detected

    def stop_stream(self):
        self.is_playing = False
        if self.playID:
            self.sdk.StopRealPlayEx(self.playID)
            self.playID = 0
        with decoding_lock:
            self.decoder = None
        self.detector = None
        print("[Dahua GPU] Stream stopped")

    def disconnect(self):
        self.stop_stream()
        if self.loginID:
            self.sdk.Logout(self.loginID)
            self.loginID = 0
            self.is_connected = False
            print("[Dahua GPU] Disconnected")

    def cleanup(self):
        self.disconnect()
        self.sdk.Cleanup()
        print("[Dahua GPU] Cleanup completed")