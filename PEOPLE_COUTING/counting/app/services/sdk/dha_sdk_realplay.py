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


class DahuaCameraViewer:
    def __init__(self, codec='h264'):
        self.loginID = C_LLONG()
        self.playID = C_LLONG()
        self.sdk = NetClient()
        self.codec = codec.lower() if codec else 'h264'
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
        self.loginID, device_info, error_msg = self.sdk.LoginWithHighLevelSecurity(stuInParam, stuOutParam)
        if self.loginID != 0:
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
        print(f"Starting stream on channel {channel}...")
        play_type = SDK_RealPlayType.Realplay if stream_type == 0 else SDK_RealPlayType.Realplay_1
        self.playID = self.sdk.RealPlayEx(self.loginID, channel, 0, play_type)
        if self.playID != 0:
            self.sdk.SetRealDataCallBackEx2(self.playID, self.m_RealDataCallBack, None, EM_REALDATA_FLAG.RAW_DATA)
            self.is_playing = True
            print("Stream started successfully!")
            self.setup_decoder(codec_hint=self.codec)
            return True
        else:
            print(f"Failed to start stream: {self.sdk.GetLastErrorMessage()}")
            return False

    def setup_decoder(self, codec_hint='h264'):
        try:
            cmd = [
                'ffmpeg', '-hwaccel', 'none', '-fflags', 'nobuffer', '-flags', 'low_delay',
                '-probesize', '100000', '-analyzeduration', '100000',
                '-i', 'pipe:0', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                '-an', '-loglevel', 'info', 'pipe:1'
            ]
            self.current_codec = codec_hint
            print(f"Setting up auto-detect FFmpeg decoder for stream...")
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
        if not self.resolution_event.wait(timeout=10):
            print("❌ Timed out waiting for resolution. Decoder thread will exit.")
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
        self.codec = codec.lower()
        
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
        self.packet_queue = queue.Queue(maxsize=10)
        self.frame_lock = threading.Lock()
        self.latest_frame_bgr = None
        self.frame_count = 0
        self.fps = 0.0
        
        self.is_connected = False
        self.is_playing = False
        self.decoder = None
        
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
        
        self.loginID, device_info, error_msg = self.sdk.LoginWithHighLevelSecurity(stuInParam, stuOutParam)
        if self.loginID != 0:
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
        
        if self.decoder is None:
            print("[Dahua GPU] Lazy init: Loading GPUMotionDetector...")
            from services.motion_detector import GPUMotionDetector
            print("[Dahua GPU] Lazy init: Setting CUDA device...")
            torch.cuda.set_device(self.gpuid)
            print("[Dahua GPU] Lazy init: CUDA device set.")
            print(f"[Dahua GPU] Lazy init: Creating PyNvVideoCodec Decoder ({self.codec})...")
            codec_id = nvc.cudaVideoCodec.HEVC if self.codec in ('hevc', 'h265') else nvc.cudaVideoCodec.H264
            with decoding_lock:
                self.decoder = nvc.CreateDecoder(
                    gpuid=self.gpuid,
                    codec=codec_id,
                    usedevicememory=True
                )
            print("[Dahua GPU] Lazy init: Decoder created.")
            
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
        
        try:
            raw_bytes = self.packet_queue.get(timeout=0.01)
        except queue.Empty:
            return None, False
            
        if len(raw_bytes) == 0:
            return None, False
            
        if raw_bytes.startswith(b'DHAV') or raw_bytes.startswith(b'dhav'):
            idx = raw_bytes.find(b'\x00\x00\x00\x01', 4)
            if idx != -1:
                raw_bytes = raw_bytes[idx:]
            else:
                idx = raw_bytes.find(b'\x00\x00\x01', 4)
                if idx != -1:
                    raw_bytes = raw_bytes[idx:]
                elif len(raw_bytes) > 32:
                    raw_bytes = raw_bytes[32:]
                    
        if len(raw_bytes) == 0:
            return None, False
            
        c_data = (c_ubyte * len(raw_bytes)).from_buffer_copy(raw_bytes)
        p_data = nvc.PacketData()
        p_data.bsl_data = cast(c_data, c_void_p).value
        p_data.bsl = len(raw_bytes)
        p_data.pts = 0
        p_data.dts = 0
        
        cpu_bgr = None
        motion_detected_flag = True
        
        try:
            with decoding_lock:
                if self.decoder is None:
                    return None, False
                decoded_frames = self.decoder.Decode(p_data)
                if len(decoded_frames) > 0:
                    self.last_frame_decoded_time = time.time()
                    decoded_frame = decoded_frames[-1]
                    gpu_tensor_raw = torch.from_dlpack(decoded_frame)
                    
                    if self.motion_detect and self.detector is not None:
                        raw_height, raw_width = gpu_tensor_raw.shape[:2]
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
                    codec_id = nvc.cudaVideoCodec.HEVC if self.codec in ('hevc', 'h265') else nvc.cudaVideoCodec.H264
                    self.decoder = nvc.CreateDecoder(
                        gpuid=self.gpuid,
                        codec=codec_id,
                        usedevicememory=True
                    )
                if self.detector is not None:
                    self.detector.reset()
                self.last_frame_decoded_time = time.time()
                print("[Dahua GPU] Decoder successfully re-created and packet queue cleared.")
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