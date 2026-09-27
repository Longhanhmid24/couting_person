# coding=utf-8

import ctypes
import sys
from ctypes import *

# Xác định kiểu hàm callback theo OS
if 'linux' in sys.platform:
    fun_ctype = CFUNCTYPE
else:
    fun_ctype = WINFUNCTYPE

# --- ĐỊNH NGHĨA KIỂU DỮ LIỆU AN TOÀN (CROSS-PLATFORM) ---
# Hikvision SDK dùng LONG (32bit) và DWORD (32bit)
# Trên Linux 64bit, c_long là 64bit -> gây lỗi. Phải dùng c_int (32bit).
HIK_LONG = c_int
HIK_DWORD = c_uint

# --------------------------------------------------------

# Code Stream Callback Data Type
NET_DVR_SYSHEAD = 1
NET_DVR_STREAMDATA = 2
NET_DVR_AUDIOSTREAMDATA = 3
NET_DVR_PRIVATE_DATA = 112

# Device parameter structure V30
class NET_DVR_DEVICEINFO_V30(ctypes.Structure):
    _pack_ = 1  # QUAN TRỌNG: Căn chỉnh 1 byte
    _fields_ = [
        ("sSerialNumber", c_byte * 48),
        ("byAlarmInPortNum", c_byte),
        ("byAlarmOutPortNum", c_byte),
        ("byDiskNum", c_byte),
        ("byDVRType", c_byte),
        ("byChanNum", c_byte),
        ("byStartChan", c_byte),
        ("byAudioChanNum", c_byte),
        ("byIPChanNum", c_byte),
        ("byZeroChanNum", c_byte),
        ("byMainProto", c_byte),
        ("bySubProto", c_byte),
        ("bySupport", c_byte),
        ("bySupport1", c_byte),
        ("bySupport2", c_byte),
        ("wDevType", c_uint16),
        ("bySupport3", c_byte),
        ("byMultiStreamProto", c_byte),
        ("byStartDChan", c_byte),
        ("byStartDTalkChan", c_byte),
        ("byHighDChanNum", c_byte),
        ("bySupport4", c_byte),
        ("byLanguageType", c_byte),
        ("byVoiceInChanNum", c_byte),
        ("byStartVoiceInChanNo", c_byte),
        ("bySupport5", c_byte ),
        ("bySupport6", c_byte),
        ("byMirrorChanNum", c_byte),
        ("wStartMirrorChanNo", c_uint16),
        ("bySupport7", c_byte),
        ("byRes2", c_byte)]
LPNET_DVR_DEVICEINFO_V30 = POINTER(NET_DVR_DEVICEINFO_V30)

# Device parameter structure V40
class NET_DVR_DEVICEINFO_V40(ctypes.Structure):
    _pack_ = 1 # QUAN TRỌNG
    _fields_ = [
        ('struDeviceV30', NET_DVR_DEVICEINFO_V30),
        ('bySupportLock', c_byte),
        ('byRetryLoginTime', c_byte),
        ('byPasswordLevel', c_byte),
        ('byProxyType', c_byte),
        ('dwSurplusLockTime', c_uint32),
        ('byCharEncodeType', c_byte),
        ('bySupportDev5', c_byte),
        ('bySupport', c_byte),
        ('byLoginMode', c_byte),
        ('dwOEMCode', c_uint32),
        ('iResidualValidity', c_uint32),
        ('byResidualValidity', c_byte),
        ('bySingleStartDTalkChan', c_byte),
        ('bySingleDTalkChanNums', c_byte),
        ('byPassWordResetLevel', c_byte),
        ('bySupportStreamEncrypt', c_byte),
        ('byMarketType', c_byte),
        ('byRes2', c_byte * 238)
    ]
LPNET_DVR_DEVICEINFO_V40 = POINTER(NET_DVR_DEVICEINFO_V40)

# Asynchronous login callback function
fLoginResultCallBack = fun_ctype(None, HIK_LONG, HIK_DWORD, LPNET_DVR_DEVICEINFO_V30, c_void_p)

# NET_DVR_Login_V40() parameter
class NET_DVR_USER_LOGIN_INFO(Structure):
    _pack_ = 1  # QUAN TRỌNG: Sửa lỗi padding struct trên Linux 64-bit
    # Giải thích: Nếu không có _pack_=1, ctypes sẽ tự động thêm 4 bytes padding
    # sau sPassword để con trỏ cbLoginResult nằm ở địa chỉ chia hết cho 8 (trên 64bit).
    # Điều này làm lệch vị trí của wPort và bUseAsynLogin.
    
    _fields_ = [
        ("sDeviceAddress", c_char * 129),
        ("byUseTransport", c_byte),
        ("wPort", c_uint16),
        ("sUserName", c_char * 64),
        ("sPassword", c_char * 64),
        ("cbLoginResult", fLoginResultCallBack),
        ("pUser", c_void_p),
        ("bUseAsynLogin", HIK_DWORD), # BOOL trong C SDK là int (4 byte)
        ("byProxyType", c_byte),
        ("byUseUTCTime", c_byte),
        ("byLoginMode", c_byte),
        ("byHttps", c_byte),
        ("iProxyID", HIK_LONG),
        ("byVerifyMode", c_byte),
        ("byRes2", c_byte * 119)]
LPNET_DVR_USER_LOGIN_INFO = POINTER(NET_DVR_USER_LOGIN_INFO)

# Component library loading path information
class NET_DVR_LOCAL_SDK_PATH(Structure):
    _fields_ = [
        ('sPath', c_char * 256),
        ('byRes', c_byte * 128),
    ]
LPNET_DVR_LOCAL_SDK_PATH = POINTER(NET_DVR_LOCAL_SDK_PATH)

# Define the preview parameter structure
class NET_DVR_PREVIEWINFO(Structure):
    _pack_ = 1
    _fields_ = [
        ('lChannel', HIK_LONG),
        ('dwStreamType', HIK_DWORD),
        ('dwLinkMode', HIK_DWORD),
        ('hPlayWnd', c_void_p), # HWND
        ('bBlocked', HIK_DWORD),
        ('bPassbackRecord', HIK_DWORD),
        ('byPreviewMode', c_ubyte),
        ('byStreamID', c_ubyte * 32),
        ('byProtoType', c_ubyte),
        ('byRes1', c_ubyte),
        ('byVideoCodingType', c_ubyte),
        ('dwDisplayBufNum', HIK_DWORD),
        ('byNPQMode', c_ubyte),
        ('byRecvMetaData', c_ubyte),
        ('byDataType', c_ubyte),
        ('byRes', c_ubyte * 213),
    ]
LPNET_DVR_PREVIEWINFO = POINTER(NET_DVR_PREVIEWINFO)

# --- SỬA CÁC CALLBACK QUAN TRỌNG ---

# Alarm information callback function
# Thay c_ulong bằng HIK_DWORD vì trên Linux c_ulong là 8 byte (sai), SDK cần 4 byte
MSGCallBack_V31 = fun_ctype(c_bool, HIK_DWORD, c_void_p, c_void_p, HIK_DWORD, c_void_p)
MSGCallBack = fun_ctype(None, HIK_DWORD, c_void_p, c_void_p, HIK_DWORD, c_void_p)

# Codeflow callback function
# QUAN TRỌNG: Thay c_long/c_ulong bằng HIK_LONG/HIK_DWORD
# Nếu để c_long (8 bytes) trên Linux, callback sẽ đọc lố bộ nhớ và crash stream
REALDATACALLBACK = fun_ctype(None, HIK_LONG, HIK_DWORD, POINTER(c_ubyte), HIK_DWORD, c_void_p)