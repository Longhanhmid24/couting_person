# coding=utf-8

from ctypes import *
import sys

# Callback function type definition
if 'linux' in sys.platform:
    fun_ctype = CFUNCTYPE
else:
    fun_ctype = WINFUNCTYPE

# Define the preview parameter structure
class FRAME_INFO(Structure):
    pass
LPFRAME_INFO = POINTER(FRAME_INFO)
FRAME_INFO._fields_ = [
    ('nWidth', c_uint32),
    ('nHeight', c_uint32),
    ('nStamp', c_uint32),
    ('nType', c_uint32),
    ('nFrameRate', c_uint32),
    ('dwFrameNum', c_uint32)
]

# Show callback function
DISPLAYCBFUN = fun_ctype(None, c_long, c_char_p, c_long, c_long, c_long, c_long, c_long, c_long)
# Decode the callback function
DECCBFUNWIN = fun_ctype(None, c_long, POINTER(c_char), c_long, POINTER(FRAME_INFO), c_void_p, c_void_p)
