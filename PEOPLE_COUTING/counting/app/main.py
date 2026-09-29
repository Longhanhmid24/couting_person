# -*- coding: utf-8 -*-
"""
main.py — Entrypoint for Vehicle Counting Application.
Invokes CentralEngine to manage all streams, GPU decoding, and central counting pipeline.
"""
import os
import sys
import engine
from core.settings import settings

if __name__ == "__main__":
    if getattr(settings, 'ENABLE_LIVE_STREAM', False):
        try:
            from utils.live_streamer import start_streamer
            start_streamer(port=8899)
        except Exception as e:
            print(f"Failed to start live streamer: {e}")
    engine.main()
