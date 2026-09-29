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
    engine.main()
