"""Lightweight ByteTrack components for people counting."""
from .byte_tracker import BYTETracker, TrackResult
from .track_stitcher import TrackStitcher

__all__ = ["BYTETracker", "TrackResult", "TrackStitcher"]
