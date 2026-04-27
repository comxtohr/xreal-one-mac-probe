"""Threaded video decoder backed by PyAV.

Reads a video file, decodes (multi-threaded software decode by default),
optionally downscales each frame to a target size, and converts to RGB24
numpy arrays for GL upload. Paces playback by PTS so the renderer's
"latest frame" approximates real-time.

`latest()` always returns the most recent decoded frame (possibly from
several real-time intervals back if the renderer is slow). The decoder
loops the file on EOF.
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import av  # PyAV
import numpy as np


class VideoStream:
    def __init__(
        self,
        path: str,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        loop: bool = True,
    ) -> None:
        self.path = path
        self.target_width = target_width
        self.target_height = target_height
        self.loop = loop

        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_pts: float = 0.0
        self._frame_count: int = 0
        self._error: Optional[str] = None
        self._eof: bool = False

        self._source_w: int = 0
        self._source_h: int = 0
        self._fps: float = 0.0
        self._duration: float = 0.0

        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._start_wall: Optional[float] = None

    @property
    def source_size(self) -> Tuple[int, int]:
        return self._source_w, self._source_h

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def duration_seconds(self) -> float:
        return self._duration

    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._frame_count

    @property
    def last_error(self) -> Optional[str]:
        return self._error

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._decode_loop, name="xreal-video", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def latest(self) -> Optional[Tuple[np.ndarray, float]]:
        """Returns (frame_rgb24, pts_seconds), or None until first frame."""
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame, self._latest_pts

    def _decode_loop(self) -> None:
        while self._running:
            try:
                self._play_once()
            except Exception as e:  # broad on purpose: surface to UI status
                self._error = f"{e.__class__.__name__}: {e}"
                time.sleep(1.0)
            if not self.loop:
                break
        with self._lock:
            self._eof = True

    def _play_once(self) -> None:
        container = av.open(self.path)
        try:
            stream = container.streams.video[0]
            # Multi-threaded software decode (Apple Silicon eats 8K HEVC at
            # ~30 fps with all P-cores). Avoids the platform-specific
            # gymnastics of wiring up VideoToolbox HW accel through PyAV.
            stream.thread_type = "AUTO"
            try:
                stream.codec_context.thread_count = 0
            except Exception:
                pass

            self._source_w = stream.codec_context.width
            self._source_h = stream.codec_context.height
            try:
                self._fps = float(stream.average_rate) if stream.average_rate else 0.0
            except Exception:
                self._fps = 0.0
            try:
                self._duration = float(stream.duration * stream.time_base) \
                    if stream.duration else 0.0
            except Exception:
                self._duration = 0.0

            tw = self.target_width or self._source_w
            th = self.target_height or self._source_h

            time_base = float(stream.time_base) if stream.time_base else 0.0
            self._start_wall = time.monotonic()

            for frame in container.decode(stream):
                if not self._running:
                    return
                rgb = frame.reformat(width=tw, height=th, format="rgb24")
                arr = rgb.to_ndarray()  # shape (th, tw, 3) uint8

                if frame.pts is not None and time_base > 0:
                    pts = float(frame.pts) * time_base
                else:
                    pts = self._latest_pts + (1.0 / max(1.0, self._fps))

                with self._lock:
                    self._latest_frame = arr
                    self._latest_pts = pts
                    self._frame_count += 1

                # Pace to PTS. If we're behind we don't sleep — better to push
                # the latest frame and let the renderer drop intermediate ones.
                target_wall = self._start_wall + pts
                now = time.monotonic()
                if target_wall > now:
                    time.sleep(min(target_wall - now, 0.1))
        finally:
            container.close()
