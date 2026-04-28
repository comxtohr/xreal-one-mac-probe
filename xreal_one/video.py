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

        self._paused: bool = False
        self._speed: float = 1.0
        self._timeline_dirty: bool = False  # set when pause/speed change so we re-anchor
        self._seek_target: Optional[float] = None  # seconds; consumed by _play_once
        self._seek_audio_compensation: float = 0.0  # bias _start_wall on seek

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

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def latest_pts(self) -> float:
        with self._lock:
            return self._latest_pts

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            if self._paused != paused:
                self._paused = paused
                self._timeline_dirty = True

    def toggle_paused(self) -> bool:
        with self._lock:
            self._paused = not self._paused
            self._timeline_dirty = True
            return self._paused

    def set_speed(self, speed: float) -> None:
        speed = max(0.1, min(8.0, speed))
        with self._lock:
            if abs(self._speed - speed) > 1e-6:
                self._speed = speed
                self._timeline_dirty = True

    def request_seek(
        self,
        pts_seconds: float,
        audio_compensation: float = 0.0,
    ) -> None:
        """Schedule a seek to `pts_seconds`. The decoder thread breaks out
        of its current decode pass and re-opens the container at the new
        offset on the next iteration.

        `audio_compensation` is the number of seconds we expect the audio
        backend to take to reflect the seek (mpv: ~0, ffplay: ~0.15). The
        wall-clock anchor is biased by this much so video and audio land
        at target_pts at roughly the same wall instant."""
        if self._duration > 0:
            pts_seconds = max(0.0, min(pts_seconds, self._duration - 0.05))
        else:
            pts_seconds = max(0.0, pts_seconds)
        with self._lock:
            self._seek_target = pts_seconds
            self._seek_audio_compensation = max(0.0, audio_compensation)

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
        with self._lock:
            start_offset = self._seek_target if self._seek_target is not None else 0.0
            audio_comp = self._seek_audio_compensation
            self._seek_target = None
            self._seek_audio_compensation = 0.0

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

            if start_offset > 0.0 and time_base > 0.0:
                try:
                    container.seek(
                        int(start_offset / time_base),
                        any_frame=False,
                        backward=True,
                        stream=stream,
                    )
                except Exception:
                    pass

            # Bias the wall-clock anchor by the audio backend's reported
            # seek latency (mpv: tens of ms, ffplay: ~150 ms). 0 on the
            # natural file-restart loop (EOF) since audio doesn't re-spawn
            # there. Only applied on user-initiated seeks.
            self._start_wall = (
                time.monotonic()
                - start_offset / max(0.01, self._speed)
                + audio_comp
            )

            # PyAV's container.seek with backward=True lands on the closest
            # keyframe BEFORE the target. The frames between keyframe and
            # target are needed for decode-chain consistency but should not
            # be displayed (they're old content). Skip them silently until
            # we reach the requested pts.
            skip_until_pts = start_offset if start_offset > 0.0 else 0.0

            for frame in container.decode(stream):
                if not self._running:
                    return

                # Bail if a seek was requested - _decode_loop will call us
                # again, picking up the new offset from _seek_target.
                with self._lock:
                    if self._seek_target is not None:
                        return

                if frame.pts is not None and time_base > 0:
                    pts = float(frame.pts) * time_base
                else:
                    pts = self._latest_pts + (1.0 / max(1.0, self._fps))

                # Drop pre-target frames after a seek. They're decoded
                # silently to keep the codec state valid; nothing visible.
                if skip_until_pts > 0.0 and pts < skip_until_pts - 0.05:
                    continue
                skip_until_pts = 0.0  # caught up; resume normal pacing

                rgb = frame.reformat(width=tw, height=th, format="rgb24")
                arr = rgb.to_ndarray()  # shape (th, tw, 3) uint8

                # Publish this frame BEFORE the pause-hold so seeking while
                # paused immediately swaps in the target frame instead of
                # leaving the old one on screen.
                with self._lock:
                    self._latest_frame = arr
                    self._latest_pts = pts
                    self._frame_count += 1

                # Pause loop runs after publish: keeps the just-shown frame
                # frozen on screen and waits for either resume or a seek.
                while self._running and self._paused:
                    with self._lock:
                        if self._seek_target is not None:
                            return
                    time.sleep(0.05)
                if not self._running:
                    return

                # Read timeline state after pause exit so re-anchor uses
                # the current pts and the wall clock at the moment of
                # resume - smooth restart without "catch-up" jumps.
                with self._lock:
                    timeline_dirty = self._timeline_dirty
                    self._timeline_dirty = False
                    speed = max(0.01, self._speed)

                if timeline_dirty:
                    self._start_wall = time.monotonic() - pts / speed

                target_wall = self._start_wall + pts / speed
                now = time.monotonic()
                if target_wall > now:
                    time.sleep(min(target_wall - now, 0.1))
        finally:
            container.close()
