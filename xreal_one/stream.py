"""Background-thread wrapper around the XREAL pose pipeline.

Owns a TCP connection, framer, and head tracker on a worker thread.
Consumers call `latest_pose()` from any thread to get a snapshot of
the current pose without blocking.
"""

from __future__ import annotations

import re
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from .protocol import (
    DEFAULT_HOST,
    DEFAULT_STREAM_PORT,
    ImuReport,
    StreamFramer,
)
from .tracker import HeadTracker, Pose


_IFCONFIG_INET_RE = re.compile(r"inet\s+(169\.254\.\d+\.\d+)")


def _pick_source_for(host: str) -> Optional[str]:
    """Return a local 169.254.x.y address whose /24 matches `host`, else None."""
    parts = host.split(".")
    if len(parts) != 4:
        return None
    prefix = ".".join(parts[:3]) + "."
    try:
        out = subprocess.run(
            ["ifconfig"], capture_output=True, text=True, timeout=2
        ).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    for ip in _IFCONFIG_INET_RE.findall(out):
        if ip.startswith(prefix):
            return ip
    return None


@dataclass(frozen=True)
class PoseSnapshot:
    absolute: Pose
    relative: Pose
    calibration_progress: float  # 0.0 .. 1.0
    is_calibrated: bool
    connected: bool
    last_error: Optional[str]


class PoseStream:
    """Threaded pose source. Reconnects automatically on disconnect."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_STREAM_PORT,
        reconnect_delay: float = 1.0,
    ) -> None:
        self.host = host
        self.port = port
        self.reconnect_delay = reconnect_delay

        self._lock = threading.Lock()
        self._abs = Pose(0.0, 0.0, 0.0)
        self._rel = Pose(0.0, 0.0, 0.0)
        self._calib_progress = 0.0
        self._is_calibrated = False
        self._connected = False
        self._last_error: Optional[str] = None
        self._zero_pending = False
        self._recalib_pending = False

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, name="xreal-pose", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self._sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.5)

    def latest(self) -> PoseSnapshot:
        with self._lock:
            return PoseSnapshot(
                absolute=self._abs,
                relative=self._rel,
                calibration_progress=self._calib_progress,
                is_calibrated=self._is_calibrated,
                connected=self._connected,
                last_error=self._last_error,
            )

    def zero_view(self) -> None:
        with self._lock:
            self._zero_pending = True

    def recalibrate(self) -> None:
        with self._lock:
            self._recalib_pending = True

    def _run_loop(self) -> None:
        framer = StreamFramer()
        tracker = HeadTracker()
        src = _pick_source_for(self.host)

        while self._running:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.settimeout(2.0)
                if src is not None:
                    try:
                        sock.bind((src, 0))
                    except OSError:
                        pass
                sock.connect((self.host, self.port))
                sock.settimeout(0.5)
                self._sock = sock
                with self._lock:
                    self._connected = True
                    self._last_error = None

                while self._running:
                    try:
                        chunk = sock.recv(4096)
                    except socket.timeout:
                        continue
                    if not chunk:
                        break
                    for report in framer.append(chunk):
                        if isinstance(report, ImuReport):
                            with self._lock:
                                if self._zero_pending and tracker.is_calibrated:
                                    tracker.zero_view()
                                    self._zero_pending = False
                                if self._recalib_pending:
                                    tracker.reset_calibration()
                                    self._recalib_pending = False
                            tracker.feed(report)
                            with self._lock:
                                self._abs = tracker.absolute()
                                self._rel = tracker.relative()
                                self._calib_progress = tracker.calibration_progress
                                self._is_calibrated = tracker.is_calibrated

            except OSError as e:
                with self._lock:
                    self._last_error = f"{e.__class__.__name__}: {e}"
            finally:
                with self._lock:
                    self._connected = False
                try:
                    sock.close()
                except OSError:
                    pass
                self._sock = None

            if self._running:
                time.sleep(self.reconnect_delay)
