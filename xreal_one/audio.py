"""Audio playback via mpv (preferred) or ffplay (fallback).

The mpv backend keeps a single mpv process running from start() to
stop(), driving pause / seek / speed by sending JSON commands over
mpv's --input-ipc-server Unix socket. There's no spawn between state
changes, so seek latency is on the order of one frame and pause is
effectively instantaneous.

The ffplay backend kills + respawns the process on every state change.
That works but adds ~150 ms of spawn delay; it's kept as a fallback
for systems without mpv installed.

afplay (macOS built-in) is the last-resort fallback. It has no IPC and
no -ss; we just play once from the beginning. set_state() is a no-op.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Optional


class AudioPlayer:
    def __init__(self, path: str, loop: bool = True) -> None:
        self.path = path
        self.loop = loop
        self._backend: Optional[str] = None

        # mpv state
        self._mpv_proc: Optional[subprocess.Popen] = None
        self._mpv_socket: Optional[socket.socket] = None
        self._mpv_socket_path: Optional[str] = None

        # ffplay/afplay state — last-applied desired state
        self._ff_proc: Optional[subprocess.Popen] = None
        self._ff_active: bool = False
        self._ff_speed: float = 1.0
        self._ff_offset: float = 0.0

    @property
    def backend(self) -> Optional[str]:
        return self._backend

    @property
    def seek_latency_seconds(self) -> float:
        """Approximate wall-clock time between a set_state() call and the
        audio actually reflecting it. Used by VideoStream to bias its
        wall-clock anchor on seeks so video and audio land on target_pts
        at roughly the same wall instant."""
        if self._backend == "mpv":
            return 0.02
        if self._backend == "ffplay":
            return 0.15
        return 0.0

    # --------------- start / stop ---------------

    def start(self) -> None:
        if shutil.which("mpv"):
            try:
                self._start_mpv()
                self._backend = "mpv"
            except Exception as e:
                print(f"audio: mpv launch failed ({e}); falling back to ffplay")
                self._cleanup_mpv()
                if shutil.which("ffplay"):
                    self._backend = "ffplay"
                    self._start_ffplay_initial()
        elif shutil.which("ffplay"):
            self._backend = "ffplay"
            self._start_ffplay_initial()
        elif shutil.which("afplay"):
            self._backend = "afplay"
            self._start_afplay_initial()
        else:
            print("audio: no backend found. install mpv (preferred) or ffmpeg.")
            return
        atexit.register(self.stop)

    def stop(self) -> None:
        self._cleanup_mpv()
        if self._ff_proc is not None:
            self._kill_proc_pg(self._ff_proc)
            self._ff_proc = None

    # --------------- public state setter ---------------

    def set_state(self, active: bool, speed: float, offset_sec: float) -> None:
        if self._backend == "mpv":
            self._set_state_mpv(active, speed, offset_sec)
        elif self._backend == "ffplay":
            self._set_state_ffplay(active, speed, offset_sec)
        # afplay has no IPC and no seek; ignore.

    # --------------- mpv backend ---------------

    def _start_mpv(self) -> None:
        sock_dir = Path(os.environ.get("TMPDIR", "/tmp"))
        sock_dir.mkdir(parents=True, exist_ok=True)
        self._mpv_socket_path = str(sock_dir / f"xreal-mpv-{os.getpid()}.sock")
        try:
            os.unlink(self._mpv_socket_path)
        except FileNotFoundError:
            pass

        args = [
            "mpv",
            "--no-video",
            "--idle=yes",
            "--keep-open=yes",
            "--audio-display=no",
            "--really-quiet",
            "--load-scripts=no",
            f"--input-ipc-server={self._mpv_socket_path}",
        ]
        if self.loop:
            args.append("--loop-file=inf")

        self._mpv_proc = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )

        # Wait for the IPC socket to be created.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if os.path.exists(self._mpv_socket_path):
                break
            time.sleep(0.02)
        else:
            raise RuntimeError("mpv IPC socket did not appear within 2 s")

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        sock.connect(self._mpv_socket_path)
        self._mpv_socket = sock

        # Load the file and start playing at 1x from the beginning.
        self._mpv_send({"command": ["loadfile", self.path]})
        self._mpv_send({"command": ["set_property", "pause", False]})
        self._mpv_send({"command": ["set_property", "speed", 1.0]})

    def _mpv_send(self, command: dict) -> None:
        if self._mpv_socket is None:
            return
        try:
            self._mpv_socket.sendall((json.dumps(command) + "\n").encode("utf-8"))
        except (OSError, socket.timeout):
            pass

    def _set_state_mpv(self, active: bool, speed: float, offset_sec: float) -> None:
        speed = max(0.1, min(8.0, speed))
        offset_sec = max(0.0, offset_sec)
        # Send all three; mpv applies them in order. Property changes are
        # near-instant; "exact" seek is sub-frame on most files.
        self._mpv_send({"command": ["set_property", "pause", not active]})
        self._mpv_send({"command": ["set_property", "speed", float(speed)]})
        self._mpv_send({"command": ["seek", float(offset_sec), "absolute", "exact"]})

    def _cleanup_mpv(self) -> None:
        if self._mpv_socket is not None:
            try:
                self._mpv_send({"command": ["quit"]})
            except OSError:
                pass
            try:
                self._mpv_socket.close()
            except OSError:
                pass
            self._mpv_socket = None
        if self._mpv_proc is not None:
            self._kill_proc_pg(self._mpv_proc)
            self._mpv_proc = None
        if self._mpv_socket_path is not None:
            try:
                os.unlink(self._mpv_socket_path)
            except OSError:
                pass
            self._mpv_socket_path = None

    # --------------- ffplay backend (kill+respawn) ---------------

    def _start_ffplay_initial(self) -> None:
        self._ff_active = True
        self._ff_speed = 1.0
        self._ff_offset = 0.0
        self._ff_proc = self._spawn_ffplay()

    def _start_afplay_initial(self) -> None:
        self._ff_active = True
        self._ff_proc = subprocess.Popen(
            ["afplay", self.path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )

    def _set_state_ffplay(self, active: bool, speed: float, offset_sec: float) -> None:
        speed = max(0.1, min(8.0, speed))
        offset_sec = max(0.0, offset_sec)
        proc_alive = self._ff_proc is not None and self._ff_proc.poll() is None
        nothing_changed = (
            active == self._ff_active
            and abs(speed - self._ff_speed) < 1e-3
            and abs(offset_sec - self._ff_offset) < 0.05
            and (proc_alive or not active)
        )
        if nothing_changed:
            return
        self._ff_active = active
        self._ff_speed = speed
        self._ff_offset = offset_sec
        if self._ff_proc is not None:
            self._kill_proc_pg(self._ff_proc)
            self._ff_proc = None
        if active:
            self._ff_proc = self._spawn_ffplay()

    def _spawn_ffplay(self) -> Optional[subprocess.Popen]:
        args = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error"]
        if self.loop:
            args += ["-loop", "0"]
        if self._ff_offset > 0.0:
            args += ["-ss", f"{self._ff_offset:.3f}"]
        if abs(self._ff_speed - 1.0) > 1e-3:
            args += ["-af", self._atempo_chain(self._ff_speed)]
        args.append(self.path)
        return subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )

    @staticmethod
    def _atempo_chain(speed: float) -> str:
        chain = []
        remaining = speed
        while remaining > 2.0:
            chain.append("atempo=2.0")
            remaining /= 2.0
        while remaining < 0.5:
            chain.append("atempo=0.5")
            remaining *= 2.0
        chain.append(f"atempo={remaining:.4f}")
        return ",".join(chain)

    # --------------- shared helpers ---------------

    @staticmethod
    def _kill_proc_pg(proc: subprocess.Popen) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            try:
                proc.kill()
            except OSError:
                pass
        try:
            proc.wait(timeout=0.5)
        except (subprocess.TimeoutExpired, OSError):
            pass
