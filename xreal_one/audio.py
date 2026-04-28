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

        # ffplay/afplay state. Because ffplay has no IPC, every change
        # respawns the process, but we estimate "where it was playing" by
        # remembering the offset at the last spawn and the wall clock at
        # spawn time, so resume from pause continues approximately at the
        # right spot instead of restarting from the last seek point.
        self._ff_proc: Optional[subprocess.Popen] = None
        self._ff_active: bool = False
        self._ff_speed: float = 1.0
        self._ff_play_offset: float = 0.0       # source pts at the moment we spawned
        self._ff_play_wall: float = 0.0         # monotonic wall clock at spawn

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

    def set_state(self, active: bool, speed: float) -> None:
        """Apply pause/play and speed only - position is left alone. Use
        seek() for an actual jump."""
        if self._backend == "mpv":
            self._set_state_mpv(active, speed)
        elif self._backend == "ffplay":
            self._set_state_ffplay(active, speed)
        # afplay has no IPC; ignore.

    def seek(self, offset_sec: float) -> None:
        """Jump to a specific position in seconds."""
        offset_sec = max(0.0, offset_sec)
        if self._backend == "mpv":
            self._seek_mpv(offset_sec)
        elif self._backend == "ffplay":
            self._seek_ffplay(offset_sec)

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

    def _set_state_mpv(self, active: bool, speed: float) -> None:
        speed = max(0.1, min(8.0, speed))
        # Pause is essentially free; speed change is internal time-stretch
        # (no buffer drop). Crucially, do NOT send seek here — that would
        # force mpv to re-decode every pause toggle, audible as "audio
        # restarts from a different spot".
        self._mpv_send({"command": ["set_property", "pause", not active]})
        self._mpv_send({"command": ["set_property", "speed", float(speed)]})

    def _seek_mpv(self, offset_sec: float) -> None:
        # Default seek mode (no "exact"). mpv jumps to the nearest
        # decode-friendly point, ~10x faster than exact mode on HEVC and
        # plenty accurate for a manual progress-bar click.
        self._mpv_send({"command": ["seek", float(offset_sec), "absolute"]})

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
        self._ff_play_offset = 0.0
        self._ff_play_wall = time.monotonic()
        self._ff_proc = self._spawn_ffplay(self._ff_play_offset)

    def _start_afplay_initial(self) -> None:
        self._ff_active = True
        self._ff_proc = subprocess.Popen(
            ["afplay", self.path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )

    def _ffplay_estimate_pos(self) -> float:
        """Best guess of where ffplay is right now in the source file."""
        if self._ff_proc is None or not self._ff_active:
            return self._ff_play_offset
        elapsed = (time.monotonic() - self._ff_play_wall) * self._ff_speed
        return self._ff_play_offset + elapsed

    def _set_state_ffplay(self, active: bool, speed: float) -> None:
        speed = max(0.1, min(8.0, speed))
        proc_alive = self._ff_proc is not None and self._ff_proc.poll() is None
        if active == self._ff_active and abs(speed - self._ff_speed) < 1e-3 and proc_alive:
            return
        # Capture where we are RIGHT NOW so the new spawn picks up from there
        # instead of restarting at the last seek point.
        new_offset = self._ffplay_estimate_pos()
        self._ff_active = active
        self._ff_speed = speed
        if self._ff_proc is not None:
            self._kill_proc_pg(self._ff_proc)
            self._ff_proc = None
        if active:
            self._ff_play_offset = new_offset
            self._ff_play_wall = time.monotonic()
            self._ff_proc = self._spawn_ffplay(new_offset)

    def _seek_ffplay(self, offset_sec: float) -> None:
        if self._ff_proc is not None:
            self._kill_proc_pg(self._ff_proc)
            self._ff_proc = None
        self._ff_play_offset = offset_sec
        self._ff_play_wall = time.monotonic()
        if self._ff_active:
            self._ff_proc = self._spawn_ffplay(offset_sec)

    def _spawn_ffplay(self, offset_sec: float) -> Optional[subprocess.Popen]:
        args = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error"]
        if self.loop:
            args += ["-loop", "0"]
        if offset_sec > 0.0:
            args += ["-ss", f"{offset_sec:.3f}"]
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
