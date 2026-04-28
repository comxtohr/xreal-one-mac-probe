"""Lightweight audio playback via an external process.

The viewer's main goal is video + head tracking; audio just needs to come
out of the speakers. We avoid pulling in a real audio library by spawning
whatever decode-and-play CLI is on the system: `ffplay` (preferred — it
supports atempo for speed-up and -ss for seek) or `afplay` (macOS fallback;
supports -r for rate but no seek).

When the viewer changes speed/seek/pause, it calls set_state() with the
new params and we kill+respawn the audio process at the new offset. There
is no proper PTS-level sync between video and audio — they're both running
off the same file independently — but the offset re-anchor on speed/seek
keeps them within ~100 ms of each other in practice.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Optional


class AudioPlayer:
    def __init__(self, path: str, loop: bool = True) -> None:
        self.path = path
        self.loop = loop
        self._proc: Optional[subprocess.Popen] = None
        self._backend: Optional[str] = None

        # Desired state. set_state() applies a change by killing the proc
        # and respawning with new args; reading these as a tuple makes it
        # cheap to check whether anything changed.
        self._active = False
        self._speed = 1.0
        self._offset = 0.0

    @property
    def backend(self) -> Optional[str]:
        return self._backend

    def start(self) -> None:
        """Detect backend and start playback at 1x from offset 0."""
        if shutil.which("ffplay"):
            self._backend = "ffplay"
        elif shutil.which("afplay"):
            self._backend = "afplay"
        else:
            print("audio: neither ffplay nor afplay found; running silent")
            return
        self.set_state(active=True, speed=1.0, offset_sec=0.0)

    def stop(self) -> None:
        self._kill_proc()

    def set_state(self, active: bool, speed: float, offset_sec: float) -> None:
        """Apply (active, speed, offset) atomically. If no relevant param
        changed and the proc is still alive, it's a no-op."""
        if self._backend is None and active:
            return
        proc_alive = self._proc is not None and self._proc.poll() is None
        nothing_changed = (
            active == self._active
            and abs(speed - self._speed) < 1e-3
            and abs(offset_sec - self._offset) < 0.05
            and (proc_alive or not active)
        )
        if nothing_changed:
            return

        self._active = active
        self._speed = max(0.1, min(8.0, speed))
        self._offset = max(0.0, offset_sec)
        self._kill_proc()
        if active:
            self._proc = self._spawn_once()

    @staticmethod
    def _atempo_chain(speed: float) -> str:
        """Build an atempo filter graph that achieves `speed`. Single-stage
        atempo is limited to [0.5, 2.0]; chain segments to extend the range."""
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

    def _kill_proc(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            self._proc.wait(timeout=1.0)
        except (subprocess.TimeoutExpired, OSError):
            try:
                self._proc.kill()
            except OSError:
                pass
        self._proc = None

    def _spawn_once(self) -> Optional[subprocess.Popen]:
        if self._backend == "ffplay":
            args = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error"]
            if self.loop:
                args += ["-loop", "0"]
            if self._offset > 0.0:
                args += ["-ss", f"{self._offset:.3f}"]
            if abs(self._speed - 1.0) > 1e-3:
                args += ["-af", self._atempo_chain(self._speed)]
            args.append(self.path)
        elif self._backend == "afplay":
            # afplay has -r for rate (no atempo, so pitch shifts) and no -ss.
            args = ["afplay"]
            if abs(self._speed - 1.0) > 1e-3:
                args += ["-r", f"{self._speed:.4f}"]
            args.append(self.path)
        else:
            return None
        return subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
