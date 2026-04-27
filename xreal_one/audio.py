"""Lightweight audio playback via an external process.

The viewer's main goal is video + head tracking; audio just needs to come
out of the speakers. We avoid pulling in a real audio library (sounddevice,
PortAudio bindings, etc.) by spawning whatever decode-and-play CLI is on
the system: `ffplay` (looped, cross-platform) preferred, `afplay` as a
macOS fallback (single-shot — Mac's built-in tool can't loop, so we
respawn on EOF).

There is no PTS-level sync with the video renderer; both start at roughly
the same wall-clock instant. For a 3-minute clip that's tight enough.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from typing import Optional


class AudioPlayer:
    def __init__(self, path: str, loop: bool = True) -> None:
        self.path = path
        self.loop = loop
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._backend: Optional[str] = None

    @property
    def backend(self) -> Optional[str]:
        return self._backend

    def start(self) -> None:
        if self._running:
            return
        if shutil.which("ffplay"):
            self._backend = "ffplay"
        elif shutil.which("afplay"):
            self._backend = "afplay"
        else:
            print("audio: neither ffplay nor afplay found; running silent")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run, name="xreal-audio", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=1.5)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    self._proc.kill()
                except OSError:
                    pass
            self._proc = None
        if self._thread is not None:
            self._thread.join(timeout=1.5)

    def _spawn_once(self) -> Optional[subprocess.Popen]:
        if self._backend == "ffplay":
            args = [
                "ffplay", "-nodisp", "-autoexit",
                "-loglevel", "error",
            ]
            if self.loop:
                args += ["-loop", "0"]
            args.append(self.path)
        elif self._backend == "afplay":
            args = ["afplay", self.path]
        else:
            return None
        return subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _run(self) -> None:
        # ffplay handles looping itself; afplay does not — we restart it.
        while self._running:
            self._proc = self._spawn_once()
            if self._proc is None:
                return
            self._proc.wait()
            if self._backend == "ffplay" or not self.loop:
                # ffplay -loop 0 only exits on terminate(), and afplay no-loop
                # mode is intentional.
                return
