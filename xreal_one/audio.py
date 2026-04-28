"""In-process audio playback via PyAV + sounddevice.

No subprocess: audio is decoded by PyAV in a worker thread, resampled
to float32 stereo at the device sample rate, buffered into a small
deque, and pulled by the sounddevice OutputStream callback.

State changes are flag flips, not respawns:
- pause: callback fills with zeros; decoder also pauses (so we don't
  decode forward into the buffer while held).
- speed != 1: callback fills with zeros (we don't time-stretch). The
  viewer must call seek(video.latest_pts) when speed returns to 1x to
  re-anchor audio to the current video position.
- seek: clear buffer, decoder breaks out of the current pass and
  re-opens the container at the new offset.
- stop: stop sounddevice, signal decoder to exit, join.

All sub-frame latency, no spawn delay.

Dependencies: sounddevice (which bundles PortAudio in its macOS wheel).
"""

from __future__ import annotations

import atexit
import threading
import time
from typing import Optional

import av
import av.filter
import numpy as np

try:
    import sounddevice as sd  # type: ignore
    _SD_IMPORT_ERROR: Optional[str] = None
except (ImportError, OSError) as e:  # OSError if portaudio dylib missing
    sd = None  # type: ignore
    _SD_IMPORT_ERROR = str(e)


SAMPLE_RATE = 48000
CHANNELS = 2
# Cap the decode-ahead buffer so we don't grow unbounded while paused.
# 2 seconds at 48 kHz stereo = 96 000 samples. Plenty of headroom for
# the audio callback to ride out a hiccup, small enough that resume
# doesn't immediately replay a wide window of "future" audio.
MAX_BUFFER_SAMPLES = SAMPLE_RATE * 2


class AudioPlayer:
    def __init__(self, path: str, loop: bool = True) -> None:
        self.path = path
        self.loop = loop

        self._state_lock = threading.Lock()
        self._buffer_lock = threading.Lock()

        self._stop = False
        self._paused = False
        self._speed = 1.0
        self._speed_dirty = False  # decoder rebuilds atempo graph when set
        self._seek_target: Optional[float] = None

        # List of (samples, channels)=float32 chunks awaiting playback.
        self._buffer: list[np.ndarray] = []
        self._buffer_samples = 0  # cached sum of [c.shape[0] for c in _buffer]

        self._stream = None
        self._thread: Optional[threading.Thread] = None
        self._available = False

    @property
    def backend(self) -> Optional[str]:
        return "sounddevice" if self._available else None

    @property
    def seek_latency_seconds(self) -> float:
        # In-process; effectively just the audio device's hardware latency.
        return 0.0

    # ---------------- start / stop ----------------

    def start(self) -> None:
        if sd is None:
            print(
                "audio: sounddevice not available "
                f"({_SD_IMPORT_ERROR}); running silent.\n"
                "  install with: pip install sounddevice"
            )
            return
        # Verify the file actually has an audio stream (some VR180 demos don't).
        try:
            container = av.open(self.path)
            try:
                if not container.streams.audio:
                    print("audio: input file has no audio stream; running silent.")
                    return
            finally:
                container.close()
        except Exception as e:
            print(f"audio: failed to probe file ({e}); running silent.")
            return

        try:
            self._stream = sd.OutputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                callback=self._audio_callback,
                latency="low",
            )
            self._stream.start()
        except Exception as e:
            print(f"audio: failed to open output stream ({e}); running silent.")
            self._stream = None
            return

        self._thread = threading.Thread(
            target=self._decode_loop, name="xreal-audio", daemon=True
        )
        self._thread.start()
        self._available = True
        atexit.register(self.stop)

    def stop(self) -> None:
        self._stop = True
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None

    # ---------------- public state setters ----------------

    def set_state(self, active: bool, speed: float) -> None:
        """Pause / resume + playback speed.

        Speed != 1 runs the source through an atempo filter chain, so
        audio plays at the requested rate without pitch shift. Speed
        changes invalidate any pre-decoded samples (they were rendered
        at the old tempo); the buffer is cleared and the viewer should
        follow up with seek(video.latest_pts) to re-anchor.
        """
        speed = max(0.1, min(8.0, speed))
        speed_changed = False
        with self._state_lock:
            self._paused = not active
            if abs(speed - self._speed) > 1e-3:
                self._speed = speed
                self._speed_dirty = True
                speed_changed = True
        if speed_changed:
            with self._buffer_lock:
                self._buffer.clear()
                self._buffer_samples = 0

    def seek(self, offset_sec: float) -> None:
        offset_sec = max(0.0, offset_sec)
        with self._state_lock:
            self._seek_target = offset_sec
        # Drop in-flight samples so the next callback gets fresh content.
        with self._buffer_lock:
            self._buffer.clear()
            self._buffer_samples = 0

    # ---------------- audio thread (callback) ----------------

    def _audio_callback(self, outdata, frames, time_info, status) -> None:
        try:
            if self._paused:
                outdata.fill(0.0)
                return
            with self._buffer_lock:
                pos = 0
                while pos < frames and self._buffer:
                    chunk = self._buffer[0]
                    avail = chunk.shape[0]
                    need = frames - pos
                    if avail <= need:
                        outdata[pos:pos + avail] = chunk
                        pos += avail
                        self._buffer.pop(0)
                        self._buffer_samples -= avail
                    else:
                        outdata[pos:pos + need] = chunk[:need]
                        self._buffer[0] = chunk[need:]
                        self._buffer_samples -= need
                        pos = frames
                if pos < frames:
                    outdata[pos:].fill(0.0)  # underrun
        except Exception:
            outdata.fill(0.0)

    # ---------------- decoder thread ----------------

    def _decode_loop(self) -> None:
        while not self._stop:
            try:
                self._play_once()
            except Exception as e:
                # Don't spam; brief pause and retry.
                if not self._stop:
                    time.sleep(0.5)
            if not self.loop:
                break

    @staticmethod
    def _build_atempo_graph(template_stream, speed: float):
        """abuffer -> atempo*N -> abuffersink. Uses explicit args (not
        template=) for PyAV-version-stability. atempo's per-stage range
        is [0.5, 2.0]; chain stages for speeds outside that. Output
        frames stay in source format - we resample after the graph.

        Raises on configure failure so the caller can fall back."""
        graph = av.filter.Graph()
        tb = template_stream.time_base
        abuf_args = (
            f"time_base={tb.numerator}/{tb.denominator}:"
            f"sample_rate={template_stream.rate}:"
            f"sample_fmt={template_stream.format.name}:"
            f"channel_layout={template_stream.layout.name}"
        )
        buf = graph.add("abuffer", abuf_args)
        sink = graph.add("abuffersink")

        prev = buf
        stages = []
        remaining = speed
        while remaining > 2.0:
            stages.append(2.0)
            remaining /= 2.0
        while remaining < 0.5:
            stages.append(0.5)
            remaining *= 2.0
        stages.append(remaining)
        for s in stages:
            node = graph.add("atempo", f"{s:.4f}")
            prev.link_to(node)
            prev = node

        prev.link_to(sink)
        graph.configure()
        return graph

    def _make_pipeline(self, stream, speed: float):
        """Returns (graph_or_None, resampler).

        - 1x: graph=None, resampler converts to flt/48k/stereo (passthrough
          on a 48 kHz source).
        - other speeds: try atempo graph for pitch-preserving time-stretch;
          if that fails to build, fall back to a chipmunk-style resampler
          (output rate = 48000/speed, fed to a fixed 48 kHz device — plays
          faster but pitch shifts up). Always preferable to silence."""
        std_resampler = av.AudioResampler(
            format="flt", layout="stereo", rate=SAMPLE_RATE
        )
        if abs(speed - 1.0) < 1e-3:
            return None, std_resampler
        try:
            graph = self._build_atempo_graph(stream, speed)
            return graph, std_resampler
        except Exception as e:
            print(
                f"audio: atempo filter unavailable ({e}); "
                f"falling back to pitch-shifted speed-up at {speed:.1f}x"
            )
            chipmunk = av.AudioResampler(
                format="flt", layout="stereo",
                rate=max(4000, int(round(SAMPLE_RATE / max(0.1, speed)))),
            )
            return None, chipmunk

    def _push_chunk(self, arr) -> bool:
        """Push one numpy chunk to the playback buffer with backpressure.
        Returns False if a seek/stop/speed-change was requested mid-push."""
        if arr is None or arr.size == 0:
            return True
        try:
            arr = arr.reshape(-1, CHANNELS)
        except ValueError:
            return True
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        while not self._stop:
            with self._state_lock:
                if self._seek_target is not None or self._speed_dirty:
                    return False
            with self._buffer_lock:
                if self._buffer_samples < MAX_BUFFER_SAMPLES:
                    self._buffer.append(arr)
                    self._buffer_samples += arr.shape[0]
                    return True
            time.sleep(0.005)
        return False

    def _play_once(self) -> None:
        with self._state_lock:
            offset = self._seek_target if self._seek_target is not None else 0.0
            self._seek_target = None
            current_speed = self._speed
            self._speed_dirty = False

        container = av.open(self.path)
        try:
            stream = container.streams.audio[0]
            time_base = float(stream.time_base) if stream.time_base else 0.0

            if offset > 0.0 and time_base > 0.0:
                try:
                    container.seek(
                        int(offset / time_base),
                        any_frame=False,
                        backward=True,
                        stream=stream,
                    )
                except Exception:
                    pass

            graph, resampler = self._make_pipeline(stream, current_speed)

            for packet in container.demux(stream):
                if self._stop:
                    return
                with self._state_lock:
                    if self._seek_target is not None:
                        return
                    if self._speed_dirty:
                        current_speed = self._speed
                        self._speed_dirty = False
                        graph, resampler = self._make_pipeline(stream, current_speed)

                for frame in packet.decode():
                    if self._stop:
                        return
                    with self._state_lock:
                        if self._seek_target is not None:
                            return
                        if self._speed_dirty:
                            current_speed = self._speed
                            self._speed_dirty = False
                            graph, resampler = self._make_pipeline(stream, current_speed)

                    while self._paused and not self._stop:
                        with self._state_lock:
                            if self._seek_target is not None:
                                return
                            if self._speed_dirty:
                                current_speed = self._speed
                                self._speed_dirty = False
                                graph, resampler = self._make_pipeline(stream, current_speed)
                        time.sleep(0.05)
                    if self._stop:
                        return

                    # Source frames go through atempo (if non-1x and graph
                    # built) and then through resampler to flt/48k/stereo.
                    if graph is not None:
                        try:
                            graph.push(frame)
                        except Exception:
                            continue
                        stretched = []
                        while True:
                            try:
                                stretched.append(graph.pull())
                            except (BlockingIOError, av.error.EOFError):
                                break
                            except av.AVError:
                                break
                        for sf in stretched:
                            self._consume_resampler(resampler, sf)
                            with self._state_lock:
                                if self._seek_target is not None or self._speed_dirty:
                                    return
                    else:
                        if not self._consume_resampler(resampler, frame):
                            return
        finally:
            container.close()

    def _consume_resampler(self, resampler, frame) -> bool:
        """Resample one source frame and push the resulting samples to the
        playback buffer. Returns False on seek/speed-change interrupt."""
        out = resampler.resample(frame)
        if out is None:
            return True
        frames_out = out if isinstance(out, list) else [out]
        for f in frames_out:
            if f is None:
                continue
            try:
                arr = f.to_ndarray()
            except Exception:
                continue
            if not self._push_chunk(arr):
                return False
        return True
