#!/usr/bin/env python3
"""XREAL One/One Pro Mac probe.

Three modes (no third-party deps, just the stdlib):

    python probe.py dump      # raw frame stats, magic-byte detection
    python probe.py imu       # tail of decoded IMU samples
    python probe.py pose      # live yaw/pitch/roll after calibration

Run with the glasses sitting still on a flat surface for calibration in
`pose` mode. After calibration completes, put the glasses on and move
your head; the angles should react smoothly. Press T+Enter to set the
current heading as forward, R+Enter to recalibrate, Q+Enter to quit.

The script connects directly to 169.254.2.1:52998 by default. macOS
auto-routes that link-local address through the corresponding
`XREAL One Pro` Ethernet interface, so no manual binding is needed.
"""

from __future__ import annotations

import argparse
import select
import socket
import sys
import time
from typing import List, Optional

from xreal_one import (
    DEFAULT_HOST,
    DEFAULT_STREAM_PORT,
    HeadTracker,
    ImuReport,
    MagReport,
    Pose,
    StreamFramer,
)


def _connect(host: str, port: int, timeout: float = 2.0) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect((host, port))
    sock.settimeout(0.5)
    return sock


def _read_chunk(sock: socket.socket) -> bytes:
    try:
        return sock.recv(4096)
    except socket.timeout:
        return b""


def _stdin_lines_nonblocking() -> List[str]:
    """Drain pending stdin lines without blocking. POSIX only."""
    out: List[str] = []
    while True:
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            break
        line = sys.stdin.readline()
        if not line:
            break
        out.append(line.strip().lower())
    return out


def _format_pose(pose: Pose) -> str:
    return (
        f"pitch {pose.pitch_deg:+7.2f}  "
        f"yaw {pose.yaw_deg:+7.2f}  "
        f"roll {pose.roll_deg:+7.2f}"
    )


def cmd_dump(host: str, port: int, seconds: float) -> int:
    print(f"connecting to {host}:{port} (dump mode, {seconds}s)...", flush=True)
    sock = _connect(host, port)
    framer = StreamFramer()
    detected_magic: Optional[int] = None
    deadline = time.monotonic() + seconds

    while time.monotonic() < deadline:
        chunk = _read_chunk(sock)
        if chunk:
            if detected_magic is None and len(chunk) >= 2:
                # First two bytes after a fresh connect are the start of a frame.
                if chunk[0] in (0x27, 0x28) and chunk[1] == 0x36:
                    detected_magic = chunk[0]
                    print(
                        f"first magic byte = 0x{detected_magic:02X} "
                        f"({'old firmware' if detected_magic == 0x28 else 'new firmware'})",
                        flush=True,
                    )
            framer.append(chunk)
        time.sleep(0.01)

    sock.close()
    s = framer.stats
    print(
        f"\nframes parsed={s.parsed}  imu={s.imu}  mag={s.mag}  "
        f"dropped_bytes={s.dropped_bytes}  invalid_len={s.invalid_length}  "
        f"decode_err={s.decode_error}  unknown_type={s.unknown_type}"
    )
    if s.imu == 0:
        print("WARNING: no IMU reports decoded. Header bytes may have changed.")
        return 1
    return 0


def cmd_imu(host: str, port: int, count: int) -> int:
    print(f"connecting to {host}:{port} (imu mode, {count} samples)...", flush=True)
    sock = _connect(host, port)
    framer = StreamFramer()
    seen = 0

    while seen < count:
        chunk = _read_chunk(sock)
        if not chunk:
            continue
        for report in framer.append(chunk):
            if isinstance(report, ImuReport):
                seen += 1
                print(
                    f"[{seen:>4}] t={report.hmd_time_nanos:>20}ns  "
                    f"g=({report.gx:+8.4f},{report.gy:+8.4f},{report.gz:+8.4f})  "
                    f"a=({report.ax:+8.4f},{report.ay:+8.4f},{report.az:+8.4f})  "
                    f"T={report.temperature_c:+5.1f}C"
                )
                if seen >= count:
                    break
            elif isinstance(report, MagReport):
                pass

    sock.close()
    return 0


def cmd_pose(host: str, port: int) -> int:
    print(f"connecting to {host}:{port} (pose mode)...", flush=True)
    print("keep glasses still on a flat surface during calibration.", flush=True)
    print("commands: t=zero  r=recalibrate  q=quit  (press Enter after letter)\n", flush=True)

    sock = _connect(host, port)
    framer = StreamFramer()
    tracker = HeadTracker()

    last_print = 0.0
    last_calib_pct = -1
    running = True

    try:
        while running:
            chunk = _read_chunk(sock)
            if chunk:
                for report in framer.append(chunk):
                    if isinstance(report, ImuReport):
                        tracker.feed(report)

            for line in _stdin_lines_nonblocking():
                if line == "q":
                    running = False
                elif line == "t":
                    if tracker.is_calibrated:
                        tracker.zero_view()
                        print("\n[zeroed]", flush=True)
                elif line == "r":
                    tracker.reset_calibration()
                    last_calib_pct = -1
                    print("\n[recalibrating]", flush=True)

            now = time.monotonic()
            if not tracker.is_calibrated:
                pct = int(tracker.calibration_progress * 100)
                if pct != last_calib_pct and now - last_print > 0.1:
                    bar = "#" * (pct // 5) + "." * (20 - pct // 5)
                    sys.stdout.write(f"\rcalibrating [{bar}] {pct:3d}%")
                    sys.stdout.flush()
                    last_calib_pct = pct
                    last_print = now
            else:
                if now - last_print > 0.05:
                    rel = tracker.relative()
                    sys.stdout.write(f"\r{_format_pose(rel)}    ")
                    sys.stdout.flush()
                    last_print = now
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        print()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="XREAL One Pro Mac protocol probe.")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_STREAM_PORT)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dump", help="count raw frames for N seconds")
    d.add_argument("--seconds", type=float, default=3.0)

    i = sub.add_parser("imu", help="print N decoded IMU samples")
    i.add_argument("--count", type=int, default=20)

    sub.add_parser("pose", help="live pose after calibration")

    args = p.parse_args()
    try:
        if args.cmd == "dump":
            return cmd_dump(args.host, args.port, args.seconds)
        if args.cmd == "imu":
            return cmd_imu(args.host, args.port, args.count)
        if args.cmd == "pose":
            return cmd_pose(args.host, args.port)
    except (ConnectionRefusedError, OSError) as e:
        print(f"connection failed: {e}", file=sys.stderr)
        print(
            f"check that the glasses are connected and `nc -zv {args.host} {args.port}` succeeds.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
