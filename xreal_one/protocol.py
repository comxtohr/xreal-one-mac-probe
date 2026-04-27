"""XREAL One/One Pro IMU stream protocol.

Faithful port of the framing logic in `io.onexr.OneXrReportMessageParser`
(https://github.com/Skarian/one-xr).

Wire format
-----------
Every message is a 6-byte header followed by a 128-byte body.

Header (6 bytes):
    [0]    magic0  : 0x28 (older firmware) or 0x27 (newer firmware)
    [1]    magic1  : 0x36
    [2..6] length  : big-endian uint32, always 128 for valid reports

Body (128 bytes, all multi-byte fields little-endian):
    [0x00] device_id              u64
    [0x08] hmd_time_nanos_device  u64
    [0x10] (8 bytes unused)
    [0x18] report_type            u32   0x0B = IMU, 0x04 = MAGNETOMETER
    [0x1C] gx, gy, gz             f32 x 3   (rad/s in device frame)
    [0x28] ax, ay, az             f32 x 3
    [0x34] mx, my, mz             f32 x 3
    [0x40] temperature_celsius    f32
    [0x44] imu_id                 u8
    [0x45] frame_id               u8 x 3
    [0x48..] trailer (56 bytes, ignored)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import struct

DEFAULT_HOST = "169.254.2.1"
DEFAULT_STREAM_PORT = 52998
DEFAULT_CONTROL_PORT = 52999

_MAGIC0_PRIMARY = 0x28
_MAGIC0_ALTERNATE = 0x27
_MAGIC1 = 0x36
_HEADER_BYTES = 6
_EXPECTED_BODY_BYTES = 128
_MAX_PENDING_BYTES = 131_072

REPORT_TYPE_IMU = 0x0000000B
REPORT_TYPE_MAG = 0x00000004


@dataclass(frozen=True)
class ImuReport:
    device_id: int
    hmd_time_nanos: int
    gx: float
    gy: float
    gz: float
    ax: float
    ay: float
    az: float
    temperature_c: float
    imu_id: int
    frame_id: Tuple[int, int, int]


@dataclass(frozen=True)
class MagReport:
    device_id: int
    hmd_time_nanos: int
    mx: float
    my: float
    mz: float
    temperature_c: float
    imu_id: int
    frame_id: Tuple[int, int, int]


@dataclass
class ParseStats:
    parsed: int = 0
    imu: int = 0
    mag: int = 0
    dropped_bytes: int = 0
    invalid_length: int = 0
    decode_error: int = 0
    unknown_type: int = 0


def _decode_body(body: bytes) -> Optional[object]:
    if len(body) != _EXPECTED_BODY_BYTES:
        return None
    try:
        device_id = struct.unpack_from("<Q", body, 0x00)[0]
        hmd_time = struct.unpack_from("<Q", body, 0x08)[0]
        report_type = struct.unpack_from("<I", body, 0x18)[0]
        gx, gy, gz, ax, ay, az, mx, my, mz, temp = struct.unpack_from(
            "<ffffffffff", body, 0x1C
        )
        imu_id = body[0x44]
        frame_id = (body[0x45], body[0x46], body[0x47])
    except struct.error:
        return None

    if report_type == REPORT_TYPE_IMU:
        return ImuReport(
            device_id=device_id,
            hmd_time_nanos=hmd_time,
            gx=gx, gy=gy, gz=gz,
            ax=ax, ay=ay, az=az,
            temperature_c=temp,
            imu_id=imu_id,
            frame_id=frame_id,
        )
    if report_type == REPORT_TYPE_MAG:
        return MagReport(
            device_id=device_id,
            hmd_time_nanos=hmd_time,
            mx=mx, my=my, mz=mz,
            temperature_c=temp,
            imu_id=imu_id,
            frame_id=frame_id,
        )
    return "unknown"


def _find_header(buf: bytes) -> int:
    n = len(buf)
    if n < 2:
        return -1
    for i in range(n - 1):
        b0 = buf[i]
        if (b0 == _MAGIC0_PRIMARY or b0 == _MAGIC0_ALTERNATE) and buf[i + 1] == _MAGIC1:
            return i
    return -1


class StreamFramer:
    """Stateful framer: feed bytes in, get parsed reports out."""

    def __init__(self) -> None:
        self._pending = bytearray()
        self.stats = ParseStats()

    def append(self, chunk: bytes) -> List[object]:
        if not chunk:
            return []
        self._pending.extend(chunk)
        out: List[object] = []

        while len(self._pending) >= _HEADER_BYTES:
            idx = _find_header(self._pending)
            if idx < 0:
                drop = len(self._pending) - 1
                if drop > 0:
                    self.stats.dropped_bytes += drop
                    del self._pending[:drop]
                break

            if idx > 0:
                self.stats.dropped_bytes += idx
                del self._pending[:idx]
                if len(self._pending) < _HEADER_BYTES:
                    break

            body_len = struct.unpack_from(">I", self._pending, 2)[0]
            if body_len != _EXPECTED_BODY_BYTES:
                self.stats.invalid_length += 1
                self.stats.dropped_bytes += 1
                del self._pending[:1]
                continue

            total = _HEADER_BYTES + body_len
            if len(self._pending) < total:
                break

            body = bytes(self._pending[_HEADER_BYTES:total])
            del self._pending[:total]

            decoded = _decode_body(body)
            if decoded is None:
                self.stats.decode_error += 1
            elif decoded == "unknown":
                self.stats.unknown_type += 1
            else:
                self.stats.parsed += 1
                if isinstance(decoded, ImuReport):
                    self.stats.imu += 1
                elif isinstance(decoded, MagReport):
                    self.stats.mag += 1
                out.append(decoded)

            if len(self._pending) > _MAX_PENDING_BYTES:
                drop = len(self._pending) - _MAX_PENDING_BYTES
                self.stats.dropped_bytes += drop
                del self._pending[:drop]

        return out
