"""Synthetic-frame self-test for the protocol/tracker port.

Run: python test_protocol.py
"""

import struct
import sys

from xreal_one import (
    REPORT_TYPE_IMU,
    REPORT_TYPE_MAG,
    HeadTracker,
    ImuReport,
    MagReport,
    StreamFramer,
)


def _build_body(report_type: int, gx=0.0, gy=0.0, gz=0.0,
                ax=0.0, ay=0.0, az=0.0, mx=0.0, my=0.0, mz=0.0,
                temp=30.0, t_ns=0, device_id=0xDEADBEEFCAFEBABE,
                imu_id=1, frame_id=(1, 2, 3)) -> bytes:
    body = bytearray(128)
    struct.pack_into("<Q", body, 0x00, device_id)
    struct.pack_into("<Q", body, 0x08, t_ns)
    struct.pack_into("<I", body, 0x18, report_type)
    struct.pack_into("<f", body, 0x1C, gx)
    struct.pack_into("<f", body, 0x20, gy)
    struct.pack_into("<f", body, 0x24, gz)
    struct.pack_into("<f", body, 0x28, ax)
    struct.pack_into("<f", body, 0x2C, ay)
    struct.pack_into("<f", body, 0x30, az)
    struct.pack_into("<f", body, 0x34, mx)
    struct.pack_into("<f", body, 0x38, my)
    struct.pack_into("<f", body, 0x3C, mz)
    struct.pack_into("<f", body, 0x40, temp)
    body[0x44] = imu_id
    body[0x45], body[0x46], body[0x47] = frame_id
    return bytes(body)


def _frame(magic0: int, body: bytes) -> bytes:
    header = bytes([magic0, 0x36]) + struct.pack(">I", len(body))
    return header + body


def test_old_magic_imu():
    f = _frame(0x28, _build_body(REPORT_TYPE_IMU, gx=0.1, ay=9.8))
    framer = StreamFramer()
    out = framer.append(f)
    assert len(out) == 1 and isinstance(out[0], ImuReport), out
    assert abs(out[0].gx - 0.1) < 1e-6
    assert abs(out[0].ay - 9.8) < 1e-6


def test_new_magic_imu():
    f = _frame(0x27, _build_body(REPORT_TYPE_IMU, gz=-0.5))
    out = StreamFramer().append(f)
    assert len(out) == 1 and isinstance(out[0], ImuReport)
    assert abs(out[0].gz + 0.5) < 1e-6


def test_mag_report():
    f = _frame(0x28, _build_body(REPORT_TYPE_MAG, mx=12.3))
    out = StreamFramer().append(f)
    assert len(out) == 1 and isinstance(out[0], MagReport)
    assert abs(out[0].mx - 12.3) < 1e-5


def test_garbage_then_two_frames():
    garbage = b"\x00\x99\xff\xab"
    f1 = _frame(0x28, _build_body(REPORT_TYPE_IMU, gx=1.0))
    f2 = _frame(0x27, _build_body(REPORT_TYPE_IMU, gx=2.0))
    framer = StreamFramer()
    out = framer.append(garbage + f1 + b"\x12" + f2)
    imus = [r for r in out if isinstance(r, ImuReport)]
    assert len(imus) == 2, out
    assert abs(imus[0].gx - 1.0) < 1e-6
    assert abs(imus[1].gx - 2.0) < 1e-6


def test_split_chunks():
    f = _frame(0x28, _build_body(REPORT_TYPE_IMU, gy=0.25))
    framer = StreamFramer()
    out = []
    for i in range(0, len(f), 7):
        out += framer.append(f[i : i + 7])
    assert len(out) == 1 and abs(out[0].gy - 0.25) < 1e-6


def test_tracker_calibration_then_pose():
    """Flat-on-table maps to tracker-frame az_t = +g, i.e. raw ax = +g
    after the (z, y, x) <- (raw) remap inside HeadTracker."""
    tracker = HeadTracker(calibration_samples=5, complementary_alpha=0.96)

    def _flat(t_ns):
        return ImuReport(
            device_id=0, hmd_time_nanos=t_ns,
            gx=0.0, gy=0.0, gz=0.0,
            ax=9.8, ay=0.0, az=0.0,
            temperature_c=30.0, imu_id=0, frame_id=(0, 0, 0),
        )

    for i in range(5):
        tracker.feed(_flat(i * 1_000_000))
    assert tracker.is_calibrated

    for i in range(200):
        tracker.feed(_flat((5 + i) * 1_000_000))

    abs_pose = tracker.absolute()
    assert abs(abs_pose.pitch_deg) < 1.0, abs_pose
    assert abs(abs_pose.roll_deg) < 1.0, abs_pose
    assert abs(abs_pose.yaw_deg) < 1.0, abs_pose


def main():
    tests = [
        test_old_magic_imu,
        test_new_magic_imu,
        test_mag_report,
        test_garbage_then_two_frames,
        test_split_chunks,
        test_tracker_calibration_then_pose,
    ]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"  ok   {fn.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL {fn.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
