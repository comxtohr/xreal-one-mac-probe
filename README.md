# xreal-one-mac-probe

Tiny Python probe to validate that the **XREAL One / One Pro** IMU TCP
stream works on macOS — step 1 of porting `vr2xr` (Android) to a Mac
VR180 player.

The protocol layer is a faithful port of [`io.onexr`](https://github.com/Skarian/one-xr)
from the `vr2xr` project: 6-byte magic header (`0x28|0x27, 0x36, len_be_u32`),
128-byte body, dual-firmware header support, complementary filter
(α=0.96), 500-sample startup gyro calibration.

No third-party dependencies — pure stdlib.

## Prerequisites

1. Plug the **XREAL One Pro** into your Mac via USB-C.
2. Confirm macOS sees it as a network interface and assigns a link-local
   address. You should see something like:

   ```bash
   networksetup -listallhardwareports | grep -A1 XREAL
   # → en9 etc.
   ifconfig en9 | grep "inet "
   # → inet 169.254.2.10  netmask 0xffffff00 ...
   nc -zv 169.254.2.1 52998
   # → Connection ... succeeded!
   ```

3. Python 3.8+ (Mac system Python is fine).

## Quick start

```bash
git clone <this-repo-url>
cd xreal-one-mac-probe

# 1. Local sanity check — does the parser work?
python3 test_protocol.py
# → 6/6 passed

# 2. Plug in glasses, then count raw frames for 3 seconds:
python3 probe.py dump
# → first magic byte = 0x27 (new firmware)
# → frames parsed=2900  imu=2700  mag=200  ...

# 3. Print 20 decoded IMU samples:
python3 probe.py imu --count 20

# 4. Live pose (yaw/pitch/roll). Place glasses still on a flat surface
#    until calibration completes, then wear them and turn your head:
python3 probe.py pose
```

In `pose` mode:

- Type `t` then Enter to **zero** the current view.
- Type `r` then Enter to **recalibrate** (place glasses still again).
- Type `q` then Enter (or Ctrl+C) to **quit**.

## What "success" looks like

- `dump` reports a non-zero `imu` count (~1000 Hz from the device, so
  several thousand IMU frames in 3s).
- `imu` mode prints samples whose `g=(...)` values are tiny (≈ 0) when
  glasses are still and one component of `a=(...)` is ≈ ±9.8 (gravity).
- `pose` mode shows `yaw` reacting to left/right head turns, `pitch` to
  nodding, `roll` to head tilt — and angles drifting back toward 0
  when you look forward.

If `dump` shows `imu=0`, the wire format may have changed in newer
firmware than the one this code was written against. Open an issue
with the `dump` output.

## Files

```
probe.py              # CLI entry point (stdlib only)
test_protocol.py      # offline self-test on synthetic frames
xreal_one/
  protocol.py         # frame parser, dual-magic, IMU/MAG decode
  tracker.py          # gyro calibration + complementary filter
  __init__.py
```

## Provenance

Algorithm logic mirrors:

- [`Skarian/one-xr`](https://github.com/Skarian/one-xr)
  (`io.onexr.OneXrReportMessageParser`, `io.onexr.OneXrHeadTracker`)
- [`SamiMitwalli/One-Pro-IMU-Retriever-Demo`](https://github.com/SamiMitwalli/One-Pro-IMU-Retriever-Demo)
  for the original Python proof-of-concept on the same TCP stream.

## License

MIT.
