# xreal-one-mac-probe

Tiny Python probe to validate that the **XREAL One / One Pro** IMU TCP
stream works on macOS — step 1 of porting `vr2xr` (Android) to a Mac
VR180 player.

The protocol layer is a faithful port of [`io.onexr`](https://github.com/Skarian/one-xr)
from the `vr2xr` project: 6-byte magic header (`0x28|0x27, 0x36, len_be_u32`),
128-byte body, dual-firmware header support, complementary filter
(α=0.96), 500-sample startup gyro calibration.

No third-party dependencies — pure stdlib.

## macOS Local Network permission (read this first)

On macOS 14+ Apple gates outbound connections to private/link-local
addresses (10/8, 192.168/16, **169.254/16**) behind a per-app **Local
Network** permission. The XREAL One Pro lives at `169.254.2.1`, so the
terminal app you run `probe.py` from needs that permission, otherwise
every `connect()` returns `EHOSTUNREACH` (errno 65) even though `nc`
works (system tools are exempt).

To enable:

1. **System Settings → Privacy & Security → Local Network**
2. Toggle on your terminal (Terminal / iTerm / Warp / ...).
3. Re-run `python3 probe.py dump` from that terminal.

If your terminal isn't in the Local Network list, it means the system
never saw an outbound link-local connection from it yet:

```bash
tccutil reset All com.apple.Terminal       # for Apple Terminal
tccutil reset All com.googlecode.iterm2    # for iTerm
# Then re-run probe.py — the system will pop a one-time prompt; choose Allow.
```

A one-shot `sudo python3 probe.py dump` also works as a sanity test
(sudo bypasses the prompt) but is not a long-term workaround.

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
probe.py              # diagnostic CLI: dump / imu / pose (stdlib only)
viewer.py             # Phase 1 viewer: GL test card + head tracking (deps below)
test_protocol.py      # offline self-test on synthetic frames
xreal_one/
  protocol.py         # frame parser, dual-magic, IMU/MAG decode
  tracker.py          # gyro calibration + complementary filter
  stream.py           # threaded pose source (used by viewer.py)
  __init__.py
```

## Viewer

Side-by-side OpenGL viewer that pulls live pose from the glasses'
TCP IMU stream and renders one of three modes per eye:

* **testcard** — procedural head-tracked colored sphere with a
  yaw/pitch grid. No video needed; validates the GL + tracker pipeline.
* **fisheye** — VR180 equiangular fisheye reverse-projection from a
  side-by-side video texture (the typical YouTube VR180 / Insta360 /
  Canon RF 5.2mm dual-fisheye layout).
* **equirect** — VR180 equirectangular reverse-projection (180° wide
  × 180° tall, two halves SBS).
* **flat-sbs** — flat 3D video, left half of frame = left eye, right
  half = right eye. Head tracking is ignored; the picture stays
  fixed to the viewport like a virtual 3D screen. Use this for
  ordinary 3D side-by-side files (e.g. Bilibili 4K [3D] dance videos).
* **flat-tb** — flat 3D top-bottom (over/under): top half = left eye,
  bottom half = right eye.

```bash
pip install -r requirements.txt

# 1. Switch the glasses to Full SBS (manual, via the glasses' button).
# 2. Confirm in System Settings -> Displays that it shows up as 3840x1080.
# 3. Run the viewer with sudo (until macOS Local Network permission is
#    granted — see the section above).

# Phase 1: test card only, no video file needed.
sudo python3 viewer.py

# Phase 2: with a VR180 video file.
sudo python3 viewer.py /path/to/vr180_test.mp4
```

The decoder uploads each frame at its native source resolution by
default. Pass `--decode-width N --decode-height M` to downscale during
decode (e.g. for older hardware or slower machines).

```bash
sudo python3 viewer.py video.mp4 --display 1            # explicit display
sudo python3 viewer.py video.mp4 --windowed             # debug 1920x540
sudo python3 viewer.py video.mp4 --proj equirect        # start in equirect mode
sudo python3 viewer.py video.mp4 --fisheye-fov 200      # tune lens FOV
sudo python3 viewer.py --no-tracker                     # GL-only, no glasses
```

### Controls

| key | action |
|-----|--------|
| F   | toggle fullscreen |
| T   | zero view (current heading = forward) |
| R   | recalibrate gyro (keep glasses still!) |
| M   | cycle projection mode (testcard / fisheye / equirect) |
| ↑/↓ | zoom in / out (decrease / increase rendered FOV) |
| 0   | reset FOV to the value passed via --fov (default 50) |
| V   | toggle vertical flip (use if the source content is upside-down) |
| O   | cycle 0 / 90 / 180 / 270 deg per-eye rotation (source layout varies) |
| P   | invert pitch sign (default already inverted; toggles back) |
| Y   | invert yaw sign |
| L   | invert roll sign |
| D   | toggle the green SBS-split debug line |
| Q   | quit |

Audio: by default the viewer spawns `ffplay -nodisp -loop 0` (or falls
back to macOS `afplay`) on the same video file so the AAC track plays
through the system speakers. Pass `--mute` to disable.

## Provenance

Algorithm logic mirrors:

- [`Skarian/one-xr`](https://github.com/Skarian/one-xr)
  (`io.onexr.OneXrReportMessageParser`, `io.onexr.OneXrHeadTracker`)
- [`SamiMitwalli/One-Pro-IMU-Retriever-Demo`](https://github.com/SamiMitwalli/One-Pro-IMU-Retriever-Demo)
  for the original Python proof-of-concept on the same TCP stream.

## License

MIT.
