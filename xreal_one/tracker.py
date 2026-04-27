"""Head tracker: gyro startup calibration + complementary filter.

Mirrors the algorithm in `io.onexr.OneXrHeadTracker` (defaults from
`HeadTrackingStreamConfig`):

* 500 stationary samples to estimate per-axis gyro bias
* Complementary filter alpha = 0.96 (96% gyro / 4% accelerometer)
* Accel frame remap to tracker frame: (x, y, z) <- (raw_z, raw_y, raw_x)
* Yaw is gyro-only (no magnetometer fusion)
* Pitch and roll are corrected from accelerometer when |a| > 0.01

Without the device control session we cannot pull factory bias / accel
bias, so they are treated as zero. Startup calibration absorbs the
constant component of gyro bias at the current temperature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math

from .protocol import ImuReport

_DEFAULT_CALIBRATION_TARGET = 500
_DEFAULT_ALPHA = 0.96
_RAD_PER_SEC_TO_DEG_PER_SEC = 180.0 / math.pi

# Stationary auto-bias-refinement (runs after the initial 500-sample
# calibration). When the corrected gyro magnitude stays below the threshold
# for `_STILL_HOLD_SAMPLES` consecutive samples, we treat the glasses as
# motionless and slowly walk the residual bias toward whatever the gyro is
# reading. With a 1 kHz IMU, alpha=2e-4 gives roughly 5-second time-constant
# correction of small residual drift while leaving real motion unaffected.
_STILL_THRESHOLD_RAD_S = 0.012     # ~0.69 deg/s; wider window catches normal head pose
_STILL_HOLD_SAMPLES = 50           # ~50 ms of stillness before bias refines
_STILL_BIAS_ALPHA = 1.0e-3         # ~1 s time-constant correction

# Motion gate during the initial calibration. If gyro magnitude exceeds this
# threshold during the 500-sample collection, we treat the sample as
# contaminated and reset the accumulator. This forces calibration to wait
# for sustained stillness instead of finishing 0.5 s after start whether or
# not the head is actually still.
_CALIB_MOTION_THRESHOLD_RAD_S = 0.05   # ~2.9 deg/s

# Magnetometer-based yaw correction. After yaw is integrated each sample,
# we tilt-compensate the latest mag reading using the current pitch/roll
# to get a horizontal magnetic vector, derive a heading from it, and pull
# integrated yaw a tiny fraction toward that heading every sample.
# Offset (hard-iron) calibration is skipped: a constant mag offset only
# biases absolute heading, which we don't care about — yaw is recentered
# via T anyway. Relative yaw changes with rotation are still preserved.
_MAG_ALPHA = 1.0e-3                    # ~1 s time constant @ 1 kHz IMU


@dataclass
class Pose:
    pitch_deg: float
    yaw_deg: float
    roll_deg: float


def _wrap(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


class HeadTracker:
    def __init__(
        self,
        calibration_samples: int = _DEFAULT_CALIBRATION_TARGET,
        complementary_alpha: float = _DEFAULT_ALPHA,
        mag_alpha: float = _MAG_ALPHA,
    ) -> None:
        self.calibration_target = calibration_samples
        self.alpha = complementary_alpha
        self.mag_alpha = mag_alpha

        self._gyro_sum = [0.0, 0.0, 0.0]
        self._gyro_bias = [0.0, 0.0, 0.0]
        self._residual_bias = [0.0, 0.0, 0.0]
        self._still_count = 0
        self._calib_count = 0
        self._calibrated = False

        # Latest tracker-frame accel during calibration; used to seed pose
        # so the complementary filter doesn't snap from 0 to gravity-derived
        # angles in the first ~75 ms after calibration completes.
        self._last_ax_t = 0.0
        self._last_ay_t = 0.0
        self._last_az_t = 0.0

        self._pitch = 0.0
        self._yaw = 0.0
        self._roll = 0.0

        self._zero_pitch = 0.0
        self._zero_yaw = 0.0
        self._zero_roll = 0.0

        self._last_ts_ns: Optional[int] = None

        # Latest magnetometer sample (in IMU body frame, no remap applied
        # yet — that happens inside _apply_mag_yaw_correction).
        self._latest_mx: float = 0.0
        self._latest_my: float = 0.0
        self._latest_mz: float = 0.0
        self._mag_seen: bool = False

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def calibration_progress(self) -> float:
        if self.calibration_target <= 0:
            return 1.0
        return min(1.0, self._calib_count / self.calibration_target)

    def feed_mag(self, mx: float, my: float, mz: float) -> None:
        """Feed a magnetometer sample. Asynchronous w.r.t. IMU samples — we
        just cache the latest reading and consume it on each IMU update."""
        self._latest_mx = mx
        self._latest_my = my
        self._latest_mz = mz
        self._mag_seen = True

    def _apply_mag_yaw_correction(self) -> None:
        """Pull integrated yaw toward the tilt-compensated magnetic heading.

        Uses the same tracker-frame remap as the accelerometer: raw mag axes
        (x, y, z) -> (z, y, x). If the mag axis convention turns out to be
        different from accel, the user can disable this with --no-mag and
        revert to gyro-only yaw.
        """
        if self.mag_alpha <= 0.0:
            return

        mx_t = self._latest_mz
        my_t = self._latest_my
        mz_t = self._latest_mx

        norm = math.sqrt(mx_t * mx_t + my_t * my_t + mz_t * mz_t)
        if norm < 1e-6:
            return
        mx_t /= norm
        my_t /= norm
        mz_t /= norm

        pitch_rad = math.radians(self._pitch)
        roll_rad = math.radians(self._roll)
        cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
        cr, sr = math.cos(roll_rad), math.sin(roll_rad)

        # Rotate the body-frame mag into a horizontal frame using pitch/roll
        # so we can take a 2-D heading on the horizontal projection.
        mag_x_h = mx_t * cp + my_t * sr * sp + mz_t * cr * sp
        mag_y_h = my_t * cr - mz_t * sr

        mag_heading = math.degrees(math.atan2(-mag_y_h, mag_x_h))

        # Circular distance between integrated yaw and mag heading, then
        # nudge yaw toward the heading.
        diff = mag_heading - self._yaw
        while diff > 180.0:
            diff -= 360.0
        while diff < -180.0:
            diff += 360.0
        self._yaw = _wrap(self._yaw + self.mag_alpha * diff)

    def reset_calibration(self) -> None:
        self._gyro_sum = [0.0, 0.0, 0.0]
        self._gyro_bias = [0.0, 0.0, 0.0]
        self._residual_bias = [0.0, 0.0, 0.0]
        self._still_count = 0
        self._calib_count = 0
        self._calibrated = False
        self._pitch = self._yaw = self._roll = 0.0
        self._zero_pitch = self._zero_yaw = self._zero_roll = 0.0
        self._last_ts_ns = None
        self._mag_seen = False
        self._latest_mx = self._latest_my = self._latest_mz = 0.0

    def zero_view(self) -> None:
        self._zero_pitch = self._pitch
        self._zero_yaw = self._yaw
        self._zero_roll = self._roll

    def absolute(self) -> Pose:
        return Pose(self._pitch, self._yaw, self._roll)

    def relative(self) -> Pose:
        return Pose(
            _wrap(self._pitch - self._zero_pitch),
            _wrap(self._yaw - self._zero_yaw),
            _wrap(self._roll - self._zero_roll),
        )

    def feed(self, report: ImuReport) -> Optional[Pose]:
        """Feed one IMU report. Returns absolute pose once calibrated."""
        # Tracker-frame accel remap (matches OneXrTrackerSampleMapper):
        ax_t, ay_t, az_t = report.az, report.ay, report.ax

        if not self._calibrated:
            # Motion gate: discard accumulated samples and restart counting
            # if the glasses moved during the calibration window. Without
            # this, a half-second of head motion right after launch bakes
            # the motion into the gyro-bias estimate, which then causes
            # systemic drift after calibration completes.
            gmag2 = (
                report.gx * report.gx +
                report.gy * report.gy +
                report.gz * report.gz
            )
            if gmag2 > _CALIB_MOTION_THRESHOLD_RAD_S * _CALIB_MOTION_THRESHOLD_RAD_S:
                self._gyro_sum = [0.0, 0.0, 0.0]
                self._calib_count = 0
                self._last_ax_t = ax_t
                self._last_ay_t = ay_t
                self._last_az_t = az_t
                return None

            self._gyro_sum[0] += report.gx
            self._gyro_sum[1] += report.gy
            self._gyro_sum[2] += report.gz
            self._last_ax_t = ax_t
            self._last_ay_t = ay_t
            self._last_az_t = az_t
            self._calib_count += 1
            if self._calib_count >= self.calibration_target:
                divisor = max(1, self._calib_count)
                self._gyro_bias = [s / divisor for s in self._gyro_sum]
                self._calibrated = True

                # Seed absolute pose from the latest accel reading so the
                # complementary filter doesn't have to converge from 0 to
                # gravity-derived angles (which looked like a sudden 90 deg
                # snap on the user's display).
                accel_mag = math.sqrt(
                    self._last_ax_t * self._last_ax_t +
                    self._last_ay_t * self._last_ay_t +
                    self._last_az_t * self._last_az_t
                )
                if accel_mag > 0.01:
                    self._pitch = math.degrees(math.atan2(
                        -self._last_ax_t,
                        math.sqrt(
                            self._last_ay_t * self._last_ay_t +
                            self._last_az_t * self._last_az_t
                        ),
                    ))
                    self._roll = math.degrees(math.atan2(
                        self._last_ay_t, self._last_az_t
                    ))
                else:
                    self._pitch = 0.0
                    self._roll = 0.0
                self._yaw = 0.0

                # Auto-zero so relative pose starts at (0, 0, 0) regardless
                # of how the glasses were physically oriented during the
                # calibration phase.
                self._zero_pitch = self._pitch
                self._zero_yaw = self._yaw
                self._zero_roll = self._roll

                self._last_ts_ns = None
            return None

        if self._last_ts_ns is None:
            self._last_ts_ns = report.hmd_time_nanos
            return None

        dt_ns = report.hmd_time_nanos - self._last_ts_ns
        if dt_ns <= 0 or dt_ns > 1_000_000_000:
            self._last_ts_ns = report.hmd_time_nanos
            return None
        dt = dt_ns / 1e9

        gx = report.gx - self._gyro_bias[0] - self._residual_bias[0]
        gy = report.gy - self._gyro_bias[1] - self._residual_bias[1]
        gz = report.gz - self._gyro_bias[2] - self._residual_bias[2]

        # Stationary auto-bias-refinement.
        gmag2 = gx * gx + gy * gy + gz * gz
        if gmag2 < _STILL_THRESHOLD_RAD_S * _STILL_THRESHOLD_RAD_S:
            self._still_count += 1
            if self._still_count >= _STILL_HOLD_SAMPLES:
                self._residual_bias[0] += _STILL_BIAS_ALPHA * gx
                self._residual_bias[1] += _STILL_BIAS_ALPHA * gy
                self._residual_bias[2] += _STILL_BIAS_ALPHA * gz
        else:
            self._still_count = 0

        pitch_gyro = self._pitch + gx * _RAD_PER_SEC_TO_DEG_PER_SEC * dt
        yaw_gyro = self._yaw + gy * _RAD_PER_SEC_TO_DEG_PER_SEC * dt
        roll_gyro = self._roll + gz * _RAD_PER_SEC_TO_DEG_PER_SEC * dt

        accel_mag = math.sqrt(ax_t * ax_t + ay_t * ay_t + az_t * az_t)
        if accel_mag > 0.01:
            pitch_accel = math.degrees(
                math.atan2(-ax_t, math.sqrt(ay_t * ay_t + az_t * az_t))
            )
            roll_accel = math.degrees(math.atan2(ay_t, az_t))
            self._pitch = self.alpha * pitch_gyro + (1.0 - self.alpha) * pitch_accel
            self._yaw = yaw_gyro
            self._roll = self.alpha * roll_gyro + (1.0 - self.alpha) * roll_accel
        else:
            self._pitch = pitch_gyro
            self._yaw = yaw_gyro
            self._roll = roll_gyro

        self._pitch = _wrap(self._pitch)
        self._yaw = _wrap(self._yaw)
        self._roll = _wrap(self._roll)
        self._last_ts_ns = report.hmd_time_nanos

        if self._mag_seen:
            self._apply_mag_yaw_correction()

        return self.absolute()
