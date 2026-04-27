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
_STILL_THRESHOLD_RAD_S = 0.01      # ~0.57 deg/s
_STILL_HOLD_SAMPLES = 200          # ~0.2 s of stillness before bias refines
_STILL_BIAS_ALPHA = 2.0e-4


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
    ) -> None:
        self.calibration_target = calibration_samples
        self.alpha = complementary_alpha

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

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def calibration_progress(self) -> float:
        if self.calibration_target <= 0:
            return 1.0
        return min(1.0, self._calib_count / self.calibration_target)

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

        return self.absolute()
