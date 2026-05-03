"""
trajectory_generator.py
========================
Converts a smooth 2-D path into a *time-parameterised* trajectory by applying
a trapezoidal (or constant) velocity profile along the arc length.

Design choices
--------------
* **Arc-length re-parameterisation** decouples speed assignment from the
  underlying spline parameter, so the robot travels at physically meaningful
  speeds rather than uniform parameter increments.
* **Trapezoidal velocity profile**: three phases — accelerate, cruise, decelerate.
  Falls back gracefully when the path is too short to reach cruise speed.
* Output format: trajectory = [(x, y, t), ...] plus optional heading (yaw).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

from path_tracking.path_smoother import PathSmoother


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryPoint:
    """A single point in the time-stamped trajectory."""
    x:   float
    y:   float
    t:   float          # time [s]
    v:   float = 0.0   # linear speed [m/s]
    yaw: float = 0.0   # heading [rad]

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.t)


@dataclass
class TrajectoryConfig:
    """Tunable parameters for trajectory generation."""
    max_speed:     float = 0.5    # m/s  — cruise speed
    accel:         float = 0.3    # m/s² — acceleration / deceleration magnitude
    dt:            float = 0.05   # s    — time step for output samples
    arc_n_segs:    int   = 2000   # integration resolution for arc length


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TrajectoryGenerator:
    """
    Generates a time-parameterised trajectory from a PathSmoother object.

    Parameters
    ----------
    smoother : PathSmoother
        Pre-fitted smooth path.
    config   : TrajectoryConfig
        Speed / timing parameters.
    """

    def __init__(self, smoother: PathSmoother, config: TrajectoryConfig | None = None):
        self._smoother = smoother
        self._cfg = config or TrajectoryConfig()
        self._trajectory: List[TrajectoryPoint] = []
        self._arc_table: np.ndarray | None = None   # (M, 2) — (s, t_spline)

        self._build_arc_length_table()
        self._generate()

    # ------------------------------------------------------------------
    # Private — arc-length look-up table
    # ------------------------------------------------------------------

    def _build_arc_length_table(self) -> None:
        """
        Build a dense table mapping arc-length s → spline parameter t so that
        we can invert s(t) to get t(s) via interpolation.
        """
        n = self._cfg.arc_n_segs
        t_vals = np.linspace(0.0, 1.0, n + 1)
        dxdt = self._smoother.evaluate_derivative(t_vals, 1)          # (n+1, 2)
        speed_t = np.hypot(dxdt[:, 0], dxdt[:, 1])                    # |dr/dt|
        # Cumulative arc length via trapezoidal rule
        dt = 1.0 / n
        ds = 0.5 * (speed_t[:-1] + speed_t[1:]) * dt
        s_vals = np.concatenate([[0.0], np.cumsum(ds)])
        self._arc_table = np.column_stack([s_vals, t_vals])            # (n+1, 2)
        self._total_length = float(s_vals[-1])

    def _s_to_t(self, s: np.ndarray) -> np.ndarray:
        """Invert s(t) to get spline parameter t for given arc-length s."""
        return np.interp(s, self._arc_table[:, 0], self._arc_table[:, 1])

    # ------------------------------------------------------------------
    # Private — trapezoidal velocity profile
    # ------------------------------------------------------------------

    def _trapezoidal_profile(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute (s_values, v_values) arrays over time using the trapezoidal
        profile.  Returns arc-length and speed at each time step.

        Three-phase motion:
            Phase 1: accelerate from 0 → v_max
            Phase 2: cruise at v_max
            Phase 3: decelerate from v_max → 0
        """
        v_max = self._cfg.max_speed
        a     = self._cfg.accel
        L     = self._total_length
        dt    = self._cfg.dt

        # Distance needed to ramp up/down
        d_ramp = v_max ** 2 / (2.0 * a)

        if 2.0 * d_ramp >= L:
            # Path too short to reach cruise speed — symmetric triangle profile
            v_peak = np.sqrt(a * L)
            t_ramp = v_peak / a
            t_total = 2.0 * t_ramp

            t_vals = np.arange(0.0, t_total + dt, dt)
            v_vals = np.where(
                t_vals <= t_ramp,
                a * t_vals,
                v_peak - a * (t_vals - t_ramp)
            )
            v_vals = np.clip(v_vals, 0.0, v_peak)
        else:
            # Full trapezoidal profile
            t_accel  = v_max / a
            d_cruise = L - 2.0 * d_ramp
            t_cruise = d_cruise / v_max
            t_total  = 2.0 * t_accel + t_cruise

            t_vals = np.arange(0.0, t_total + dt, dt)
            v_vals = np.zeros_like(t_vals)
            for i, t in enumerate(t_vals):
                if t <= t_accel:
                    v_vals[i] = a * t
                elif t <= t_accel + t_cruise:
                    v_vals[i] = v_max
                else:
                    v_vals[i] = v_max - a * (t - t_accel - t_cruise)
            v_vals = np.clip(v_vals, 0.0, v_max)

        # Integrate speed to get arc-length at each time step
        s_vals = np.zeros_like(t_vals)
        for i in range(1, len(t_vals)):
            s_vals[i] = s_vals[i - 1] + 0.5 * (v_vals[i - 1] + v_vals[i]) * dt
        s_vals = np.clip(s_vals, 0.0, self._total_length)

        return t_vals, v_vals, s_vals

    # ------------------------------------------------------------------
    # Private — assemble trajectory
    # ------------------------------------------------------------------

    def _generate(self) -> None:
        t_vals, v_vals, s_vals = self._trapezoidal_profile()
        spline_t = self._s_to_t(s_vals)                    # spline parameter
        positions = self._smoother.evaluate(spline_t)      # (N, 2)
        derivs    = self._smoother.evaluate_derivative(spline_t, 1)  # (N, 2)

        self._trajectory = []
        for i in range(len(t_vals)):
            yaw = float(np.arctan2(derivs[i, 1], derivs[i, 0]))
            pt  = TrajectoryPoint(
                x   = float(positions[i, 0]),
                y   = float(positions[i, 1]),
                t   = float(t_vals[i]),
                v   = float(v_vals[i]),
                yaw = yaw,
            )
            self._trajectory.append(pt)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_trajectory(self) -> List[TrajectoryPoint]:
        """
        Return the complete time-stamped trajectory.

        Returns
        -------
        list of TrajectoryPoint
            Each point carries (x, y, t, v, yaw).
        """
        return self._trajectory

    def as_tuples(self) -> List[Tuple[float, float, float]]:
        """
        Return trajectory as a list of (x, y, t) tuples — the format required
        by the assignment specification.

        Returns
        -------
        list of (x, y, t)
        """
        return [pt.as_tuple() for pt in self._trajectory]

    def get_arrays(self):
        """
        Return trajectory data as separate NumPy arrays for easy plotting.

        Returns
        -------
        xs, ys, ts, vs, yaws — all 1-D NumPy arrays
        """
        pts   = self._trajectory
        xs    = np.array([p.x   for p in pts])
        ys    = np.array([p.y   for p in pts])
        ts    = np.array([p.t   for p in pts])
        vs    = np.array([p.v   for p in pts])
        yaws  = np.array([p.yaw for p in pts])
        return xs, ys, ts, vs, yaws

    @property
    def total_time(self) -> float:
        """Total duration of the trajectory in seconds."""
        return self._trajectory[-1].t if self._trajectory else 0.0

    @property
    def total_length(self) -> float:
        """Total arc length of the path in metres."""
        return self._total_length
