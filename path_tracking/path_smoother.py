"""
path_smoother.py
================
Implements cubic spline-based path smoothing for a sequence of 2D waypoints.

Algorithm
---------
* Chord-length parameterisation — arc-length proxy that avoids the uniform-
  parameter artefacts that produce oscillations near clustered waypoints.
* scipy.interpolate.CubicSpline with the 'not-a-knot' end condition gives C2
  continuity (continuous position, first derivative, second derivative).
* The public API works purely with plain Python lists / NumPy arrays so that
  the module is usable both standalone and from inside ROS2 nodes.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Waypoint = Tuple[float, float]


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class PathSmoother:
    """
    Smooths a list of discrete 2-D waypoints into a C2-continuous cubic spline.

    Parameters
    ----------
    waypoints : list of (x, y) tuples
        Raw waypoints from the global planner.  At least 3 points are required
        for the 'not-a-knot' end condition; 2-point paths fall back to linear.
    num_samples : int
        Number of points sampled from the spline to represent the smooth path.
    """

    def __init__(self, waypoints: List[Waypoint], num_samples: int = 500):
        if len(waypoints) < 2:
            raise ValueError("At least 2 waypoints are required.")

        self._waypoints = np.asarray(waypoints, dtype=float)
        self._num_samples = num_samples
        self._cs_x: CubicSpline | None = None
        self._cs_y: CubicSpline | None = None
        self._t_knots: np.ndarray | None = None
        self._smooth_path: np.ndarray | None = None

        self._fit()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _chord_parameterise(self) -> np.ndarray:
        """Return chord-length parameter values in [0, 1] for each waypoint."""
        pts = self._waypoints
        diffs = np.diff(pts, axis=0)                     # (n-1, 2)
        seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])  # (n-1,)
        cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        total = cumulative[-1]
        if total < 1e-12:
            raise ValueError("Waypoints are all coincident — cannot parameterise.")
        return cumulative / total                         # normalised to [0, 1]

    def _fit(self) -> None:
        """Fit cubic splines for x(t) and y(t) using chord-length parameter."""
        t = self._chord_parameterise()
        pts = self._waypoints

        bc_type = "not-a-knot" if len(pts) >= 3 else None
        self._cs_x = CubicSpline(t, pts[:, 0], bc_type=bc_type)
        self._cs_y = CubicSpline(t, pts[:, 1], bc_type=bc_type)
        self._t_knots = t

        # Pre-compute the sampled smooth path
        t_fine = np.linspace(0.0, 1.0, self._num_samples)
        self._smooth_path = np.column_stack(
            [self._cs_x(t_fine), self._cs_y(t_fine)]
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_smooth_path(self) -> np.ndarray:
        """
        Return the smooth path as an (N, 2) NumPy array of (x, y) points.

        Returns
        -------
        np.ndarray, shape (num_samples, 2)
        """
        return self._smooth_path.copy()

    def evaluate(self, t: float | np.ndarray) -> np.ndarray:
        """
        Evaluate the spline at arbitrary parameter value(s) t ∈ [0, 1].

        Parameters
        ----------
        t : float or array-like

        Returns
        -------
        np.ndarray, shape (..., 2)
        """
        t = np.asarray(t)
        return np.stack([self._cs_x(t), self._cs_y(t)], axis=-1)

    def evaluate_derivative(self, t: float | np.ndarray, order: int = 1) -> np.ndarray:
        """
        Evaluate the n-th derivative of the spline at parameter t.

        Parameters
        ----------
        t     : float or array-like
        order : int (1 = tangent, 2 = curvature-related)

        Returns
        -------
        np.ndarray, shape (..., 2)
        """
        t = np.asarray(t)
        return np.stack(
            [self._cs_x(t, order), self._cs_y(t, order)], axis=-1
        )

    def arc_length(self, t_start: float = 0.0, t_end: float = 1.0,
                   n_segments: int = 1000) -> float:
        """
        Numerically integrate the arc length of the spline between t_start and t_end.

        Parameters
        ----------
        t_start, t_end : float  — parameter range
        n_segments      : int   — integration resolution

        Returns
        -------
        float  — arc length in the same units as the waypoint coordinates
        """
        t_vals = np.linspace(t_start, t_end, n_segments + 1)
        dxdt = self._cs_x(t_vals, 1)
        dydt = self._cs_y(t_vals, 1)
        speed = np.hypot(dxdt, dydt)
        dt = (t_end - t_start) / n_segments
        # np.trapezoid is the NumPy 2.0 name; fall back to np.trapz for older installs
        trapz = getattr(np, "trapezoid", np.trapz) if hasattr(np, "trapz") else np.trapezoid
        return float(trapz(speed, dx=dt))

    def total_arc_length(self) -> float:
        """Return the total arc length of the smooth path."""
        return self.arc_length(0.0, 1.0)

    @property
    def waypoints(self) -> np.ndarray:
        """Original waypoints as (N, 2) array."""
        return self._waypoints.copy()

    @property
    def t_knots(self) -> np.ndarray:
        """Chord-length parameter values at each original waypoint."""
        return self._t_knots.copy()
