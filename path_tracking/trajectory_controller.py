"""
trajectory_controller.py
========================
Pure Pursuit controller for a differential drive robot tracking a 2-D trajectory.

Algorithm — Pure Pursuit
-------------------------
Pure Pursuit computes the curvature κ of the circular arc that connects the
robot's current position to a *look-ahead point* on the reference path:

    κ = 2 * ld_y / L²

where
    L    = look-ahead distance
    ld_y = lateral error to the look-ahead point in the robot's local frame

This gives:
    angular velocity ω = v · κ
    linear  velocity v = reference speed (from trajectory) × progress gain

The look-ahead distance is made *adaptive* (scales with speed) to improve
stability at high speeds and precision at low speeds.

References
----------
Coulter, R.C. (1992). Implementation of the Pure Pursuit Path Tracking
Algorithm. CMU-RI-TR-92-01.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from path_tracking.trajectory_generator import TrajectoryPoint


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ControllerConfig:
    """Tunable parameters for the Pure Pursuit controller."""
    lookahead_min:   float = 0.3    # m  — minimum look-ahead distance
    lookahead_max:   float = 1.5    # m  — maximum look-ahead distance
    lookahead_k:     float = 0.6    # s  — ld = k · v  (adaptive gain)
    max_linear_vel:  float = 0.5    # m/s
    max_angular_vel: float = 1.5    # rad/s
    wheel_base:      float = 0.160  # m  — TurtleBot3 Burger wheelbase
    goal_tolerance:  float = 0.1    # m  — stop when within this distance of goal


# ---------------------------------------------------------------------------
# Robot state
# ---------------------------------------------------------------------------

@dataclass
class RobotState:
    """Current pose of the differential drive robot."""
    x:   float = 0.0
    y:   float = 0.0
    yaw: float = 0.0   # heading [rad]

    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])


# ---------------------------------------------------------------------------
# Controller output
# ---------------------------------------------------------------------------

@dataclass
class ControlCommand:
    """Velocity command for a differential drive robot."""
    linear:  float = 0.0   # m/s
    angular: float = 0.0   # rad/s

    def wheel_velocities(self, wheel_base: float) -> Tuple[float, float]:
        """
        Convert (v, ω) to individual left/right wheel speeds.

        v_left  = v - ω·b/2
        v_right = v + ω·b/2
        """
        v_left  = self.linear - self.angular * wheel_base / 2.0
        v_right = self.linear + self.angular * wheel_base / 2.0
        return v_left, v_right


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class PurePursuitController:
    """
    Pure Pursuit trajectory tracking controller.

    Parameters
    ----------
    trajectory : list of TrajectoryPoint
        Pre-computed time-stamped trajectory.
    config     : ControllerConfig
        Controller tuning parameters.
    """

    def __init__(self, trajectory: List[TrajectoryPoint],
                 config: ControllerConfig | None = None):
        self._traj  = trajectory
        self._cfg   = config or ControllerConfig()
        self._last_idx = 0       # track search progress for efficiency
        self._done  = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _adaptive_lookahead(self, speed: float) -> float:
        """Return look-ahead distance, clamped to [min, max]."""
        ld = self._cfg.lookahead_k * abs(speed)
        return float(np.clip(ld, self._cfg.lookahead_min, self._cfg.lookahead_max))

    def _find_lookahead_point(self, robot: RobotState,
                               ld: float) -> Tuple[np.ndarray, int]:
        """
        Search forward along the trajectory for the first point that is
        approximately `ld` metres ahead of the robot.

        Returns
        -------
        (lookahead_xy, index)
        """
        traj  = self._traj
        n     = len(traj)
        robot_pos = robot.position()

        # Search starting from last known index
        for i in range(self._last_idx, n):
            pt  = np.array([traj[i].x, traj[i].y])
            dist = float(np.linalg.norm(pt - robot_pos))
            if dist >= ld:
                self._last_idx = i
                return pt, i

        # Fell off the end — return final waypoint
        final = np.array([traj[-1].x, traj[-1].y])
        return final, n - 1

    def _nearest_point_index(self, robot: RobotState) -> int:
        """Return index of the nearest trajectory point to the robot."""
        traj    = self._traj
        robot_p = robot.position()
        pts     = np.array([[p.x, p.y] for p in traj])
        dists   = np.linalg.norm(pts - robot_p, axis=1)
        return int(np.argmin(dists))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_command(self, robot: RobotState) -> ControlCommand:
        """
        Compute velocity command for the current robot state.

        Parameters
        ----------
        robot : RobotState  — current pose

        Returns
        -------
        ControlCommand — (linear, angular) velocities
        """
        if self._done:
            return ControlCommand(0.0, 0.0)

        # Check goal reached
        goal = np.array([self._traj[-1].x, self._traj[-1].y])
        if np.linalg.norm(robot.position() - goal) < self._cfg.goal_tolerance:
            self._done = True
            return ControlCommand(0.0, 0.0)

        # Reference speed from nearest trajectory point
        nearest_idx = max(self._last_idx,
                          self._nearest_point_index(robot))
        nearest_idx = min(nearest_idx, len(self._traj) - 1)
        ref_speed   = self._traj[nearest_idx].v

        # Adaptive look-ahead
        ld = self._adaptive_lookahead(ref_speed)

        # Find look-ahead point
        lookahead_world, _ = self._find_lookahead_point(robot, ld)

        # Transform look-ahead point to robot frame
        dx = lookahead_world[0] - robot.x
        dy = lookahead_world[1] - robot.y
        cos_h = np.cos(robot.yaw)
        sin_h = np.sin(robot.yaw)
        ld_x  =  cos_h * dx + sin_h * dy   # forward in robot frame
        ld_y  = -sin_h * dx + cos_h * dy   # lateral in robot frame

        # Compute curvature κ = 2*ld_y / ld²
        ld_actual = float(np.hypot(ld_x, ld_y))
        if ld_actual < 1e-6:
            kappa = 0.0
        else:
            kappa = 2.0 * ld_y / (ld_actual ** 2)

        # Compute velocities
        v = float(np.clip(ref_speed, 0.0, self._cfg.max_linear_vel))
        omega = float(np.clip(v * kappa,
                              -self._cfg.max_angular_vel,
                               self._cfg.max_angular_vel))

        return ControlCommand(linear=v, angular=omega)

    def reset(self) -> None:
        """Reset the controller's search state."""
        self._last_idx = 0
        self._done = False

    @property
    def is_done(self) -> bool:
        """True when the robot has reached the goal tolerance."""
        return self._done
