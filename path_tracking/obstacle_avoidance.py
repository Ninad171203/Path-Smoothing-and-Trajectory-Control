"""
obstacle_avoidance.py
=====================
Extra Credit: Obstacle avoidance via Artificial Potential Fields (APF).

Approach
--------
The global path planner (cubic spline) is unaware of dynamic obstacles.
This module adds a *local* reactive layer that modifies the control command
at runtime by superimposing a repulsive potential-field force:

    F_rep = η · (1/d - 1/d0) · (1/d²) · ∇d      when d < d0
          = 0                                       otherwise

where
    d   = distance to nearest obstacle point
    d0  = obstacle influence radius
    η   = repulsive gain

The resulting repulsive velocity perturbation is blended with the Pure Pursuit
command so the robot both tracks the path *and* avoids obstacles.

Extending to a real robot
--------------------------
On a real TurtleBot3 the obstacle distances come from the LiDAR scan topic
(/scan). A ROS2 subscriber converts LaserScan → obstacle point cloud and
passes the nearest k points to ObstacleAvoider.avoid().

References
----------
Khatib, O. (1986). Real-time obstacle avoidance for manipulators and
mobile robots. IJRR, 5(1), 90-98.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

from path_tracking.trajectory_controller import ControlCommand, RobotState


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Obstacle:
    """A circular obstacle in the environment."""
    x:      float
    y:      float
    radius: float = 0.15    # m — physical radius of the obstacle

    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])


@dataclass
class APFConfig:
    """Tunable parameters for the Artificial Potential Field avoider."""
    influence_radius:  float = 0.8    # m  — d0: APF starts repelling
    repulsive_gain:    float = 0.4    # η  — repulsive force magnitude
    max_avoid_angular: float = 1.2    # rad/s — cap on avoidance angular velocity
    max_avoid_linear:  float = 0.3    # m/s  — cap on reduced speed near obstacles
    blend_alpha:       float = 0.6    # weight for avoidance vs tracking (0=tracking, 1=avoid)


# ---------------------------------------------------------------------------
# Obstacle avoider
# ---------------------------------------------------------------------------

class ObstacleAvoider:
    """
    Augments a Pure Pursuit command with APF-based repulsion.

    Parameters
    ----------
    obstacles : list of Obstacle
        Static (or periodically updated) obstacle list.
    config    : APFConfig
        APF tuning parameters.
    """

    def __init__(self, obstacles: List[Obstacle],
                 config: APFConfig | None = None):
        self._obstacles = obstacles
        self._cfg = config or APFConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_obstacles(self, obstacles: List[Obstacle]) -> None:
        """Replace the current obstacle list (call from LiDAR callback in ROS2)."""
        self._obstacles = obstacles

    def avoid(self, robot: RobotState,
              base_cmd: ControlCommand) -> ControlCommand:
        """
        Blend the base controller command with an APF repulsion signal.

        Parameters
        ----------
        robot    : RobotState      — current robot pose
        base_cmd : ControlCommand  — command from Pure Pursuit

        Returns
        -------
        ControlCommand  — modified (v, ω) accounting for obstacles
        """
        if not self._obstacles:
            return base_cmd

        repulsive_force = np.zeros(2)
        robot_pos = robot.position()
        d0 = self._cfg.influence_radius
        eta = self._cfg.repulsive_gain

        for obs in self._obstacles:
            # Vector from obstacle surface to robot
            diff  = robot_pos - obs.position()
            dist  = float(np.linalg.norm(diff))
            d_eff = max(dist - obs.radius, 1e-4)  # distance to surface

            if d_eff < d0:
                # APF repulsive force magnitude
                magnitude = eta * (1.0 / d_eff - 1.0 / d0) / (d_eff ** 2)
                direction = diff / (dist + 1e-9)
                repulsive_force += magnitude * direction

        if np.linalg.norm(repulsive_force) < 1e-6:
            return base_cmd

        # Convert repulsive force to robot-frame (v, ω) adjustment
        cos_h = np.cos(robot.yaw)
        sin_h = np.sin(robot.yaw)
        f_x =  cos_h * repulsive_force[0] + sin_h * repulsive_force[1]  # forward
        f_y = -sin_h * repulsive_force[0] + cos_h * repulsive_force[1]  # lateral

        # Suppress forward speed near obstacles; add turning to steer away
        speed_factor = float(np.clip(1.0 - self._cfg.blend_alpha * np.linalg.norm(repulsive_force), 0.05, 1.0))
        avoid_angular = float(np.clip(f_y, -self._cfg.max_avoid_angular, self._cfg.max_avoid_angular))

        new_linear  = float(np.clip(base_cmd.linear  * speed_factor, 0.0, self._cfg.max_avoid_linear))
        new_angular = float(np.clip(base_cmd.angular + avoid_angular,
                                     -2.0, 2.0))

        return ControlCommand(linear=new_linear, angular=new_angular)

    def nearest_obstacle_distance(self, robot: RobotState) -> float:
        """Return distance to the nearest obstacle surface. Useful for safety checks."""
        if not self._obstacles:
            return float("inf")
        robot_pos = robot.position()
        dists = [np.linalg.norm(robot_pos - o.position()) - o.radius
                 for o in self._obstacles]
        return float(min(dists))
