"""
simulator.py
============
Lightweight kinematic simulator for a differential drive robot.

Uses the unicycle model:
    ẋ   = v · cos(ψ)
    ẏ   = v · sin(ψ)
    ψ̇   = ω

Euler integration (step size = dt) is sufficient for simulation purposes.
On a real robot these equations are replaced by the actual odometry topic.
"""

import numpy as np
from typing import List
from path_tracking.trajectory_controller import RobotState, ControlCommand


class DifferentialDriveSimulator:
    """
    Simulates a differential drive robot's motion under velocity commands.

    Parameters
    ----------
    initial_state : RobotState
    dt            : float — integration step [s]
    noise_std     : float — Gaussian noise std on v and ω (0 = perfect)
    """

    def __init__(self, initial_state: RobotState,
                 dt: float = 0.05,
                 noise_std: float = 0.0):
        self._state = RobotState(initial_state.x, initial_state.y, initial_state.yaw)
        self._dt = dt
        self._noise_std = noise_std
        self._history: List[RobotState] = [RobotState(self._state.x,
                                                        self._state.y,
                                                        self._state.yaw)]

    def step(self, cmd: ControlCommand) -> RobotState:
        """
        Advance the simulator by one time step.

        Parameters
        ----------
        cmd : ControlCommand — (v, ω) from the controller

        Returns
        -------
        RobotState — updated pose after dt seconds
        """
        v = cmd.linear
        w = cmd.angular

        # Optional sensor / actuator noise
        if self._noise_std > 0:
            v += np.random.normal(0, self._noise_std)
            w += np.random.normal(0, self._noise_std * 2)

        dt = self._dt
        yaw = self._state.yaw

        self._state.x   += v * np.cos(yaw) * dt
        self._state.y   += v * np.sin(yaw) * dt
        self._state.yaw += w * dt
        # Keep yaw in [-π, π]
        self._state.yaw = float((self._state.yaw + np.pi) % (2 * np.pi) - np.pi)

        self._history.append(RobotState(self._state.x,
                                         self._state.y,
                                         self._state.yaw))
        return RobotState(self._state.x, self._state.y, self._state.yaw)

    @property
    def state(self) -> RobotState:
        return self._state

    @property
    def history(self) -> List[RobotState]:
        return list(self._history)

    def tracking_error(self, ref_x: float, ref_y: float) -> float:
        """Euclidean distance from current state to reference point."""
        return float(np.hypot(self._state.x - ref_x, self._state.y - ref_y))
