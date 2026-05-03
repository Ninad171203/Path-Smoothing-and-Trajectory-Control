"""
test_path_tracking.py
=====================
Unit + integration tests covering:
  - PathSmoother  (continuity, arc length, edge cases)
  - TrajectoryGenerator  (monotonic time, velocity profile bounds)
  - PurePursuitController  (command bounds, goal reaching)
  - ObstacleAvoider  (repulsion direction, speed reduction)
  - DifferentialDriveSimulator  (kinematics correctness)

Run with:
    pytest tests/test_path_tracking.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import math

from path_tracking.path_smoother         import PathSmoother
from path_tracking.trajectory_generator  import TrajectoryGenerator, TrajectoryConfig
from path_tracking.trajectory_controller import (
    PurePursuitController, ControllerConfig, RobotState, ControlCommand
)
from path_tracking.obstacle_avoidance    import ObstacleAvoider, Obstacle, APFConfig
from path_tracking.simulator             import DifferentialDriveSimulator


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

SIMPLE_WAYPOINTS = [(0, 0), (1, 1), (2, 0), (3, 1), (4, 0)]
STRAIGHT_WAYPOINTS = [(0, 0), (1, 0), (2, 0), (3, 0)]


@pytest.fixture
def smoother():
    return PathSmoother(SIMPLE_WAYPOINTS, num_samples=300)


@pytest.fixture
def straight_smoother():
    return PathSmoother(STRAIGHT_WAYPOINTS, num_samples=200)


@pytest.fixture
def traj_cfg():
    return TrajectoryConfig(max_speed=0.5, accel=0.3, dt=0.05)


@pytest.fixture
def generator(smoother, traj_cfg):
    return TrajectoryGenerator(smoother, config=traj_cfg)


@pytest.fixture
def controller(generator):
    return PurePursuitController(generator.get_trajectory(), ControllerConfig())


# ─────────────────────────────────────────────────────────────
# PathSmoother tests
# ─────────────────────────────────────────────────────────────

class TestPathSmoother:

    def test_output_shape(self, smoother):
        path = smoother.get_smooth_path()
        assert path.ndim == 2
        assert path.shape[1] == 2
        assert path.shape[0] == 300

    def test_passes_through_first_waypoint(self, smoother):
        """The spline must pass through (or very near) the first waypoint."""
        pt = smoother.evaluate(0.0)
        np.testing.assert_allclose(pt, SIMPLE_WAYPOINTS[0], atol=1e-6)

    def test_passes_through_last_waypoint(self, smoother):
        pt = smoother.evaluate(1.0)
        np.testing.assert_allclose(pt, SIMPLE_WAYPOINTS[-1], atol=1e-6)

    def test_arc_length_positive(self, smoother):
        L = smoother.total_arc_length()
        assert L > 0.0, "Arc length must be positive."

    def test_arc_length_straight_path(self, straight_smoother):
        """Arc length of a straight 3-metre path should be ≈ 3 m."""
        L = straight_smoother.total_arc_length()
        assert abs(L - 3.0) < 0.05, f"Expected ~3.0, got {L:.4f}"

    def test_derivative_shape(self, smoother):
        t_vals = np.linspace(0, 1, 50)
        d = smoother.evaluate_derivative(t_vals, 1)
        assert d.shape == (50, 2)

    def test_minimum_waypoints_error(self):
        with pytest.raises(ValueError):
            PathSmoother([(0, 0)])

    def test_coincident_waypoints_error(self):
        with pytest.raises(ValueError):
            PathSmoother([(1, 1), (1, 1), (1, 1)])

    def test_waypoints_property(self, smoother):
        wp = smoother.waypoints
        np.testing.assert_array_equal(wp, np.array(SIMPLE_WAYPOINTS, dtype=float))

    def test_t_knots_bounds(self, smoother):
        t = smoother.t_knots
        assert t[0] == pytest.approx(0.0)
        assert t[-1] == pytest.approx(1.0)

    def test_path_is_smooth(self, smoother):
        """Consecutive point distances should be roughly uniform (no jumps)."""
        path = smoother.get_smooth_path()
        diffs = np.diff(path, axis=0)
        dists = np.hypot(diffs[:, 0], diffs[:, 1])
        # No single segment should be more than 10× the mean
        assert dists.max() < 10 * dists.mean()


# ─────────────────────────────────────────────────────────────
# TrajectoryGenerator tests
# ─────────────────────────────────────────────────────────────

class TestTrajectoryGenerator:

    def test_trajectory_not_empty(self, generator):
        traj = generator.get_trajectory()
        assert len(traj) > 0

    def test_time_monotonically_increasing(self, generator):
        ts = np.array([p.t for p in generator.get_trajectory()])
        assert np.all(np.diff(ts) >= 0), "Time stamps must be non-decreasing."

    def test_speed_within_bounds(self, generator, traj_cfg):
        vs = np.array([p.v for p in generator.get_trajectory()])
        assert vs.min() >= -1e-9, "Speed must be non-negative."
        assert vs.max() <= traj_cfg.max_speed + 1e-6, "Speed exceeds max_speed."

    def test_starts_and_ends_at_zero_speed(self, generator):
        traj = generator.get_trajectory()
        assert traj[0].v  < 0.05, "Initial speed should be ~0."
        assert traj[-1].v < 0.05, "Final speed should be ~0."

    def test_as_tuples_format(self, generator):
        tuples = generator.as_tuples()
        assert isinstance(tuples, list)
        assert len(tuples[0]) == 3, "Each element must be (x, y, t)."

    def test_total_time_positive(self, generator):
        assert generator.total_time > 0

    def test_arrays_consistent_length(self, generator):
        xs, ys, ts, vs, yaws = generator.get_arrays()
        assert len(xs) == len(ys) == len(ts) == len(vs) == len(yaws)

    def test_start_position_matches_first_waypoint(self, smoother, traj_cfg):
        gen  = TrajectoryGenerator(smoother, traj_cfg)
        traj = gen.get_trajectory()
        np.testing.assert_allclose(
            [traj[0].x, traj[0].y], SIMPLE_WAYPOINTS[0], atol=0.05
        )

    def test_end_position_matches_last_waypoint(self, smoother, traj_cfg):
        gen  = TrajectoryGenerator(smoother, traj_cfg)
        traj = gen.get_trajectory()
        np.testing.assert_allclose(
            [traj[-1].x, traj[-1].y], SIMPLE_WAYPOINTS[-1], atol=0.1
        )

    def test_short_path_triangle_profile(self):
        """Short 2-waypoint path should use triangle profile (no cruise phase)."""
        short_wp = [(0, 0), (0.1, 0.0), (0.2, 0.0)]
        sm  = PathSmoother(short_wp)
        cfg = TrajectoryConfig(max_speed=2.0, accel=1.0, dt=0.01)
        gen = TrajectoryGenerator(sm, cfg)
        vs  = np.array([p.v for p in gen.get_trajectory()])
        assert vs.max() < 2.0, "Triangle profile should not reach cruise speed."


# ─────────────────────────────────────────────────────────────
# PurePursuitController tests
# ─────────────────────────────────────────────────────────────

class TestPurePursuitController:

    def test_command_type(self, controller):
        state = RobotState(0.0, 0.0, 0.0)
        cmd = controller.compute_command(state)
        assert isinstance(cmd, ControlCommand)

    def test_linear_velocity_non_negative(self, controller, generator):
        traj = generator.get_trajectory()
        robot = RobotState(traj[0].x, traj[0].y, traj[0].yaw)
        cmd = controller.compute_command(robot)
        assert cmd.linear >= 0.0

    def test_angular_velocity_within_bounds(self, controller, generator):
        cfg  = ControllerConfig()
        traj = generator.get_trajectory()
        robot = RobotState(traj[0].x, traj[0].y, traj[0].yaw)
        cmd = controller.compute_command(robot)
        assert abs(cmd.angular) <= cfg.max_angular_vel + 1e-6

    def test_goal_tolerance_stops_robot(self, controller, generator):
        """If robot is at the goal, command should be (0, 0)."""
        traj  = generator.get_trajectory()
        goal  = traj[-1]
        robot = RobotState(goal.x, goal.y, goal.yaw)
        # Force internal done flag by computing from goal position
        controller._last_idx = len(traj) - 1
        cmd = controller.compute_command(robot)
        assert controller.is_done or (cmd.linear == 0.0 and cmd.angular == 0.0)

    def test_reset(self, controller):
        controller.reset()
        assert not controller.is_done
        assert controller._last_idx == 0

    def test_wheel_velocities(self):
        cmd = ControlCommand(linear=0.3, angular=0.5)
        vl, vr = cmd.wheel_velocities(wheel_base=0.16)
        assert abs(vr - vl - 0.5 * 0.16) < 1e-9

    def test_integration_robot_reaches_goal(self, generator):
        """Full simulation: robot should reach within goal tolerance."""
        traj = generator.get_trajectory()
        ctrl = PurePursuitController(traj, ControllerConfig(goal_tolerance=0.15))
        sim  = DifferentialDriveSimulator(
            RobotState(traj[0].x, traj[0].y, traj[0].yaw), dt=0.05
        )
        t = 0.0
        max_t = 60.0
        while not ctrl.is_done and t < max_t:
            cmd = ctrl.compute_command(sim.state)
            sim.step(cmd)
            t += 0.05
        assert ctrl.is_done, f"Robot did not reach goal within {max_t}s."


# ─────────────────────────────────────────────────────────────
# ObstacleAvoider tests
# ─────────────────────────────────────────────────────────────

class TestObstacleAvoider:

    def setup_method(self):
        self.obs = [Obstacle(x=2.0, y=0.0, radius=0.2)]
        self.avoider = ObstacleAvoider(self.obs, APFConfig(influence_radius=1.0))

    def test_no_obstacles_passes_through(self):
        avoider = ObstacleAvoider([], APFConfig())
        base = ControlCommand(linear=0.4, angular=0.1)
        result = avoider.avoid(RobotState(0, 0, 0), base)
        assert result.linear  == base.linear
        assert result.angular == base.angular

    def test_speed_reduced_near_obstacle(self):
        base  = ControlCommand(linear=0.5, angular=0.0)
        robot = RobotState(x=1.5, y=0.0, yaw=0.0)  # 0.3m from obstacle surface
        cmd   = self.avoider.avoid(robot, base)
        assert cmd.linear <= base.linear, "Speed should be reduced near obstacle."

    def test_nearest_obstacle_distance(self):
        robot = RobotState(x=1.0, y=0.0, yaw=0.0)
        d = self.avoider.nearest_obstacle_distance(robot)
        expected = 2.0 - 1.0 - 0.2  # dist to centre - radius = 0.8
        assert abs(d - expected) < 0.01

    def test_update_obstacles(self):
        new_obs = [Obstacle(x=5.0, y=5.0, radius=0.1)]
        self.avoider.update_obstacles(new_obs)
        assert self.avoider._obstacles[0].x == 5.0

    def test_far_obstacle_no_effect(self):
        """Obstacle far outside influence radius should not change command."""
        base  = ControlCommand(linear=0.4, angular=0.2)
        robot = RobotState(x=10.0, y=10.0, yaw=0.0)  # 12 m from obstacle
        cmd   = self.avoider.avoid(robot, base)
        assert cmd.linear  == pytest.approx(base.linear,  abs=0.05)
        assert cmd.angular == pytest.approx(base.angular, abs=0.05)


# ─────────────────────────────────────────────────────────────
# DifferentialDriveSimulator tests
# ─────────────────────────────────────────────────────────────

class TestSimulator:

    def test_straight_line_motion(self):
        """Forward velocity with ω=0 should produce straight-line motion."""
        sim = DifferentialDriveSimulator(RobotState(0, 0, 0), dt=0.1)
        for _ in range(10):
            sim.step(ControlCommand(linear=1.0, angular=0.0))
        # After 1 second at 1 m/s heading 0, x ≈ 1.0, y ≈ 0
        s = sim.state
        assert abs(s.x - 1.0) < 0.01
        assert abs(s.y) < 0.01

    def test_pure_rotation(self):
        """Zero linear, positive ω should rotate heading."""
        sim = DifferentialDriveSimulator(RobotState(0, 0, 0), dt=0.1)
        for _ in range(10):
            sim.step(ControlCommand(linear=0.0, angular=1.0))
        # After 1 s at ω=1 rad/s, yaw ≈ 1.0 rad; position unchanged
        s = sim.state
        assert abs(s.yaw - 1.0) < 0.01
        assert abs(s.x) < 0.01
        assert abs(s.y) < 0.01

    def test_history_length(self):
        sim = DifferentialDriveSimulator(RobotState(0, 0, 0), dt=0.1)
        for _ in range(5):
            sim.step(ControlCommand())
        assert len(sim.history) == 6  # initial + 5 steps

    def test_tracking_error_exact(self):
        sim = DifferentialDriveSimulator(RobotState(3.0, 4.0, 0), dt=0.1)
        err = sim.tracking_error(0.0, 0.0)
        assert abs(err - 5.0) < 1e-6  # 3-4-5 triangle

    def test_yaw_wraps_around(self):
        sim = DifferentialDriveSimulator(RobotState(0, 0, 0), dt=1.0)
        # Step with ω such that yaw goes well past π
        sim.step(ControlCommand(linear=0.0, angular=4.0))
        assert -math.pi <= sim.state.yaw <= math.pi
