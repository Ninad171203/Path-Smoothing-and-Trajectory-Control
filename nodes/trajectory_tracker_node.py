#!/usr/bin/env python3
"""
trajectory_tracker_node.py
===========================
ROS2 node that:
  1. Receives a nav_msgs/Path on /global_path
  2. Smooths it with PathSmoother
  3. Generates a TrajectoryGenerator trajectory
  4. Subscribes to /odom for the robot pose
  5. Publishes geometry_msgs/Twist on /cmd_vel via PurePursuitController
  6. Optionally subscribes to /scan for obstacle avoidance
"""

import rclpy
from rclpy.node import Node

import numpy as np
import math

from geometry_msgs.msg import Twist
from nav_msgs.msg      import Odometry, Path
from sensor_msgs.msg   import LaserScan

from path_tracking.path_smoother         import PathSmoother
from path_tracking.trajectory_generator  import TrajectoryGenerator, TrajectoryConfig
from path_tracking.trajectory_controller import (
    PurePursuitController, ControllerConfig, RobotState
)
from path_tracking.obstacle_avoidance    import ObstacleAvoider, Obstacle, APFConfig


def quat_to_yaw(q) -> float:
    """Convert geometry_msgs quaternion to yaw angle [rad]."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class TrajectoryTrackerNode(Node):

    def __init__(self):
        super().__init__("trajectory_tracker")

        # ── Parameters ────────────────────────────────────────────
        self.declare_parameter("max_speed",         0.22)    # TurtleBot3 Burger limit
        self.declare_parameter("accel",             0.3)
        self.declare_parameter("dt",                0.05)
        self.declare_parameter("lookahead_min",     0.3)
        self.declare_parameter("lookahead_max",     1.0)
        self.declare_parameter("lookahead_k",       0.5)
        self.declare_parameter("goal_tolerance",    0.1)
        self.declare_parameter("enable_avoidance",  True)
        self.declare_parameter("influence_radius",  0.6)

        max_speed  = self.get_parameter("max_speed").value
        accel      = self.get_parameter("accel").value
        dt         = self.get_parameter("dt").value

        self._traj_cfg = TrajectoryConfig(max_speed=max_speed, accel=accel, dt=dt)
        self._ctrl_cfg = ControllerConfig(
            lookahead_min   = self.get_parameter("lookahead_min").value,
            lookahead_max   = self.get_parameter("lookahead_max").value,
            lookahead_k     = self.get_parameter("lookahead_k").value,
            goal_tolerance  = self.get_parameter("goal_tolerance").value,
            max_linear_vel  = max_speed,
        )
        self._enable_avoidance = self.get_parameter("enable_avoidance").value
        self._apf_cfg = APFConfig(
            influence_radius = self.get_parameter("influence_radius").value
        )

        # ── State ─────────────────────────────────────────────────
        self._robot_state   = RobotState()
        self._controller    = None
        self._avoider       = ObstacleAvoider([], self._apf_cfg)
        self._active        = False

        # ── Publishers ────────────────────────────────────────────
        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # ── Subscribers ───────────────────────────────────────────
        self.create_subscription(Path,     "/global_path", self._path_cb,  10)
        self.create_subscription(Odometry, "/odom",        self._odom_cb,  10)
        if self._enable_avoidance:
            self.create_subscription(LaserScan, "/scan",   self._scan_cb,  10)

        # ── Control loop timer ────────────────────────────────────
        self.create_timer(dt, self._control_loop)

        self.get_logger().info("TrajectoryTrackerNode ready. Waiting for /global_path …")

    # ──────────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────────

    def _path_cb(self, msg: Path) -> None:
        """Receive a nav_msgs/Path and initialise the smooth trajectory."""
        if len(msg.poses) < 2:
            self.get_logger().warn("Received path with < 2 poses — ignored.")
            return

        waypoints = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.get_logger().info(f"New path received: {len(waypoints)} waypoints.")

        try:
            smoother    = PathSmoother(waypoints)
            generator   = TrajectoryGenerator(smoother, self._traj_cfg)
            traj        = generator.get_trajectory()
            self._controller = PurePursuitController(traj, self._ctrl_cfg)
            self._active = True
            self.get_logger().info(
                f"Trajectory ready: {len(traj)} points, "
                f"{generator.total_time:.1f} s, "
                f"{generator.total_length:.2f} m."
            )
        except Exception as exc:
            self.get_logger().error(f"Failed to build trajectory: {exc}")

    def _odom_cb(self, msg: Odometry) -> None:
        """Update robot pose from odometry."""
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        self._robot_state = RobotState(
            x   = pos.x,
            y   = pos.y,
            yaw = quat_to_yaw(ori),
        )

    def _scan_cb(self, msg: LaserScan) -> None:
        """
        Convert LaserScan to a list of Obstacle objects and update the avoider.
        Only ranges within influence_radius are kept to limit computation.
        """
        if not self._enable_avoidance:
            return

        obstacles = []
        angle     = msg.angle_min
        r_max     = self._apf_cfg.influence_radius * 1.5
        rx, ry, ryaw = (self._robot_state.x,
                        self._robot_state.y,
                        self._robot_state.yaw)

        for r in msg.ranges:
            if msg.range_min < r < min(r_max, msg.range_max):
                ox = rx + r * math.cos(ryaw + angle)
                oy = ry + r * math.sin(ryaw + angle)
                obstacles.append(Obstacle(x=ox, y=oy, radius=0.05))
            angle += msg.angle_increment

        self._avoider.update_obstacles(obstacles)

    # ──────────────────────────────────────────────────────────────
    # Control loop
    # ──────────────────────────────────────────────────────────────

    def _control_loop(self) -> None:
        if not self._active or self._controller is None:
            return

        if self._controller.is_done:
            self._publish(0.0, 0.0)
            self._active = False
            self.get_logger().info("Goal reached! Stopping.")
            return

        cmd = self._controller.compute_command(self._robot_state)

        if self._enable_avoidance:
            cmd = self._avoider.avoid(self._robot_state, cmd)

        self._publish(cmd.linear, cmd.angular)

    def _publish(self, linear: float, angular: float) -> None:
        twist = Twist()
        twist.linear.x  = float(linear)
        twist.angular.z = float(angular)
        self._cmd_pub.publish(twist)


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
