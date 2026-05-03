"""
path_tracking.launch.py
=======================
Launch the trajectory tracker node with TurtleBot3 Gazebo simulation.

Usage:
    ros2 launch path_tracking path_tracking.launch.py use_sim_time:=true robot_model:=burger
"""

from launch                            import LaunchDescription
from launch.actions                    import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions                 import IfCondition
from launch.substitutions             import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions               import Node
from launch_ros.substitutions         import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os


def generate_launch_description():

    pkg_share = FindPackageShare("path_tracking")

    # ── Launch arguments ──────────────────────────────────────────
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time", default_value="true",
        description="Use simulated clock from Gazebo"
    )
    robot_model_arg = DeclareLaunchArgument(
        "robot_model", default_value="burger",
        description="TurtleBot3 model (burger | waffle | waffle_pi)"
    )
    enable_avoidance_arg = DeclareLaunchArgument(
        "enable_avoidance", default_value="true",
        description="Enable LiDAR-based obstacle avoidance"
    )

    use_sim_time     = LaunchConfiguration("use_sim_time")
    robot_model      = LaunchConfiguration("robot_model")
    enable_avoidance = LaunchConfiguration("enable_avoidance")

    # ── Trajectory tracker node ───────────────────────────────────
    tracker_node = Node(
        package    = "path_tracking",
        executable = "trajectory_tracker_node",
        name       = "trajectory_tracker",
        output     = "screen",
        parameters = [
            PathJoinSubstitution([pkg_share, "config", "params.yaml"]),
            {"use_sim_time":     use_sim_time},
            {"enable_avoidance": enable_avoidance},
        ],
        remappings = [
            ("/odom",        "/odom"),
            ("/cmd_vel",     "/cmd_vel"),
            ("/global_path", "/global_path"),
            ("/scan",        "/scan"),
        ],
    )

    # ── TurtleBot3 Gazebo simulation (optional — comment out for real robot) ──
    turtlebot3_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("turtlebot3_gazebo"),
                "launch",
                "turtlebot3_world.launch.py",
            ])
        ]),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    return LaunchDescription([
        use_sim_time_arg,
        robot_model_arg,
        enable_avoidance_arg,
        turtlebot3_gazebo,
        tracker_node,
    ])
