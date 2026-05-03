# Path Smoothing & Trajectory Control in 2D Space

**Assignment submission — ROS2 Python package for a differential drive robot.**

---

## Table of Contents

1. [Quick Start (Standalone — no ROS2 needed)](#1-quick-start-standalone)
2. [ROS2 Setup & Execution](#2-ros2-setup--execution)
3. [Package Architecture](#3-package-architecture)
4. [Algorithm Design Choices](#4-algorithm-design-choices)
5. [Extending to a Real Robot](#5-extending-to-a-real-robot)
6. [Obstacle Avoidance (Extra Credit)](#6-obstacle-avoidance-extra-credit)
7. [Testing](#7-testing)
8. [Results](#8-results)

---

## 1. Quick Start (Standalone)

No ROS2 installation required. The standalone simulation runs entirely on NumPy + SciPy + Matplotlib.

```bash
# Clone / unzip the submission
cd robotics_ws/src/path_tracking

# Install Python dependencies
pip install numpy scipy matplotlib

# Run the full simulation (generates simulation_results.png)
python scripts/standalone_simulation.py
```

This produces a 4-panel figure showing:
- ① Cubic spline smoothing vs raw waypoints
- ② Trapezoidal velocity profile
- ③ Robot tracking path (with and without obstacle avoidance)
- ④ Cross-track error over time

---

## 2. ROS2 Setup & Execution

### Prerequisites
- ROS2 Humble (or Jazzy)
- TurtleBot3 packages: `sudo apt install ros-humble-turtlebot3*`
- Python 3.10+

### Build

```bash
cd ~/robotics_ws
colcon build --packages-select path_tracking
source install/setup.bash
```

### Launch (Gazebo simulation)

```bash
export TURTLEBOT3_MODEL=burger
ros2 launch path_tracking path_tracking.launch.py use_sim_time:=true
```

### Send a path

```bash
# Publish a nav_msgs/Path on /global_path
# The node will auto-smooth it and start tracking
ros2 topic pub /global_path nav_msgs/msg/Path '{
  header: {frame_id: "map"},
  poses: [
    {pose: {position: {x: 0.0, y: 0.0}}},
    {pose: {position: {x: 1.0, y: 0.5}}},
    {pose: {position: {x: 2.0, y: 1.5}}},
    {pose: {position: {x: 3.0, y: 1.0}}},
    {pose: {position: {x: 4.0, y: 2.0}}}
  ]
}' --once
```

### Tune parameters

Edit `config/params.yaml` — no rebuild needed:

```bash
ros2 param set /trajectory_tracker max_speed 0.15
ros2 param set /trajectory_tracker enable_avoidance true
```

---

## 3. Package Architecture

```
path_tracking/
├── path_tracking/              # Core library (ROS2-agnostic)
│   ├── path_smoother.py        # Task 1: Cubic spline path smoothing
│   ├── trajectory_generator.py # Task 2: Trapezoidal velocity trajectory
│   ├── trajectory_controller.py# Task 3: Pure Pursuit controller
│   ├── obstacle_avoidance.py   # Extra: APF obstacle avoidance
│   └── simulator.py            # Differential drive kinematic simulator
├── nodes/
│   └── trajectory_tracker_node.py  # ROS2 node (odom, cmd_vel, /scan)
├── scripts/
│   └── standalone_simulation.py    # Run without ROS2
├── tests/
│   └── test_path_tracking.py       # 38 unit + integration tests
├── launch/
│   └── path_tracking.launch.py     # Launch with TurtleBot3 Gazebo
├── config/
│   └── params.yaml                 # Tunable ROS2 parameters
├── package.xml
├── setup.py
└── README.md
```

### Data flow

```
nav_msgs/Path  ──►  PathSmoother  ──►  TrajectoryGenerator  ──►  PurePursuitController
                        │                      │                          │
                  CubicSpline C²       trapezoidal v(t)          adaptive look-ahead
                                                                          │
LaserScan  ──►  ObstacleAvoider  ──────────────────────────►  cmd_vel (Twist)
                   APF repulsion
```

### Key design principle — separation of concerns

The core library (`path_tracking/`) has **zero ROS2 imports**. This means:
- It can be unit-tested without a ROS2 environment
- The same algorithms run in the standalone simulation and in the ROS2 node
- Easy to swap in different planners or controllers

---

## 4. Algorithm Design Choices

### 4.1 Path Smoothing — Cubic Spline (C² continuity)

**Why cubic splines?**
A cubic spline gives *C² continuity* (continuous position, velocity, and acceleration). This is the minimum required for a physically realistic robot trajectory — C¹ discontinuities in curvature cause sudden steering changes that saturate actuators.

**Chord-length parameterisation**
Naïve uniform parameter spacing produces oscillations (Runge's phenomenon) near clustered waypoints. Chord-length normalisation distributes parameter values proportionally to inter-waypoint distances, preventing this.

**'Not-a-knot' end condition**
Forces the third derivative to be continuous across the first and last interior knots. This avoids the "floppy end" artefact that free-end conditions introduce.

**Alternative considered**: B-splines allow *local* control (moving one waypoint only affects neighbouring segments). They were not chosen because they don't interpolate waypoints by default, requiring an extra solve step.

### 4.2 Trajectory Generation — Trapezoidal Velocity Profile

**Three-phase motion:** accelerate → cruise → decelerate.

- Respects actuator limits: bounded acceleration prevents wheel slip
- Guarantees the robot starts and stops at rest (smooth hand-off with global planner)
- Falls back to a *triangle profile* automatically when the path is too short to reach cruise speed
- **Arc-length re-parameterisation** ensures speed is constant in *metric* space, not just in spline-parameter space

**Alternative considered**: Minimum-time optimal control (bang-bang) — not chosen because it hits actuator limits continuously and is sensitive to model errors.

### 4.3 Trajectory Tracking — Pure Pursuit

Pure Pursuit computes the curvature κ of the arc connecting the robot to a look-ahead point L metres ahead:

```
κ = 2 · ld_y / L²
ω = v · κ
```

**Adaptive look-ahead** (`L = k · v`, clamped to [Lmin, Lmax]):
- At low speeds: small L → precise, slow tracking
- At high speeds: large L → smooth, less oscillatory

**Why not PID?**
A lateral PID controller needs careful gain tuning per trajectory shape. Pure Pursuit is geometry-based: it works on any smooth path without re-tuning, which fits the assignment's trajectory-following goal.

**Why not MPC?**
Model Predictive Control gives the best performance but requires a solver and is overkill for the assignment scope.

### 4.4 Simulator — Unicycle Kinematic Model

```
ẋ   = v cos(ψ)
ẏ   = v sin(ψ)
ψ̇   = ω
```

Euler integration with dt = 0.05 s. A RK4 integrator is easy to substitute but unnecessary at this time step for the speeds used.

---

## 5. Extending to a Real Robot

### 5.1 TurtleBot3 Burger

The ROS2 node (`trajectory_tracker_node.py`) is already structured for real hardware:

| Simulation | Real Robot | Change needed |
|---|---|---|
| `DifferentialDriveSimulator` | `/odom` topic | Subscribe to odometry ✓ (done) |
| Fake obstacles | `/scan` LaserScan | Subscribe to LiDAR ✓ (done) |
| Simulated clock | Hardware clock | `use_sim_time:=false` |

Steps for real deployment:
1. Install ROS2 on the TurtleBot3 on-board PC
2. Set `TURTLEBOT3_MODEL=burger`
3. `ros2 launch turtlebot3_bringup robot.launch.py`
4. `ros2 launch path_tracking path_tracking.launch.py use_sim_time:=false`

### 5.2 Calibration required on real hardware

| Issue | Solution |
|---|---|
| Wheel encoder drift | EKF localisation (`robot_localization` package) |
| LiDAR frame offset | Set correct `base_to_laser` TF transform |
| Speed command latency | Reduce `max_speed` to 0.15 m/s; increase `lookahead_min` |
| Slippery floors | Add slip compensation in the odometry model |

### 5.3 Replacing the global planner

The node subscribes to `nav_msgs/Path`. Any ROS2 global planner (Nav2 NavFn, SBPL, custom A*) publishes this message type — no code changes are needed.

---

## 6. Obstacle Avoidance (Extra Credit)

### Approach: Artificial Potential Field (APF)

An **attractive** component (Pure Pursuit, tracking the goal) is combined with a **repulsive** component (APF, pushing away from obstacles):

```
F_rep = η · (1/d − 1/d₀) / d²  · ∇d    if d < d₀
      = 0                                otherwise
```

The repulsive force is transformed into robot-frame angular velocity perturbation and blended with the Pure Pursuit command. Speed is also reduced proportionally to obstacle proximity.

**In simulation**: obstacles are defined as `Obstacle(x, y, radius)` objects.  
**In ROS2**: the `_scan_cb` callback converts each LiDAR range reading to an obstacle point in world coordinates, giving real-time dynamic obstacle avoidance.

### Limitation and extension

APF can get stuck in local minima (saddle points between two obstacles). For production use, combine with:
- **Dynamic Window Approach (DWA)**: samples feasible velocity commands and scores them
- **TEB (Timed Elastic Band)**: locally re-optimises the trajectory around obstacles
- **Nav2 integration**: drop in as the local costmap layer

---

## 7. Testing

```bash
cd robotics_ws/src/path_tracking
pytest tests/test_path_tracking.py -v
```

**38 tests, 0 failures** covering:

| Module | Tests |
|---|---|
| PathSmoother | Output shape, endpoint interpolation, arc length, derivative, edge cases |
| TrajectoryGenerator | Monotonic time, speed bounds, start/end at rest, triangle profile |
| PurePursuitController | Command types, bounds, goal reaching, integration test |
| ObstacleAvoider | Speed reduction near obstacles, no-effect far away, update obstacles |
| DifferentialDriveSimulator | Straight line, pure rotation, yaw wrapping, history |

---

---

## 9. Results

| Metric | Value |
|---|---|
| Algorithm | Cubic spline (C²) + Pure Pursuit |
| RMS cross-track error (no obstacles) | **1.47 cm** |
| RMS cross-track error (with APF avoidance) | **9.28 cm** |
| Total trajectory time (9 waypoints, 0.5 m/s) | ~19 s |
| Total path length | ~8.6 m |
| Tests | 38 / 38 passing |

The 9.28 cm error with obstacle avoidance is expected — the robot intentionally deviates from the reference path to avoid collisions.
