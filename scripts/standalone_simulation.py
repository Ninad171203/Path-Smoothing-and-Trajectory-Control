"""
standalone_simulation.py
========================
Self-contained demo that runs the full pipeline WITHOUT ROS2:

    Waypoints → Path Smoother → Trajectory Generator → Controller → Simulator

Produces a 4-panel figure:
  1. Path comparison (raw waypoints vs smoothed spline)
  2. Velocity profile over time
  3. Robot tracking performance (trajectory vs actual path)
  4. Cross-track error over time

Also runs the obstacle avoidance scenario.

Usage
-----
    python standalone_simulation.py

Dependencies: numpy, scipy, matplotlib (all in requirements.txt)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for file output
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

from path_tracking.path_smoother         import PathSmoother
from path_tracking.trajectory_generator  import TrajectoryGenerator, TrajectoryConfig
from path_tracking.trajectory_controller import (
    PurePursuitController, ControllerConfig, RobotState, ControlCommand
)
from path_tracking.obstacle_avoidance    import ObstacleAvoider, Obstacle, APFConfig
from path_tracking.simulator             import DifferentialDriveSimulator


# ============================================================
# Scenario definition
# ============================================================

WAYPOINTS = [
    (0.0, 0.0),
    (1.0, 0.5),
    (2.0, 1.5),
    (3.0, 1.0),
    (4.0, 2.0),
    (5.0, 1.5),
    (6.0, 3.0),
    (7.0, 2.5),
    (8.0, 3.5),
]

OBSTACLES = [
    Obstacle(x=2.5, y=1.2, radius=0.25),
    Obstacle(x=5.0, y=1.8, radius=0.20),
    Obstacle(x=6.5, y=2.8, radius=0.22),
]


# ============================================================
# Helper: run one simulation episode
# ============================================================

def run_simulation(waypoints, traj_cfg, ctrl_cfg, obstacles=None, noise_std=0.0):
    """
    Run the complete pipeline and return arrays of results.

    Returns
    -------
    smooth_path  : (N, 2)
    traj_xs, traj_ys, traj_ts, traj_vs : trajectory arrays
    robot_xs, robot_ys, errors         : simulation results
    times                              : simulation time stamps
    """
    # 1. Path smoothing
    smoother = PathSmoother(waypoints, num_samples=600)
    smooth_path = smoother.get_smooth_path()

    # 2. Trajectory generation
    gen = TrajectoryGenerator(smoother, config=traj_cfg)
    traj = gen.get_trajectory()
    traj_xs, traj_ys, traj_ts, traj_vs, _ = gen.get_arrays()

    # 3. Controller
    controller = PurePursuitController(traj, config=ctrl_cfg)

    # 4. Obstacle avoider
    avoider = ObstacleAvoider(obstacles or [], APFConfig()) if obstacles else None

    # 5. Simulate
    sim = DifferentialDriveSimulator(
        initial_state=RobotState(traj[0].x, traj[0].y, traj[0].yaw),
        dt=traj_cfg.dt,
        noise_std=noise_std,
    )

    robot_xs, robot_ys, errors, times = [], [], [], []
    t = 0.0

    while not controller.is_done and t < gen.total_time * 2.5:
        state = sim.state
        cmd   = controller.compute_command(state)
        if avoider:
            cmd = avoider.avoid(state, cmd)
        sim.step(cmd)

        # Nearest reference point for cross-track error
        dists = np.hypot(traj_xs - state.x, traj_ys - state.y)
        nearest_idx = int(np.argmin(dists))

        robot_xs.append(state.x)
        robot_ys.append(state.y)
        errors.append(float(dists[nearest_idx]))
        times.append(t)
        t += traj_cfg.dt

    return (smooth_path, traj_xs, traj_ys, traj_ts, traj_vs,
            np.array(robot_xs), np.array(robot_ys),
            np.array(errors), np.array(times))


# ============================================================
# Plot
# ============================================================

def make_figure(waypoints, obstacles):
    traj_cfg = TrajectoryConfig(max_speed=0.5, accel=0.3, dt=0.05)
    ctrl_cfg = ControllerConfig(lookahead_min=0.3, lookahead_max=1.5)

    wp_arr = np.array(waypoints)

    # --- Run without obstacle avoidance ---
    (smooth_path, traj_xs, traj_ys, traj_ts, traj_vs,
     robot_xs, robot_ys, errors, times) = run_simulation(
        waypoints, traj_cfg, ctrl_cfg, obstacles=None, noise_std=0.0)

    # --- Run WITH obstacle avoidance ---
    (_, _, _, _, _,
     avoid_robot_xs, avoid_robot_ys, avoid_errors, avoid_times) = run_simulation(
        waypoints, traj_cfg, ctrl_cfg, obstacles=obstacles, noise_std=0.0)

    # ---- Build figure ----
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#0f1117")
    gs = GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.35)

    ACCENT   = "#00c8ff"
    SMOOTH   = "#ff6b6b"
    TRACK    = "#ffd166"
    AVOID    = "#06d6a0"
    BG_AX    = "#1a1e2e"
    GRID_C   = "#2a2e3e"
    WP_C     = "#ffffff"
    OBS_C    = "#ff4444"
    TEXT_C   = "#e0e0f0"

    def style_ax(ax, title):
        ax.set_facecolor(BG_AX)
        ax.tick_params(colors=TEXT_C, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_C)
        ax.grid(True, color=GRID_C, linewidth=0.6, alpha=0.7)
        ax.set_title(title, color=TEXT_C, fontsize=11, fontweight="bold", pad=8)
        ax.xaxis.label.set_color(TEXT_C)
        ax.yaxis.label.set_color(TEXT_C)

    # ── Panel 1: Path Smoothing ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, "① Path Smoothing — Cubic Spline")
    ax1.plot(wp_arr[:, 0], wp_arr[:, 1], "o--",
             color=WP_C, markersize=8, linewidth=1.2,
             label="Raw waypoints", alpha=0.7, zorder=3)
    ax1.plot(smooth_path[:, 0], smooth_path[:, 1],
             color=SMOOTH, linewidth=2.5, label="Cubic spline (C²)", zorder=4)
    ax1.legend(facecolor=BG_AX, edgecolor=GRID_C, labelcolor=TEXT_C, fontsize=8)
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_aspect("equal", adjustable="box")

    # ── Panel 2: Velocity Profile ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, "② Trapezoidal Velocity Profile")
    ax2.fill_between(traj_ts, traj_vs, alpha=0.25, color=ACCENT)
    ax2.plot(traj_ts, traj_vs, color=ACCENT, linewidth=2.2, label="Speed [m/s]")
    ax2.axhline(traj_cfg.max_speed, color=SMOOTH, linewidth=1.0, linestyle="--",
                label=f"Max speed = {traj_cfg.max_speed} m/s")
    ax2.legend(facecolor=BG_AX, edgecolor=GRID_C, labelcolor=TEXT_C, fontsize=8)
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Speed [m/s]")
    ax2.set_ylim(bottom=0)

    # ── Panel 3: Tracking Performance ───────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :])
    style_ax(ax3, "③ Trajectory Tracking — Pure Pursuit Controller")
    ax3.plot(smooth_path[:, 0], smooth_path[:, 1],
             color=SMOOTH, linewidth=2.0, linestyle="--",
             label="Reference trajectory", zorder=3, alpha=0.8)
    ax3.plot(robot_xs, robot_ys,
             color=TRACK, linewidth=2.0, label="Robot path (no obstacles)", zorder=4)
    ax3.plot(avoid_robot_xs, avoid_robot_ys,
             color=AVOID, linewidth=2.0,
             label="Robot path (with obstacle avoidance)", zorder=4, alpha=0.85)
    # Waypoints
    ax3.scatter(wp_arr[:, 0], wp_arr[:, 1], color=WP_C, s=60, zorder=6,
                label="Waypoints", edgecolors=BG_AX, linewidths=0.8)
    # Start / Goal markers
    ax3.annotate("START", (wp_arr[0, 0], wp_arr[0, 1]),
                 textcoords="offset points", xytext=(6, 8),
                 color=TEXT_C, fontsize=8, fontweight="bold")
    ax3.annotate("GOAL", (wp_arr[-1, 0], wp_arr[-1, 1]),
                 textcoords="offset points", xytext=(6, -14),
                 color=TEXT_C, fontsize=8, fontweight="bold")
    # Obstacles
    for obs in obstacles:
        circ = patches.Circle((obs.x, obs.y), obs.radius,
                               color=OBS_C, alpha=0.35, zorder=5)
        edge = patches.Circle((obs.x, obs.y), obs.radius,
                               fill=False, edgecolor=OBS_C,
                               linewidth=1.8, zorder=5)
        ax3.add_patch(circ)
        ax3.add_patch(edge)
        ax3.text(obs.x, obs.y + obs.radius + 0.08, "Obstacle",
                 color=OBS_C, fontsize=7, ha="center", va="bottom")

    ax3.legend(facecolor=BG_AX, edgecolor=GRID_C, labelcolor=TEXT_C, fontsize=9)
    ax3.set_xlabel("x [m]")
    ax3.set_ylabel("y [m]")
    ax3.set_aspect("equal", adjustable="box")

    # ── Panel 4: Cross-Track Error ───────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    style_ax(ax4, "④ Cross-Track Error over Time")
    ax4.plot(times, errors * 100,
             color=TRACK, linewidth=1.8, label="Error (no obstacles) [cm]")
    ax4.plot(avoid_times, avoid_errors * 100,
             color=AVOID, linewidth=1.8, alpha=0.85,
             label="Error (with obstacle avoidance) [cm]")
    rms_err  = float(np.sqrt(np.mean(errors ** 2)))
    rms_err2 = float(np.sqrt(np.mean(avoid_errors ** 2)))
    ax4.axhline(rms_err * 100,  color=TRACK, linestyle=":", linewidth=1.2,
                label=f"RMS = {rms_err*100:.1f} cm")
    ax4.axhline(rms_err2 * 100, color=AVOID, linestyle=":", linewidth=1.2,
                label=f"RMS (avoid) = {rms_err2*100:.1f} cm")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Cross-track error [cm]")
    ax4.set_ylim(bottom=0)
    ax4.legend(facecolor=BG_AX, edgecolor=GRID_C, labelcolor=TEXT_C, fontsize=9)

    # ── Super title ──────────────────────────────────────────────────────
    fig.suptitle(
        "Path Smoothing & Trajectory Control — Differential Drive Robot",
        color=TEXT_C, fontsize=15, fontweight="bold", y=0.98
    )

    out_path = "./simulation_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"[✓] Figure saved → {out_path}")
    print(f"    RMS cross-track error (no obstacles)      : {rms_err*100:.2f} cm")
    print(f"    RMS cross-track error (obstacle avoidance): {rms_err2*100:.2f} cm")
    return out_path


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Path Smoothing & Trajectory Control — Simulation")
    print("=" * 60)
    make_figure(WAYPOINTS, OBSTACLES)
    print("Done.")
