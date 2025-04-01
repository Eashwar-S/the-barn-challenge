#!/usr/bin/env python3
import os
import rospy
import rospkg
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg for interactive plotting
import matplotlib.pyplot as plt

from env import Robot
from utils import load_config
from predictive_dwa_planner import DWA, Config  # Assuming your DWA planner is in dwa_planner.py
from sklearn.cluster import DBSCAN
from filterpy.kalman import KalmanFilter

CONTROL_FREQ = 20  # Hz. Change the control frequency here
OBSTACLE_TYPE_DETERMINATION_STEPS = 2  # Number of steps to compare laser scans
OBSTACLE_MOVEMENT_THRESHOLD = 0.01  # Meters, threshold to detect movement
NUM_LASER_SAMPLES = 300
PLOT_FIGURE = False

def process_laser_scan(odom, laser):
    """Convert laser scan to global obstacle positions."""
    obstacles = []
    angle_min = -2.35619449  # -135 degrees in radians
    angle_max = 2.35619449   # 135 degrees in radians
    sampling_factor = len(laser) // NUM_LASER_SAMPLES
    sampled_laser = laser[::sampling_factor]
    num_measurements = len(sampled_laser)
    angle_increment = (angle_max - angle_min) / (num_measurements - 1)

    # Initialize sectors (9 sectors, each 30° wide)
    num_sectors = 9
    sector_width = (angle_max - angle_min) / num_sectors
    sector_min_distances = [float('inf')] * num_sectors

    for i, d in enumerate(sampled_laser):
        if np.isinf(d) or np.isnan(d):
            continue
        angle = angle_min + i * angle_increment
        obs_robot = d * np.array([np.cos(angle), np.sin(angle)])
        x_obs = odom[0] + obs_robot[0] * np.cos(odom[2]) - obs_robot[1] * np.sin(odom[2])
        y_obs = odom[1] + obs_robot[0] * np.sin(odom[2]) + obs_robot[1] * np.cos(odom[2])
        obstacles.append(np.array([x_obs, y_obs]))

        # Assign to sector and update minimum distance
        sector_index = min(int((angle - angle_min) / sector_width), num_sectors - 1)
        sector_min_distances[sector_index] = min(sector_min_distances[sector_index], d)

    return obstacles, sector_min_distances

def check_obstacle_movement(prev_obs, current_obs):
    """Check if obstacles moved beyond the threshold."""
    if len(prev_obs) != len(current_obs):
        return True  # Different number of obstacles implies movement
    distances = np.linalg.norm(np.array(prev_obs, dtype=np.float32) - np.array(current_obs, dtype=np.float32), axis=1)

    return np.max(distances) > OBSTACLE_MOVEMENT_THRESHOLD

def determine_obstacle_type(robot, steps=OBSTACLE_TYPE_DETERMINATION_STEPS):
    """Determine if obstacles are static or dynamic by comparing laser scans."""
    prev_obstacles = None
    i = 0
    while i < steps:
        laser = robot.laser
        if laser is None:
            rospy.sleep(0.1)
        else:
            obstacles, _ = process_laser_scan(robot.odom, laser)
            if prev_obstacles is not None:
                if check_obstacle_movement(prev_obstacles, obstacles):
                    return "dynamic"
            prev_obstacles = obstacles
            i += 1
            rospy.sleep(1.0 / CONTROL_FREQ)
    return "static"

def get_local_goal_from_path(path, current_pos, los):
    """Select a point on the path approximately 'los' distance along the path."""
    if len(path) == 0:
        return current_pos
    cumulative_dist = 0.0
    for i in range(1, len(path)):
        segment_dist = np.linalg.norm(path[i] - path[i-1])
        cumulative_dist += segment_dist
        if cumulative_dist > los:
            return path[i]
    return path[-1]  # Fallback to last point

def centroid_code(current_obs, dwa):
    obs_array = np.array(current_obs)
    if len(obs_array) == 0:
        return None

    centroids = []
    # Cluster points with DBSCAN
    clustering = DBSCAN(eps=0.6, min_samples=3).fit(obs_array)
    labels = clustering.labels_
    unique_labels = set(labels) - {-1}  # Exclude noise

    clustered_obs = {}
    for label in unique_labels:
        cluster_points = obs_array[labels == label]
        if len(cluster_points) >= 3:  # Need at least 3 points for circle fitting
            center_x, center_y, r = dwa.fit_circle_to_points(cluster_points)
            if r <= 0.6:
                clustered_obs[label] = [center_x, center_y]
                centroids.append(np.array([center_x, center_y]))
    return np.array(centroids)

def track_obstacle_trajectories(tracked_obstacles):
    """Predict future trajectories of tracked obstacles.
    
    Returns:
        List of predicted trajectories, each an array of [x, y] positions
    """
    num_steps = 11
    times = np.arange(1, num_steps + 1) * 0.05
    predicted = []
    for tracked in tracked_obstacles.values():
        traj = []
        # kf_copy = tracked.kf.copy()  # Avoid modifying the actual filter
        kf_copy = KalmanFilter(dim_x=4, dim_z=2)  # Match the original filter's dimensions
        kf_copy.x = tracked.kf.x.copy()           # State vector
        kf_copy.P = tracked.kf.P.copy()           # Covariance matrix
        kf_copy.F = tracked.kf.F.copy()           # State transition matrix
        kf_copy.H = tracked.kf.H.copy()           # Measurement function
        kf_copy.R = tracked.kf.R.copy()           # Measurement noise
        kf_copy.Q = tracked.kf.Q.copy()           # Process noise
        for t in times:
            kf_copy.F = np.array([[1, 0, 0.05, 0],
                                [0, 1, 0, 0.05],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
            kf_copy.predict()
            traj.append(kf_copy.x[:2].copy())
        predicted.append(np.array(traj))
    return predicted

if __name__ == '__main__':
    config = load_config()
    rospack = rospkg.RosPack()

    print('Initializing robot...')
    robot = Robot(**config['robot'])
    print('Robot initialized')

    # Get goal parameters from ROS parameters
    goal_x = rospy.get_param('~goal_x', 0)
    goal_y = rospy.get_param('~goal_y', 10)
    goal_psi = rospy.get_param('~goal_psi', 1.57)
    robot.set_goal(goal_x, goal_y, goal_psi)
    final_goal = np.array([goal_x, goal_y, goal_psi])

    rate = rospy.Rate(CONTROL_FREQ)

    for _ in range(2):
        _ = determine_obstacle_type(robot)
        rate.sleep()
    # Determine obstacle type
    obstacle_type = determine_obstacle_type(robot)

    # Initialize the DWA planner
    dwa_planner_config = Config()
    dwa = DWA(dwa_planner_config)  

    # Initialize Matplotlib Plot
    if PLOT_FIGURE:
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 8))
        global_path_plot, = ax.plot([], [], 'g-', label='Global Path')
        robot_pos_plot, = ax.plot([], [], 'bo', label='Robot Position')
        goal_plot = ax.plot(final_goal[0], final_goal[1], 'rx', label='Final Goal')

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_title("Robot Navigation with DWA")
        ax.grid(True)
        ax.legend()

    print('Starting control loop...')
    v_prev, w_prev = 0.0, 0.0
    while not rospy.is_shutdown():
        # Get the current robot state from odometry ([x, y, psi, v, w])
        current_state = list(robot.odom)
        current_state.append(v_prev)
        current_state.append(w_prev)
        current_state = np.array(current_state, dtype=np.float32)

        local_goal_pos = get_local_goal_from_path(robot.original_global_path, current_state[:2], robot.los)
        yaw_to_goal = np.arctan2(local_goal_pos[1] - current_state[1], local_goal_pos[0] - current_state[0])
        local_goal = np.array([local_goal_pos[0], local_goal_pos[1], yaw_to_goal], np.float32)

        # Process the laser scan to extract obstacle positions in the global frame
        obstacles, sector_min_distances  = process_laser_scan(current_state, robot.laser)
        
        if obstacle_type == 'dynamic':
            u, traj = dwa.plan(current_state, final_goal, obstacles, sector_min_distances, obstacle_type)  # For dynamic obstacles

        else:
            u, traj = dwa.plan(current_state, local_goal, obstacles, sector_min_distances, obstacle_type)  # For static obstacles
         
        v, w = u

        # Set the velocity commands to the robot
        robot.set_velocity(v, w)
        v_prev = v
        w_prev = w

        # Update Plot
        if PLOT_FIGURE:
            ax.clear()

            # Plot global path (optional, since DWA navigates directly to the goal)
            if obstacle_type == 'static':
                if robot.global_path.shape[0] > 0:
                    ax.scatter(robot.original_global_path[:, 0], robot.original_global_path[:, 1], c='green', label='Global Path')
                else:
                    print('Global path generation failed')

            if obstacle_type == 'dynamic':
                centroids = centroid_code(obstacles, dwa)
                if centroids.size > 0:
                    ax.scatter(centroids[:, 0], centroids[:, 1], s=5, c="gray", label='Obstacles')
                
                predicted_trajectories = track_obstacle_trajectories(dwa.tracked_obstacles)
                
                for obs_traj in predicted_trajectories:
                    # Each traj is a numpy array of shape (num_steps, 2)
                    ax.plot(obs_traj[:, 0], obs_traj[:, 1], marker='o', linestyle='-', color='orange', label='Predicted Trajectory')


                ax.plot(traj[:, 0], traj[:, 1], "-g", label='Robot trajectory')
                if dwa.global_path is not None:
                    ax.scatter(dwa.global_path[:, 0], dwa.global_path[:, 1], c='blue', label='Global Path')
                # else:
                #     print('RRT global path generation failed')
            # Plot robot position
            ax.plot(current_state[0], current_state[1], 'bo', label='Robot Position')

            # Plot final goal
            ax.plot(final_goal[0], final_goal[1], 'rx', label='Final Goal')

            # Plot obstacles
            if obstacles:
                obs_array = np.array(obstacles)
                ax.scatter(obs_array[:, 0], obs_array[:, 1], s=2, c="black", label='Obstacles')

            ax.scatter(local_goal[0], local_goal[1], s=5, c="red", label='local goal')
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_title("Robot Navigation with DWA")
            ax.grid(True)
            ax.legend()

        plt.pause(0.001)
        print(f'Control inputs - v: {v:.2f}, w: {w:.2f}')

        rate.sleep()