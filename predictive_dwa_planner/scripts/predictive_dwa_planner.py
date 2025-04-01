import numpy as np
import random
from sklearn.cluster import DBSCAN
from scipy.interpolate import splprep, splev
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

class Config:
    def __init__(self):
        # Robot Configuration Parameters
        self.max_vel_x = 2.0  # maximum forward velocity [m/s]
        self.min_vel_x = -2.0  # minimum forward velocity [m/s]
        self.max_vel_theta = 1.57  # maximum angular velocity [rad/s]
        self.min_vel_theta = -1.57  # minimum angular velocity [rad/s]
        self.min_in_place_vel_theta = 0.314  # minimum in-place angular velocity [rad/s]
        self.acc_lim_x = 10.0  # maximum forward acceleration [m/s^2]
        self.acc_lim_theta = 20.0  # maximum angular acceleration [rad/s^2]
        
        # Goal Tolerance Parameters
        self.xy_goal_tolerance = 0.25  # distance to goal considered as reached [m]
        self.yaw_goal_tolerance = 0.157  # angle to goal considered as reached [rad]
        
        
        # Trajectory Scoring Parameters
        self.pdist_scale = 0.75  # weighting for how much to follow the global path
        
        # Other parameters
        self.controller_frequency = 1.0  # frequency of the control loop [Hz]
        self.robot_radius = 0.3  # robot radius [m]
        self.obstacle_radius = 0.5  # Obstacle radius [m]
        self.to_goal_cost_gain = 1.0
        self.dt = 0.05

        # Hyperparameters to tune
        self.danger_distance = 5.0
        self.frontal_danger_angle = np.pi/2    # 10 degrees 
        self.tracking_max_dist = 0.4  # Max distance for track association (m)
        self.tracking_max_age = 0.1    # Time to keep unmatched tracks (s)
        self.reverse_turn_rate = 0.01
        self.reverse_speed = 2.0
        self.max_reverse_duration = 0.2
        self.cost_to_goal_penalty = 80.0
        self.obstacle_cost_gain = 90.0
        self.orientation_penalty = 15.0
        self.safety_weight = 15.0

        # Parameters for static obstacles
        self.max_vel_x_static = 1.0  # Lower velocity for careful maneuvering
        self.vx_samples_static = 10  # More samples for finer control
        self.vtheta_samples_static = 15
        self.cost_to_goal_penalty_static = 30.0
        self.obstacle_cost_gain_static = 80.0
        self.orientation_penalty_static = 5.0
        self.safety_weight_static = 1.5

"""
  distance based tracking - unable to keep up with obstacles
"""
# class TrackedObstacle:
#     def __init__(self, pos, timestamp, obstacle_id):
#         self.id = obstacle_id
#         self.positions = [pos]  # [(x1, y1), (x2, y2), ...]
#         self.timestamps = [timestamp]
#         self.velocity = (0.0, 0.0)  # (vx, vy)
    
#     def update(self, new_pos, new_timestamp):
#         self.positions.append(new_pos)
#         self.timestamps.append(new_timestamp)
        
#         # Update velocity if we have at least 2 observations
#         if len(self.positions) >= 2:
#             dt = new_timestamp - self.timestamps[-2]
#             dx = new_pos[0] - self.positions[-2][0]
#             dy = new_pos[1] - self.positions[-2][1]
#             if dt > 0:
#                 self.velocity = (dx/dt, dy/dt)
#             else:
#                 self.velocity = (0.0, 0.0)


class TrackedObstacle:
    def __init__(self, initial_pos, timestamp, obstacle_id, dt):
        """Initialize a tracked obstacle with a Kalman filter.
        
        Args:
            initial_pos: [x, y] initial position from measurement
            timestamp: Initial time
            obstacle_id: Unique identifier
            dt: Time step for initialization (updated later)
        """
        self.id = obstacle_id
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        # Initial state: [x, y, vx, vy]
        self.kf.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0])
        # Initial covariance: low position uncertainty, high velocity uncertainty
        self.kf.P = np.diag([0.1, 0.1, 10.0, 10.0])
        # State transition matrix (updated with dt later)
        self.kf.F = np.array([[1, 0, dt, 0],
                              [0, 1, 0, dt],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        # Measurement matrix
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])
        # Measurement noise: 5 cm std dev
        self.kf.R = np.diag([1.0, 1.0])
        # Process noise: tuned for moderate acceleration
        self.kf.Q = np.diag([0.01, 0.01, 0.1, 0.1])
        self.timestamp = timestamp
        self.positions = [initial_pos]  # List of [x, y]
        self.timestamps = [timestamp]

    def predict(self, dt):
        """Predict the next state using the Kalman filter.
        
        Args:
            dt: Time step since last update
        """
        # Update F with current dt
        self.kf.F = np.array([[1, 0, dt, 0],
                              [0, 1, 0, dt],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        self.kf.predict()

    def update(self, measurement, timestamp):
        """Update the state with a new measurement.
        
        Args:
            measurement: [x, y] from laser scan
            timestamp: Current time
        """
        self.kf.update(measurement)
        self.positions.append(self.kf.x[:2].tolist())
        self.timestamps.append(timestamp)
        self.timestamp = timestamp

    @property
    def velocity(self):
        """Return current velocity estimate."""
        return self.kf.x[2:4]

class DWA:
    def __init__(self, config):
        self.config = config
        self.reverse_mode = False
        self.reverse_turn_direction = 0  # 0: not turning, 1: left, -1: right
        self.tracked_obstacles = {}  # {id: TrackedObstacle}
        self.next_obstacle_id = 0
        self.current_time = 0.0
        self.reverse_duration = 0.0  # Initialized
        self.sim_time = 2.0  # simulation time for trajectory evaluation [s]
        self.sim_time_static = 2.0
        self.sim_granularity = 0.5  # step size for trajectory simulation [s]
        self.vx_samples = 10 #6  # number of velocity samples in x direction
        self.vtheta_samples = 15 # 20  # number of angular velocity samples
        self.R = 3.0    # radius to consider obstacles [m]
        self.horizon = 10.0
        self.percent_speed_reduction = 0.25
        self.traj_collision_distance = 5.0
        self.goal_distance = 3.0
        self.direction = None
        self.global_path = None
        self.last_replan_time = 0.0
        self.replan_interval = 1.0  # Replan every 0.5 seconds
        self.current_pos = None
        self.local_goal = None
        self.angle_min = -2.35619449  # -135 degrees
        self.num_sectors = 9
        self.sector_width = (2.35619449 - self.angle_min) / self.num_sectors
        self.obstacle_count_weight = 1.0  # New cost weight for current obstacles
        self.future_obstacle_count_weight = 1.5  # New cost weight for future obstacles

    def plan(self, x, goal, ob, sector_min_distances, sector_obstacle_counts, sector_future_obstacle_counts, obstacle_type):
        self.current_pos = x
        """Branch to static or dynamic planning based on obstacle type."""
        if obstacle_type == "static":
            return self.plan_static(x, goal, ob)
        else:
            return self.plan_dynamic(x, goal, ob, sector_min_distances, sector_obstacle_counts, sector_future_obstacle_counts)

    """
    All function from now onwards are for DWA planner to deal
    with static obstacles
    """

    def plan_static(self, x, goal, ob):
        """Plan for static obstacles with lower velocity and no reverse."""
        dw = self.calc_dynamic_window_static(x)
        u, trajectory = self.calc_control_and_trajectory_static(x, dw, goal, ob)
        return u, trajectory

    def calc_dynamic_window_static(self, x):
        """Calculate dynamic window for static obstacles with lower velocity."""
        vs = [-0.5, self.config.max_vel_x_static,
              self.config.min_vel_theta, self.config.max_vel_theta]
        vd = [x[3] - self.config.acc_lim_x * self.sim_granularity,
              x[3] + self.config.acc_lim_x * self.sim_granularity,
              x[4] - self.config.acc_lim_theta * self.sim_granularity,
              x[4] + self.config.acc_lim_theta * self.sim_granularity]
        dw = [max(vs[0], vd[0]), min(vs[1], vd[1]),
              max(vs[2], vd[2]), min(vs[3], vd[3])]
        return dw
    
    def calc_control_and_trajectory_static(self, x, dw, goal, ob):
        """Calculate control inputs and trajectory for static obstacles."""
        x_init = x.copy()
        min_cost = float('inf')
        best_u = [0.0, 0.0]
        best_trajectory = np.array([x])

        for v in np.linspace(dw[0], dw[1], self.config.vx_samples_static):
            for w in np.linspace(dw[2], dw[3], self.config.vtheta_samples_static):
                trajectory = self.predict_trajectory_static(x_init, v, w)
                cost = self.calc_cost_static(trajectory, goal, ob)
                if cost < min_cost:
                    min_cost = cost
                    best_u = [v, w]
                    best_trajectory = trajectory
        if min_cost == float('inf'):
            best_u = [0.0, 0.0]
        return best_u, best_trajectory
    
    def predict_trajectory_static(self, x_init, v, w):
        x = np.array(x_init)
        trajectory = np.array([x])
        time = 0
        while time <= self.sim_time_static:
            x = self.motion(x, [v, w], self.sim_granularity)
            trajectory = np.vstack((trajectory, x))
            time += self.sim_granularity
        return trajectory

    def calc_cost_static(self, trajectory, goal, ob):
        """Calculate cost of a trajectory."""
        to_goal_cost = np.linalg.norm(trajectory[-1, :2] - goal[:2])
        obstacle_cost = 0.0
        # Orientation cost
        psi_diff = abs(self.normalize_angle(trajectory[-1, 2] - goal[2]))
        # Safety cost
        safety_cost = self.calc_safety_cost_static(trajectory, ob)
        for point in trajectory:
            if len(ob) > 0:
                dists = np.linalg.norm(ob - point[:2], axis=1)
                min_dist = np.min(dists)
                if min_dist < self.config.robot_radius:
                    obstacle_cost += min_dist
       
        return self.config.cost_to_goal_penalty_static * to_goal_cost +\
               self.config.obstacle_cost_gain_static *obstacle_cost +\
               self.config.orientation_penalty_static*psi_diff +\
               min(self.config.safety_weight_static*safety_cost, 10.0)


    def calc_safety_cost_static(self, trajectory, ob):
        """Calculate safety cost based on obstacle density"""
        if not ob:
            return 0.0
        ob_array = np.array(ob)
        safety_cost = 0.0
        for point in trajectory[:, :2]:
            distances = np.hypot(point[0] - ob_array[:, 0], point[1] - ob_array[:, 1])
            num_nearby = np.sum(distances < self.config.robot_radius + 0.1)
            safety_cost += num_nearby
        return safety_cost / len(trajectory)  # Normalize by trajectory length

    """
    All function from now onwards are for DWA planner to deal
    with dynamic obstacles.
    """

    def plan_dynamic(self, x, goal, ob, sector_min_distances, sector_obstacle_counts, sector_future_obstacle_counts):
        """Plan for dynamic obstacles with reverse maneuver."""
        # Update tracks with new observations
        self.update_obstacle_tracks(x, ob)

        # Compute obstacle metrics
        if len(ob) > 0:
            ob_array = np.array(ob)  # Convert list of arrays to 2D NumPy array
            distances = np.hypot(ob_array[:, 0] - x[0], ob_array[:, 1] - x[1])
            min_distance = np.min(distances)
            obstacle_density = np.sum(distances < self.R)
        else:
            min_distance = float('inf')
            obstacle_density = 0

        # Set dynamic parameters
        if min_distance < 2.0 or obstacle_density > 250:  # Crowded mode
            self.sim_time = 2.0
            self.horizon = 5.0
            self.vx_samples = 15
            self.vtheta_samples = 20
            self.percent_speed_reduction = 0.5
        else:  # Free mode
            self.sim_time = 5.0
            self.horizon = 10.0
            self.vx_samples = 10
            self.vtheta_samples = 10
            self.percent_speed_reduction = 0.4


        # Get predicted obstacle positions
        predicted_obs = self.track_obstacle_trajectories()
        # print(f'pred obs - {predicted_obs}')
        check_flag, self.direction = self.check_oncoming_obstacle(x, predicted_obs)
        
        if np.linalg.norm(x[:2] - goal[:2]) > self.goal_distance:
            if check_flag and self.direction == 'front':
                if min([d for d in sector_min_distances]) < self.R:
                    print(f'-------- reverse mechanism triggered ----------s')
                    return self.execute_reverse_maneuver(x)
        
        if self.direction == 'back':
            dw = self.calc_dynamic_window(x, front=True)
        else:
            dw = self.calc_dynamic_window(x)
        u, trajectory = self.calc_control_and_trajectory(x, dw, goal, predicted_obs, sector_min_distances, sector_obstacle_counts, sector_future_obstacle_counts)
        return u, trajectory
    
    def trajectories_intersect(self, vehicle_traj, obstacle_traj, threshold):
        
        vehicle_future = vehicle_traj[1:]  # Exclude initial position for future prediction
        if len(obstacle_traj) == 1:
            # Static obstacle: check distance from its position to all vehicle future positions
            obstacle_pos = obstacle_traj[0]
            distances = np.hypot(vehicle_future[:, 0] - obstacle_pos[0], 
                                vehicle_future[:, 1] - obstacle_pos[1])
            return np.any(distances < threshold)
        else:
            # Dynamic obstacle: check distances at corresponding time steps
            # Assumes obstacle_traj matches vehicle_future in length
            distances = np.hypot(vehicle_future[:, 0] - obstacle_traj[:, 0], 
                                vehicle_future[:, 1] - obstacle_traj[:, 1])
            return np.any(distances < threshold)


    # def update_obstacle_tracks(self, x, current_obs):
    #     """
    #     Update tracked obstacles using global-frame laser points.
    #     x: vehicle state [x, y, theta, ...]
    #     current_obs: list of [x_global, y_global] points from process_laser_scan
    #     """
    #     obs_array = np.array(current_obs)
    #     if len(obs_array) == 0:
    #         return

    #     # Cluster points with DBSCAN
    #     clustering = DBSCAN(eps=0.6, min_samples=3).fit(obs_array)
    #     labels = clustering.labels_
    #     unique_labels = set(labels) - {-1}  # Exclude noise

    #     clustered_obs = {}
    #     for label in unique_labels:
    #         cluster_points = obs_array[labels == label]
    #         if len(cluster_points) >= 3:  # Need at least 3 points for circle fitting
    #             center_x, center_y, r = self.fit_circle_to_points(cluster_points)
    #             if r <= 0.6:
    #                 clustered_obs[label] = [center_x, center_y]
    #                 # print(f'centroid - {[center_x, center_y]}, radius - {r}')
        
    #     # Predict vehicle trajectory
    #     vehicle_traj = self.predict_trajectory(x, x[3], x[4])  # Assuming speed, steering

    #     # Update or create tracks
    #     threatening_ids = set()
    #     for o_id, tracked in self.tracked_obstacles.items():
    #         # Predict obstacle trajectory using its latest position and velocity
    #         num_steps = len(vehicle_traj) - 1  # Match future steps
    #         times = np.arange(1, num_steps + 1) * self.sim_granularity
    #         pos = np.array(tracked.positions[-1])  # Latest position
    #         vel = np.array(tracked.velocity)       # Velocity estimate
    #         obstacle_traj = pos + np.outer(times, vel)  # Linear prediction
    #         if self.trajectories_intersect(vehicle_traj, obstacle_traj, self.traj_collision_distance):
    #             threatening_ids.add(o_id)

    #     updated_ids = set()
    #     for label, center in clustered_obs.items():
    #         closest_id, min_dist = None, float('inf')
    #         for o_id in threatening_ids:
    #             tracked = self.tracked_obstacles[o_id]
    #             distance = np.hypot(center[0] - tracked.positions[-1][0], 
    #                                 center[1] - tracked.positions[-1][1])
    #             if distance < min_dist and distance < self.config.tracking_max_dist:
    #                 closest_id = o_id
    #                 min_dist = distance
    #         if closest_id is not None:
    #             self.tracked_obstacles[closest_id].update(center, self.current_time)
    #             updated_ids.add(closest_id)
    #         else:
    #             obs_pos = np.array([center])
    #             if self.trajectories_intersect(vehicle_traj, obs_pos, self.R):
    #                 new_id = self.next_obstacle_id
    #                 self.tracked_obstacles[new_id] = TrackedObstacle(
    #                     center, self.current_time, new_id
    #                 )
    #                 self.next_obstacle_id += 1
    #                 updated_ids.add(new_id)
        
    #     stale_ids = []
    #     for o_id, tracked in self.tracked_obstacles.items():
    #         if o_id not in updated_ids:
    #             if self.current_time - tracked.timestamps[-1] > self.config.tracking_max_age:
    #                 stale_ids.append(o_id)
    #     for o_id in stale_ids:
    #         del self.tracked_obstacles[o_id]
    #     print(f'-----------Tracked {len(self.tracked_obstacles)} obstacles-------------')
    #     self.current_time += self.config.dt

    """
    Kalman filter without threatening id logic
    """
    def update_obstacle_tracks(self, x, current_obs):
        """Update tracked obstacles using global-frame laser points.
        
        Args:
            x: Vehicle state [x, y, theta, ...]
            current_obs: List of [x_global, y_global] points from process_laser_scan
        """
        obs_array = np.array(current_obs)
        if len(obs_array) == 0:
            return

        # Cluster points with DBSCAN
        clustering = DBSCAN(eps=0.6, min_samples=3).fit(obs_array)
        labels = clustering.labels_
        unique_labels = set(labels) - {-1}  # Exclude noise

        clustered_obs = {}
        for label in unique_labels:
            cluster_points = obs_array[labels == label]
            if len(cluster_points) >= 3:  # Need at least 3 points for circle fitting
                center_x, center_y, r = self.fit_circle_to_points(cluster_points)
                if r <= 0.6:  # Max radius from PDF
                    clustered_obs[label] = [center_x, center_y]

        # Predict vehicle trajectory
        vehicle_traj = self.predict_trajectory(x, x[3], x[4])  # Assuming speed, steering

        # Predict current positions of tracked obstacles
        for tracked in self.tracked_obstacles.values():
            dt = self.current_time - tracked.timestamp
            tracked.predict(dt)

        # Associate measurements with tracks
        measurements = list(clustered_obs.values())
        tracks = list(self.tracked_obstacles.values())
        updated_ids = set()
        if tracks and measurements:
            # Nearest neighbor association using Hungarian algorithm
            dist_matrix = np.zeros((len(tracks), len(measurements)))
            for i, track in enumerate(tracks):
                for j, meas in enumerate(measurements):
                    dist_matrix[i, j] = np.hypot(track.kf.x[0] - meas[0], 
                                                track.kf.x[1] - meas[1])
            row_ind, col_ind = linear_sum_assignment(dist_matrix)
            for r, c in zip(row_ind, col_ind):
                if dist_matrix[r, c] < self.config.tracking_max_dist:
                    tracks[r].update(measurements[c], self.current_time)
                    updated_ids.add(tracks[r].id)
                    measurements[c] = None  # Mark as assigned
            measurements = [m for m in measurements if m is not None]

        # Create new tracks for unassigned measurements
        for meas in measurements:
            obs_pos = np.array([meas])
            if self.trajectories_intersect(vehicle_traj, obs_pos, self.traj_collision_distance):
                new_id = self.next_obstacle_id
                self.tracked_obstacles[new_id] = TrackedObstacle(
                    meas, self.current_time, new_id, self.config.dt
                )
                self.next_obstacle_id += 1
                updated_ids.add(new_id)

        # Remove stale tracks
        stale_ids = []
        for o_id, tracked in self.tracked_obstacles.items():
            if self.current_time - tracked.timestamps[-1] > self.config.tracking_max_age:
                stale_ids.append(o_id)
        for o_id in stale_ids:
            del self.tracked_obstacles[o_id]
        
        print(f'-----------Tracked {len(self.tracked_obstacles)} obstacles-------------')
        self.current_time += self.config.dt

    # def update_obstacle_tracks(self, x, current_obs):
    #     """
    #     Update tracked obstacles using global-frame laser points.

    #     Args:
    #         x: Vehicle state [x, y, theta, speed, steering, ...]
    #         current_obs: List of [x_global, y_global] points from process_laser_scan
    #     """
    #     obs_array = np.array(current_obs)
    #     if len(obs_array) == 0:
    #         return

    #     # Cluster points with DBSCAN
    #     clustering = DBSCAN(eps=0.6, min_samples=3).fit(obs_array)
    #     labels = clustering.labels_
    #     unique_labels = set(labels) - {-1}  # Exclude noise

    #     clustered_obs = {}
    #     for label in unique_labels:
    #         cluster_points = obs_array[labels == label]
    #         if len(cluster_points) >= 3:  # Need at least 3 points for circle fitting
    #             center_x, center_y, r = self.fit_circle_to_points(cluster_points)
    #             if r <= 0.6:
    #                 clustered_obs[label] = [center_x, center_y]

    #     # Predict vehicle trajectory
    #     vehicle_traj = self.predict_trajectory(x, x[3], x[4])  # Assuming speed, steering

    #     # Update or create tracks with Kalman filters
    #     threatening_ids = set()
    #     for o_id, tracked in self.tracked_obstacles.items():
    #         # Predict obstacle trajectory using Kalman filter
    #         num_steps = len(vehicle_traj) - 1  # Match vehicle trajectory steps
    #         obstacle_traj = []
    #         kf_copy = self.copy_kalman_filter(tracked.kf)  # Avoid modifying original filter
    #         for _ in range(num_steps):
    #             kf_copy.predict()
    #             obstacle_traj.append(kf_copy.x[:2].copy())  # Append [x, y] position
    #         obstacle_traj = np.array(obstacle_traj)

    #         # Check for collision threat
    #         if self.trajectories_intersect(vehicle_traj, obstacle_traj, self.traj_collision_distance):
    #             threatening_ids.add(o_id)

    #     updated_ids = set()
    #     for label, center in clustered_obs.items():
    #         closest_id, min_dist = None, float('inf')
    #         for o_id in threatening_ids:
    #             tracked = self.tracked_obstacles[o_id]
    #             distance = np.hypot(center[0] - tracked.kf.x[0], 
    #                                 center[1] - tracked.kf.x[1])
    #             if distance < min_dist and distance < self.config.tracking_max_dist:
    #                 closest_id = o_id
    #                 min_dist = distance
    #         if closest_id is not None:
    #             # Update Kalman filter with new measurement
    #             self.tracked_obstacles[closest_id].update(center, self.current_time)
    #             updated_ids.add(closest_id)
    #         else:
    #             # Check if unassigned obstacle is a threat
    #             obs_pos = np.array([center])
    #             if self.trajectories_intersect(vehicle_traj, obs_pos, self.R):
    #                 new_id = self.next_obstacle_id
    #                 self.tracked_obstacles[new_id] = TrackedObstacle(
    #                     center, self.current_time, new_id, self.config.dt
    #                 )
    #                 self.next_obstacle_id += 1
    #                 updated_ids.add(new_id)

    #     # Remove stale tracks
    #     stale_ids = []
    #     for o_id, tracked in self.tracked_obstacles.items():
    #         if o_id not in updated_ids:
    #             if self.current_time - tracked.timestamps[-1] > self.config.tracking_max_age:
    #                 stale_ids.append(o_id)
    #     for o_id in stale_ids:
    #         del self.tracked_obstacles[o_id]

    #     print(f'-----------Tracked {len(self.tracked_obstacles)} obstacles-------------')
    #     self.current_time += self.config.dt

    def copy_kalman_filter(self, kf):
        """
        Create a copy of the Kalman filter to predict trajectories without modifying the original.

        Args:
            kf: Original KalmanFilter instance
        Returns:
            kf_copy: A new KalmanFilter instance with copied parameters
        """
        kf_copy = KalmanFilter(dim_x=4, dim_z=2)  # Assuming 4D state [x, y, vx, vy], 2D measurement [x, y]
        kf_copy.x = kf.x.copy()  # State vector
        kf_copy.P = kf.P.copy()  # Covariance matrix
        kf_copy.F = kf.F.copy()  # State transition matrix
        kf_copy.H = kf.H.copy()  # Measurement function
        kf_copy.R = kf.R.copy()  # Measurement noise covariance
        kf_copy.Q = kf.Q.copy()  # Process noise covariance
        return kf_copy

    def check_oncoming_obstacle(self, x, pred_obs):
    
        robot_heading = x[2]  # Robot's heading in radians

        robot_traj = self.predict_trajectory(x, x[3], x[4], a=0.1)
        threshold = self.config.robot_radius + self.config.obstacle_radius + 0.8


        for obs in pred_obs:
            # for future_pos in obs:
            obs_traj = np.array(obs)

            # Check if the robot's trajectory intersects with the obstacle's trajectory
            if self.trajectories_intersect(robot_traj, obs_traj, threshold):
                # Calculate relative position of obstacle to robot
                dx = obs_traj[0][0] - x[0]
                dy = obs_traj[0][1] - x[1]
                distance = np.hypot(dx, dy)

                # Check if obstacle is within dangerous distance
                if distance < self.config.danger_distance:
                    # Calculate angle from robot to obstacle
                    obstacle_angle = np.arctan2(dy, dx)
                    relative_angle = self.normalize_angle(obstacle_angle - robot_heading)

                    # Ensure angle is within laser scan range [-135, 135] degrees
                    if abs(relative_angle) <= np.deg2rad(135):
                        # Determine approach direction based on relative angle
                        # print(f'relative angle - {np.rad2deg(relative_angle)}')
                        if -np.pi/2 < relative_angle and relative_angle < np.pi/2:  # [-55, 55] degrees: front
                            approach_direction = 'front'
                        else:  # [90, 135] or [-135, -90] degrees: back
                            approach_direction = 'back'

                        # # Check if obstacle is a threat (within frontal_danger_angle)
                        # if abs(relative_angle) < self.config.frontal_danger_angle:
                        return True, approach_direction
        return False, None
    
    def execute_reverse_maneuver(self, x):
        if not self.reverse_mode:
            self.reverse_mode = True
            self.reverse_duration = 0.0
            self.reverse_turn_direction = 0.0# if np.random.rand() > 0.5 else -0.25
        
        v = -self.config.reverse_speed
        w = self.reverse_turn_direction * self.config.reverse_turn_rate
        self.reverse_duration += self.config.dt
            
        # Check if we've reversed enough
        if self.reverse_duration > self.config.max_reverse_duration:
            self.reverse_mode = False
            self.reverse_turn_direction = 0
        
        trajectory = self.predict_trajectory(x, v, w)
        return [v, w], trajectory


    # def track_obstacle_trajectories(self):
    #     num_steps = int(self.horizon / self.sim_granularity) + 1
    #     times = np.arange(1, num_steps + 1) * self.config.dt
    #     predicted = []
    #     for o_id, tracked in self.tracked_obstacles.items():
    #         if len(tracked.positions) < 2:
    #             continue
    #         pos = np.array(tracked.positions[-1])
    #         vel = np.array(tracked.velocity)
    #         future_pos = pos + np.outer(times, vel)  # Vectorized prediction
    #         predicted.append(future_pos)
    #     return predicted    

    def track_obstacle_trajectories(self):
        """Predict future trajectories of tracked obstacles.
        
        Returns:
            List of predicted trajectories, each an array of [x, y] positions
        """
        num_steps = int(self.horizon / self.sim_granularity) + 1
        times = np.arange(1, num_steps + 1) * self.config.dt
        predicted = []
        for tracked in self.tracked_obstacles.values():
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
                kf_copy.F = np.array([[1, 0, self.config.dt, 0],
                                    [0, 1, 0, self.config.dt],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
                kf_copy.predict()
                traj.append(kf_copy.x[:2].copy())
            predicted.append(np.array(traj))
        return predicted
    
    def calc_dynamic_window(self, x, front=False):
        if front:
            vs = [0.8, self.config.max_vel_x,
                  0, self.config.max_vel_theta]
        else:
            vs = [0, (1 - self.percent_speed_reduction)*self.config.max_vel_x,
                -self.config.max_vel_theta, self.config.max_vel_theta]
        vd = [x[3] - self.config.acc_lim_x * self.sim_granularity,
              x[3] + self.config.acc_lim_x * self.sim_granularity,
              x[4] - self.config.acc_lim_theta * self.sim_granularity,
              x[4] + self.config.acc_lim_theta * self.sim_granularity]
        
        # vd = [x[3] - self.config.acc_lim_x * self.config.dt,
        #       x[3] + self.config.acc_lim_x * self.config.dt,
        #       x[4] - self.config.acc_lim_theta * self.config.dt,
        #       x[4] + self.config.acc_lim_theta * self.config.dt]

        dw = [max(vs[0], vd[0]), min(vs[1], vd[1]),
              max(vs[2], vd[2]), min(vs[3], vd[3])]
        print(f'control inputs range - {dw}')
        return dw

    def calc_control_and_trajectory(self, x, dw, goal, ob, sector_min_distances, sector_obstacle_counts, sector_future_obstacle_counts):
        x_init = x[:]
        min_cost = float('inf')
        best_u = [0.0, 0.0]
        best_trajectory = np.array([x])

        for v in np.linspace(dw[0], dw[1], self.vx_samples):
            for w in np.linspace(dw[2], dw[3], self.vtheta_samples):
                trajectory = self.predict_trajectory(x_init, v, w)
                dx = trajectory[-1, 0] - x_init[0]
                dy = trajectory[-1, 1] - x_init[1]
                global_angle = np.arctan2(dy, dx)
                relative_angle = self.normalize_angle(global_angle - x_init[2])
                sector_index = self.get_sector_index(relative_angle)
                sector_min_distance = sector_min_distances[sector_index]
                obstacle_count = sector_obstacle_counts[sector_index]
                future_obstacle_count = sector_future_obstacle_counts[sector_index]
                cost = self.calc_cost(trajectory, goal, ob, sector_min_distance, obstacle_count, future_obstacle_count)
                if cost < min_cost:
                    min_cost = cost
                    best_u = [v, w]
                    best_trajectory = trajectory

        if min_cost == float('inf'):
            if self.direction is not None:
                if self.direction == 'front':
                    best_u = [-2.0, 0]
                else:
                    best_u = [2.0, 0]
            else:
                best_u = [1.0, 0]

        return best_u, best_trajectory

    def get_sector_index(self, relative_angle):
        """Map a relative angle to a sector index."""
        sector_index = min(int((relative_angle - self.angle_min) / self.sector_width), self.num_sectors - 1)
        return max(0, min(sector_index, self.num_sectors - 1))  # Clamp to valid range

    def predict_trajectory(self, x_init, v, w, a=0.5):
        x = np.array(x_init)
        trajectory = np.array([x])
        time = 0
        v_current = v
        while time <= self.horizon:
            # x = self.motion(x, [v, w], self.sim_granularity)
            # x = self.motion(x, [v_current, w], self.sim_granularity)
            # v_current = min(self.config.max_vel_x, v_current + a * self.sim_granularity)  # Accelerate
            # trajectory = np.vstack((trajectory, x))
            # time += self.sim_granularity
            x = self.motion(x, [v_current, w], self.config.dt*5)
            v_current = min(self.config.max_vel_x, v_current + a * self.config.dt)  # Accelerate
            trajectory = np.vstack((trajectory, x))
            time += self.sim_granularity
        return trajectory
    
    def motion(self, x, u, dt):
        x[0] += u[0] * np.cos(x[2]) * dt
        x[1] += u[0] * np.sin(x[2]) * dt
        x[2] += u[1] * dt
        x[3] = u[0]
        x[4] = u[1]
        return x


    def calc_cost(self, trajectory, goal, ob, sector_min_distance, obstacle_count, future_obstacle_count):
        goal_cost = self.calc_to_goal_cost(trajectory, goal)
        obs_cost = self.calc_obstacle_cost(trajectory, ob)

        # Orientation cost
        psi_diff = abs(self.normalize_angle(trajectory[-1, 2] - goal[2]))
        # Safety cost
        # safety_cost = self.calc_safety_cost(trajectory, ob)
        direction_cost = -1.0 * sector_min_distance  # Favor farther obstacles
        obstacle_count_cost = self.obstacle_count_weight * obstacle_count
        future_obstacle_count_cost = self.future_obstacle_count_weight * future_obstacle_count
        # print(f'obstacle count - {obstacle_count}')
        # print(f'future obstacle count - {future_obstacle_count}')
        # path_cost = self.config.pdist_scale * self.calc_path_cost(trajectory)  # New method to calculate path following cost
        # print(f'obstacle cost - {self.config.obstacle_cost_gain *obs_cost}')
        # print(f'goal cost - {self.config.cost_to_goal_penalty * goal_cost}')
        # print(f'orientation cost - {self.config.orientation_penalty*psi_diff}')
        # print(f'safety cost - {self.config.safety_weight*safety_cost}')
        return self.config.cost_to_goal_penalty * goal_cost +\
               self.config.obstacle_cost_gain *obs_cost +\
               self.config.orientation_penalty*psi_diff +\
               direction_cost +\
               obstacle_count_cost #+\
            #    future_obstacle_count_cost
            

    def calc_to_goal_cost(self, trajectory, goal):
        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        
        return self.config.to_goal_cost_gain * np.hypot(dx, dy)

    
    def calc_obstacle_cost(self, trajectory, pred_obs):
        """Calculate cost based on predicted collisions."""
        if not pred_obs:
            return 0.0
        
        traj_pos = trajectory[0:, :2]  # Robot's future positions (including start)
        min_r = float('inf')
        
        for i in range(len(traj_pos)):
            robot_future_pos = traj_pos[i]
            for obs_traj in pred_obs:
                if i < len(obs_traj):  # Ensure time steps align
                    obs_future_pos = obs_traj[i]
                    dx = robot_future_pos[0] - obs_future_pos[0]
                    dy = robot_future_pos[1] - obs_future_pos[1]
                    distance = np.hypot(dx, dy)
                    # Check for collision at this time step
                    if distance <= (self.config.robot_radius + self.config.obstacle_radius) + 0.5:
                        return float('inf')  # Collision detected
                    min_r = min(min_r, distance)
        # Return cost based on closest approach if no collision
        return 1.0 / min_r if min_r < float('inf') else 0.0

    def calc_safety_cost(self, trajectory, ob):
        """Calculate safety cost based on obstacle density"""
        if not ob:
            return 0.0
        ob_array = np.array(ob)
        safety_cost = 0.0
        for point in trajectory[:, :2]:
            distances = np.hypot(point[0] - ob_array[:, 0], point[1] - ob_array[:, 1])
            num_nearby = np.sum(distances < self.R)
            safety_cost += num_nearby
        return safety_cost / len(trajectory)  # Normalize by trajectory length

    @staticmethod
    def normalize_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    @staticmethod
    def fit_circle_to_points(points):
        """
        Fit a circle to a set of 2D points using least-squares.
        points: Nx2 array of [x, y] coordinates
        Returns: (center_x, center_y, radius)
        """
        x = points[:, 0]
        y = points[:, 1]
        A = np.column_stack([x, y, np.ones(len(x))])
        b = x**2 + y**2
        c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        center_x = c[0] / 2
        center_y = c[1] / 2
        radius = np.sqrt(c[2] + center_x**2 + center_y**2)
        return center_x, center_y, radius
 
    