import os
import rospy
import rospkg
import subprocess
import numpy as np
from scipy.signal import savgol_filter
from dynamic_reconfigure.client import Client
from scipy.spatial.transform import Rotation as R

from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
from robot_localization.srv import SetPose
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped

rospack = rospkg.RosPack()
navigation_path = rospack.get_path('jackal_navigation')
helper_path = rospack.get_path('jackal_helper')

STATE_NORMAL = 0


class Robot:
    def __init__(
            self,
            use_sim_time=True,
            remap_cmd_vel='move_base/cmd_vel',
            base_global_planner='navfn/NavfnROS',
            base_local_planner='eband_local_planner/EBandPlannerROS',
            los=2.0,  # Line of sight distance
            laser_clip=5.0,  # Maximum laser distance
            threshold_dist=0.5,  # Threshold distance for reducing the velocity
            threshold_v=0.25,
            max_v=2.0,
            min_v=-0.5,
            max_w=1.57,
            min_w=-1.57,
            global_path_sampling_length=10
    ):
        self.inflater_client = None

        # ROS initialization and parameter setting
        rospy.init_node('robot', anonymous=True)
        rospy.set_param('/use_sim_time', use_sim_time)

        # Launch the move_base node
        self.move_base = subprocess.Popen([
            'roslaunch',
            os.path.join(helper_path, 'launch', 'move_base.launch'),
            'base_global_planner:=' + base_global_planner,
            'base_local_planner:=' + base_local_planner,
            'remap_cmd_vel:=' + remap_cmd_vel
        ])

        # Subscribers
        rospy.Subscriber('/front/scan', LaserScan, self.update_laser, queue_size=1)
        rospy.Subscriber('/odometry/filtered', Odometry, self.update_state, queue_size=1)
        rospy.Subscriber('/move_base/GlobalPlanner/plan', Path, self.update_path, queue_size=1)

        # Publishers and service proxies
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
        self.pose_srv = rospy.ServiceProxy('/set_pose', SetPose)
        self.clear_costmap_srv = rospy.ServiceProxy('/move_base/clear_costmaps', Empty)

        # Parameters in float32
        self.los = np.float32(los)
        self.v_multiplier = np.float32(1.0)
        self.max_v = np.float32(max_v)
        self.min_v = np.float32(min_v)
        self.max_w = np.float32(max_w)
        self.min_w = np.float32(min_w)
        self.laser_dist = np.float32(laser_clip)
        self.threshold_dist = np.float32(threshold_dist)
        self.threshold_v = np.float32(threshold_v)

        self.base_local_planner = base_local_planner

        # Initial state in float32
        self.state = STATE_NORMAL
        self.inflated = False
        self.odom = np.zeros((3,), dtype=np.float32)
        self.lp_vel = np.zeros((2,), dtype=np.float32)
        self.local_goal = np.zeros((2,), dtype=np.float32)
        self.global_path = np.zeros((0, 2), dtype=np.float32)
        self.original_global_path = np.zeros((0, 2), dtype=np.float32)
        self.laser = np.full((720,), self.laser_dist, dtype=np.float32)

        self.reset()

    def update_state(self, msg):
        pose = msg.pose.pose
        twist = msg.twist.twist
        r = R.from_quat([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        self.odom[:] = (pose.position.x, pose.position.y, r.as_euler('zyx')[0])


    def update_path(self, msg):
        gp = np.array([[pose.pose.position.x, pose.pose.position.y] for pose in msg.poses], dtype=np.float32)
        self.original_global_path = gp.copy()


    def update_laser(self, msg):
        self.laser[:] = np.clip(np.array(msg.ranges, dtype=np.float32), 0, self.laser_dist)

    def set_velocity(self, v, w):
        # Scale the velocity to the maximum and minimum values and clip
        if self.state == STATE_NORMAL:
            twist = Twist()
            twist.linear.x = np.clip(v * self.max_v * self.v_multiplier, self.min_v, self.max_v)
            twist.angular.z = np.clip(w * self.max_w, self.min_w, self.max_w)
            self.vel_pub.publish(twist)


    def set_goal(self, x, y, psi):
        pose = PoseStamped()
        pose.header.frame_id = 'odom'
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.z = np.sin(psi / 2)
        pose.pose.orientation.w = np.cos(psi / 2)
        self.goal_pub.publish(pose)

    def clear_costmap(self):
        for _ in range(3):
            rospy.wait_for_service('/move_base/clear_costmaps')
            try:
                self.clear_costmap_srv()
            except rospy.ServiceException:
                print("/clear_costmaps service call failed")

            rospy.sleep(0.1)

    def reset(self):
        # Stop the robot
        self.set_velocity(0, 0)

        # Reset the robot's pose
        rospy.wait_for_service('/set_pose')
        reset_pose = PoseWithCovarianceStamped()
        reset_pose.header.frame_id = 'odom'
        self.pose_srv(reset_pose)

        self.clear_costmap()  # Clear the costmap
        self.odom[:] = (0, 0, 0)  # Reset the state

