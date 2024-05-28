import rospy
import numpy as np

from std_msgs.msg import Header
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from tiago_gym.utils.ros_utils import Publisher, Listener, create_pose_command
from tiago_gym.utils.camera_utils import Camera

class TiagoHead:

    def __init__(self, head_enabled) -> None:
        self.head_enabled = head_enabled
        
        self.img_topic = "/xtion/rgb/image_raw"
        self.depth_topic = "/xtion/depth/image_raw"
        self.head_camera = Camera(img_topic=self.img_topic, depth_topic=self.depth_topic)
        self.setup_listeners()
        self.setup_actors()

    def setup_listeners(self):
        def joint_process_func(data):
            return np.array(data.actual.positions)

        self.joint_reader = Listener(f'/head_controller/state', JointTrajectoryControllerState, post_process_func=joint_process_func)

    def setup_actors(self):
        self.head_writer = Publisher('/head_controller/command', JointTrajectory)

        self.reset_state = 0

    def write(self, trans, quat):
        if self.head_enabled:
            self.head_writer.write(create_pose_command(trans, quat))

    def get_head_joints(self):
        return self.joint_reader.get_most_recent_msg()
    
    def get_camera_obs(self):
        return self.head_camera.get_camera_obs()
    
    def create_head_command(self, pos, duration=0.25):
        message = JointTrajectory()
        message.header = Header()
        message.joint_names = ['head_1_joint', 'head_2_joint']
        point = JointTrajectoryPoint(positions = pos, time_from_start = rospy.Duration(duration))
        message.points.append(point)
        
        return message

    def step(self, action):
        delta = np.zeros(2)
        if action == 1:
            delta[0] = 0.3
        elif action == 2:
            delta[0] = -0.3
        elif action == 3:
            delta[1] = 0.3
        elif action == 4:
            delta[1] = -0.3

        head_joint_goal = self.get_head_joints() + delta

        # joint limits
        head_joint_goal = np.clip(head_joint_goal, -0.9, 0.9)

        self.head_writer.write(self.create_head_command(head_joint_goal))

    def reset(self, abs_pos):
        if abs_pos is not None:
            cmd = self.create_head_command(abs_pos['head'], duration=0.5)
            self.head_writer.write(cmd)
        


