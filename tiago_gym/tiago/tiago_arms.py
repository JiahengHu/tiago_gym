import rospy
import tf

import numpy as np
from geometry_msgs.msg import PoseStamped
from real_tiago.utils.ros_utils import Publisher, create_pose_command, TFTransformListener
from real_tiago.utils.transformations import euler_to_quat, quat_to_euler, add_angles


class TiagoArms:
    
    def __init__(self, arm_enabled, side='right') -> None:
        self.arm_enabled = arm_enabled
        self.side = side
        
        self.setup_listeners()
        self.setup_actors()

    def setup_listeners(self):
        self.arm_reader = TFTransformListener('/base_footprint')

    @property
    def arm_pose(self):
        pos, quat = self.arm_reader.get_transform(f'/arm_{self.side}_tool_link')

        if pos is None:
            return None
        return np.concatenate((pos, quat))
    
    def setup_actors(self):
        self.arm_writer = None
        if self.arm_enabled:
            self.arm_writer = Publisher(f'/whole_body_kinematic_controller/arm_{self.side}_tool_link_goal', PoseStamped)

    def process_action(self, action):
        # convert deltas to absolute positions
        pos_delta, euler_delta = action[:3], action[3:6]
        
        cur_pose = self.arm_pose
        cur_pos, cur_euler = cur_pose[:3], quat_to_euler(cur_pose[3:])
        
        target_pos = cur_pos + pos_delta
        target_euler = add_angles(euler_delta, cur_euler)
        target_quat = euler_to_quat(target_euler)
        return target_pos, target_quat
    
    def write(self, target_pos, target_quat):
        pose_command = create_pose_command(target_pos, target_quat)
        if self.arm_writer is not None:
            self.arm_writer.write(pose_command)
            
    def step(self, action):
        if self.arm_enabled:
            target_pos, target_quat = self.process_action(action)
            self.write(target_pos, target_quat)

    def reset(self, action):
        if self.arm_enabled:
            target_pos, target_quat = action[:3], action[3:7]
            self.write(target_pos, target_quat)