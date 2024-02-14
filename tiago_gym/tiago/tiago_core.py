import rospy
import time
import numpy as np



from real_tiago.utils.transformations import euler_to_quat, quat_to_euler, add_angles, pose_diff
from real_tiago.utils.ros_utils import Publisher, create_pose_command
from real_tiago.tiago.grippers import PALGripper, RobotiqGripper2F_140, RobotiqGripper2F_85
from real_tiago.tiago.head import TiagoHead
from real_tiago.tiago.tiago_mobile_base import TiagoBaseVelocityControl
from real_tiago.tiago.tiago_torso import TiagoTorso
from real_tiago.tiago.tiago_arms import TiagoArms

class Tiago:
    gripper_map = {'pal': PALGripper, 'robotiq2F-140': RobotiqGripper2F_140, 'robotiq2F-85': RobotiqGripper2F_85}

    def __init__(self,
                    head_policy=None,
                    base_enabled=False,
                    torso_enabled=False,
                    right_arm_enabled=True,
                    left_arm_enabled=True,
                    right_gripper_type=None,
                    left_gripper_type=None):
        

        self.head_enabled = head_policy is not None
        self.base_enabled = base_enabled
        self.torso_enabled = torso_enabled
        
        self.head = TiagoHead(head_policy=head_policy)
        self.base = TiagoBaseVelocityControl(base_enabled=base_enabled)
        self.torso = TiagoTorso(torso_enabled=torso_enabled)
        self.arms = {
            'right': TiagoArms(right_arm_enabled, side='right'),
            'left': TiagoArms(left_arm_enabled, side='left'),
        }

        # set up grippers
        self.gripper = {'right': None, 'left': None}
        for side in ['right', 'left']:
            gripper_type = right_gripper_type if side=='right' else left_gripper_type
            if gripper_type is not None:
                self.gripper[side] = self.gripper_map[gripper_type](side)

        # define reset poses
        # hands sideways
        # self.reset_pose = {
        #         'right': [0.30396375, -0.53687817, 0.9420241, 0.71603554, 0.29001453, -0.35806947, -0.5243767, 1], #[0.37959829, -0.35538707,  0.97079442,  0.79074465,  0.58323787, -0.16615811,  0.08335446],
        #         'left': [0.47942768, 0.34670412, 0.93795076, 0.05600708, -0.30537332, 0.60274801, -0.73505454, 1], #[0.04875318, 0.54197792, 1.1605404, -0.16659694, -0.26293025, -0.62166113, 0.7187841]   
        #         'torso': 0.15
        #     }
        
        # elbows out 
        # self.reset_pose = {
        #         'right': [0.38778598, -0.41286138, 0.91738961, 0.66607035, 0.44145962, -0.42317228, -0.42707014, 1],
        #         'left': [0.45538002, 0.35416999, 0.91831515, -0.31917555, 0.41893294, -0.46209671, 0.71350458, 1],
        #         # 'torso': 0.15
        #     }
        
        # table wiping 
        # self.reset_pose = {
        #         'right': [0.4116969, -0.33867811, 0.8728758, 0.58221538, 0.41260542, -0.36656786, -0.59700086, 0],
        #         'left': [0.21728141, 0.35833528, 0.49116128, -0.05623011, -0.01709008, -0.67002179, 0.74001142, 0],
        #     }

        # table wiping with cloth 
        # self.reset_pose = {
        #         'right': [0.40252533, -0.31732214, 0.9846104, 0.88028449, 0.36428899, -0.30376927, 0.01081556, 1],
        #         'left': [0.24322931, 0.40431961, 0.39713217, -0.51687655, -0.4396324, -0.37472597, 0.6317772, 1],
        #     }
        
        # sweeping pose 
        # self.reset_pose = {
        #         'right': [0.50266193, -0.22134536, 1.05599596, 0.62923094, 0.62728394, -0.38494165, -0.24980634, 1],
        #         'left': [0.51952967, 0.21295595, 0.71741947, 0.32540531, -0.44424084, 0.51187365, -0.65935334, 1],
        #         'torso': 0.15
        #     }

        # setting table 
        # self.reset_pose = {
        #         'right': [0.40252533, -0.31732214, 0.9846104, 0.88028449, 0.36428899, -0.30376927, 0.01081556, 1],
        #         'left': [0.22810874, 0.40165014, 0.42926794, 0.78939804, 0.01106771, 0.4160613, -0.45124408, 1],
        #         # 'torso': 0.15
        #     }

        # top down 
        self.reset_pose = {
                'right': [0.40252533, -0.31732214, 0.9846104, 0.88028449, 0.36428899, -0.30376927, 0.01081556, 1],
                'left': [0.44251871, 0.345624, 0.94518278, 0.0712286, 0.29554832, -0.2544916, 0.91804777, 1],
                'torso': 0.15
            }

    @property
    def right_gripper_pos(self):
        if self.gripper['right'] is None:
            return None
        return self.gripper['right'].get_state()

    @property
    def left_gripper_pos(self):
        if self.gripper['left'] is None:
            return None
        return self.gripper['left'].get_state()

    def step(self, action):
        
        for side in ['right', 'left']:
            if action[side] is None:
                continue
            
            arm_action = action[side][:6]
            gripper_action = action[side][6]

            self.arms[side].step(arm_action)
            
            if self.gripper[side] is not None:
                self.gripper[side].step(gripper_action)

        if self.head_enabled:
            self.head.step(action)
        
        if self.base_enabled:
            self.base.step(action['base'])

        if self.torso_enabled and (self.torso is not None):
            self.torso.step(action['torso'])

    def reset(self, reset_arms=True):
        for side in ['right', 'left']:
            if (self.reset_pose[side] is not None) and (self.arms[side].arm_enabled):
                arm_pose = self.arms[side].arm_pose
                self.gripper[side].step(self.reset_pose[side][7])

                while reset_arms and not (rospy.is_shutdown()):
                    self.arms[side].reset(self.reset_pose[side][:7])
                    rospy.sleep(0.1)

                    arm_pose = self.arms[side].arm_pose
                    # diff = pose_diff(self.reset_pose[side], arm_pose)
                    # print(diff)

                    if pose_diff(self.reset_pose[side], arm_pose) < 0.25:
                        break

                    print(f'resetting {side}...{time.time()}')

        if self.head_enabled:
            self.head.reset_step(self.reset_pose)

        if ('torso' in self.reset_pose.keys()) and (self.torso is not None):
            self.torso.reset(self.reset_pose['torso'])
        
        rospy.sleep(0.5)

        input('Reset complete. Press ENTER to continue')