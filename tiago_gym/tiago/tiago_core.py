import rospy
import time
from tiago_gym.tiago.grippers import PALGripper, RobotiqGripper2F_140, RobotiqGripper2F_85
from tiago_gym.tiago.head import TiagoHead
from tiago_gym.tiago.tiago_mobile_base import TiagoBaseVelocityControl
from tiago_gym.tiago.tiago_torso import TiagoTorso
from tiago_gym.tiago.tiago_arms import TiagoArms

class Tiago:
    gripper_map = {'pal': PALGripper, 'robotiq2F-140': RobotiqGripper2F_140, 'robotiq2F-85': RobotiqGripper2F_85}

    def __init__(self,
                    head_enabled=False,
                    base_enabled=False,
                    torso_enabled=False,
                    right_arm_enabled=True,
                    left_arm_enabled=True,
                    right_gripper_type=None,
                    left_gripper_type=None):
        

        self.head_enabled = head_enabled
        self.base_enabled = base_enabled
        self.torso_enabled = torso_enabled
        
        self.head = TiagoHead(head_enabled=head_enabled)
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

        # top down (cover table / slide chair)
        self.reset_pose = {
                'right': [0.43, -0.81, 1.60, 1.78, 1.34, -0.49, 1.15, 1], # [0.89, -0.35, 1.76, 2.13, 1.90, 0.56, 1.80, 1],#
                'left': [0.43, -0.81, 1.60, 1.78, 1.34, -0.49, 1.15, 1],
                'torso': 0.15, # 0.25,
                'head': [0.0, -0.6],
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
            if (side not in action) or action[side] is None:
                continue
            
            arm_action = action[side][:6]
            gripper_action = action[side][6]
            # gripper_action = 0

            self.arms[side].step(arm_action)
            
            if self.gripper[side] is not None:
                self.gripper[side].step(gripper_action)
        
        if self.base_enabled and ('base' in action):
            self.base.step(action['base'])

        if self.torso_enabled and (self.torso is not None) and ('torso' in action):
            self.torso.step(action['torso'])
        
        if self.head_enabled and ('head' in action):
            self.head.step(action['head'])

    def reset(self, reset_arms=True):
        for side in ['right', 'left']:
            for _ in range(2):
                if (self.reset_pose[side] is not None) and (self.arms[side].arm_enabled):
                    self.gripper[side].step(self.reset_pose[side][-1])

                    if reset_arms:
                        print(f'resetting {side}...{time.time()}')
                        self.arms[side].reset(self.reset_pose[side][:-1])
                        rospy.sleep(1)

        if 'head' in self.reset_pose:
            self.head.reset(self.reset_pose)

        if ('torso' in self.reset_pose.keys()) and (self.torso is not None):
            self.torso.reset(self.reset_pose['torso'])
        
        rospy.sleep(0.5)

        # input('Reset complete. Press ENTER to continue')