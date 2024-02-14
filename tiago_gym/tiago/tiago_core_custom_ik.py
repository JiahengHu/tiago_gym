import rospy
import time
from real_tiago.tiago.grippers import PALGripper, RobotiqGripper2F_140, RobotiqGripper2F_85
from real_tiago.tiago.head import TiagoHead
from real_tiago.tiago.tiago_mobile_base import TiagoBaseVelocityControl
from real_tiago.tiago.tiago_torso import TiagoTorso
from real_tiago.tiago.tiago_arms_joint_control import TiagoArms

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

        # table wiping 
        # self.reset_pose = {
        #         'right': [0.6879232370415255, -0.18938817836598199, 1.652623384906656, 1.2307891223629752, 1.732222493750843, -1.2336764897519692, 1.4830483982125786, 1],
        #         'left': None,
        #         'torso': 0.20
        #     }

        # setting table 
        # self.reset_pose = {
        #         'right': [0.43, -0.81, 1.60, 1.78, 1.34, -0.49, 1.15, 1],#[0.8146930908512172, -1.0292691398778935, 1.9145233987126662, 1.5102534491665771, 1.0963280790720527, -1.0774586792690246, 1.4287736303476284, 1],
        #         'left': None, 
        #         'torso': 0.20
        #     }
                
        # setting table real kitchen
        # self.reset_pose = {
        #         'right': [0.43, -0.81, 1.60, 1.78, 1.34, -0.49, 1.15, 1],
        #         'left': None,#[0.43, -0.81, 1.60, 1.78, 1.34, -0.49, 1.15, 1],
        #         'torso': 0.20
        #     }
                
        # pick bag  
        # self.reset_pose = {
        #         'right': [0.43, -0.81, 1.60, 1.78, 1.34, -0.49, 1.15, 1],
        #         'left': [0.16588688275629324, -0.06832011995888876, 2.4623632821488286, 1.457491917348019, 0.30084924746003866, -0.7996450353112055, 1.4652705216276503, 1],
        #         'torso': 0.15
        #     }
        
        # table dusting 
        # self.reset_pose = {
        #         'right': [-1.1, 1.4679, 2.714, 1.7095, -1.5707999999999998, 1.3898, 0.0, 1], #[0.7677679961266471, 0.5666749611207413, 1.071022194521775, 2.053427911115679, 1.6631266532848683, 0.9905107596792719, 0.81808986016318, 1],
        #         'left': [0.5981070425992115, 0.546507535013576, 1.2578042307751307, 1.9281388056267161, -1.677812113262637, -0.8479098582364406, -0.5461565253477394, 0],
        #         'torso': 0.15
        #     }
        
        # user study - cover table
        # self.reset_pose = {
        #         'right': [0.5627022963996994, -1.020397924496464, 1.8496372087315345, 1.60724110816242, 1.1465655869515277, -0.9684700206753963, 1.3062607915934876, 1],
        #         'left': [0.5627022963996994, -1.020397924496464, 1.8496372087315345, 1.60724110816242, 1.1465655869515277, -0.9684700206753963, 1.3062607915934876, 1],
        #         'torso': 0.15
        #     } 
        
        # reshelving / open fridge
        # self.reset_pose = {
        #         'right': [0.6639007792649066, -0.09363582750924238, 1.7101484274751875, 1.729431531358358, 1.376650140979673, 0.6752109664332347, 1.443691039920559, 1],
        #         'left': None,
        #         'torso': 0.2
        #     }
        
        # top down (cover table / slide chair)
        self.reset_pose = {
                'right': [0.43, -0.81, 1.60, 1.78, 1.34, -0.49, 1.15, 1],
                'left': [0.43, -0.81, 1.60, 1.78, 1.34, -0.49, 1.15, 1],
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
            # gripper_action = 0

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
                self.gripper[side].step(self.reset_pose[side][-1])

                if reset_arms:
                    print(f'resetting {side}...{time.time()}')
                    self.arms[side].reset(self.reset_pose[side][:-1])
                    rospy.sleep(1)

        if self.head_enabled:
            self.head.reset_step(self.reset_pose)

        if ('torso' in self.reset_pose.keys()) and (self.torso is not None):
            self.torso.reset(self.reset_pose['torso'])
        
        rospy.sleep(0.5)

        input('Reset complete. Press ENTER to continue')