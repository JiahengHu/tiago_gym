import rospy
from tiago_gym.tiago.tiago_gym import TiagoGym
from tiago_gym.tiago.head import FollowHandPolicy, LookAtFixedPoint

rospy.init_node('tiago_teleop')
from telemoma.input_interface.teleop_policy import TeleopPolicy
from telemoma.configs.only_vr import teleop_config
# from telemoma.configs.vr_hand_human_base import teleop_config
from telemoma.configs.only_human_kpts import teleop_config
# from telemoma.configs.only_mobile_phone import teleop_config


env = TiagoGym(
        frequency=10,
        head_policy=None,#LookAtFixedPoint(point=np.array([0.8, 0, 0.5])),
        base_enabled=teleop_config.base_controller is not None,
        torso_enabled=teleop_config.base_controller is not None,
        right_arm_enabled=teleop_config.arm_right_controller is not None,
        left_arm_enabled=teleop_config.arm_left_controller is not None,
        right_gripper_type='robotiq2F-140',
        left_gripper_type='robotiq2F-85'
    )
obs = env.reset(reset_arms=True)

teleop = TeleopPolicy(teleop_config)
teleop.start()

def shutdown_helper():
    teleop.stop()
rospy.on_shutdown(shutdown_helper)

count = 0
while not rospy.is_shutdown():
    action = teleop.get_action(obs)
    buttons = action.extra['buttons'] if 'buttons' in action.extra else {}
    
    if buttons.get('A', False) or buttons.get('B', False):
        break
    obs, _,  _, _ = env.step(action)

shutdown_helper()