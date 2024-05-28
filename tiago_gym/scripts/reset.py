import rospy

from telemoma.human_interface.teleop_policy import TeleopPolicy
from telemoma.configs.only_vr import teleop_config

from importlib.machinery import SourceFileLoader
from tiago_gym.tiago.tiago_gym import TiagoGym

COMPATIBLE_ROBOTS = ['tiago', 'hsr']

def main():

    env = TiagoGym(
            frequency=10,
            head_enabled=False,#LookAtFixedPoint(point=np.array([0.8, 0, 0.8])),
            base_enabled=teleop_config.base_controller is not None,
            right_arm_enabled=teleop_config.arm_right_controller is not None,
            left_arm_enabled=False,
            right_gripper_type='robotiq2F-140',
            left_gripper_type='robotiq2F-85',)
    obs, _ = env.reset()

    teleop = TeleopPolicy(teleop_config)
    teleop.start()

    def shutdown_helper():
        teleop.stop()
    rospy.on_shutdown(shutdown_helper)

    while not rospy.is_shutdown():
        action = teleop.get_action(obs) # get_random_action()
        buttons = action.extra['buttons'] if 'buttons' in action.extra else {}
    
        if buttons.get('A', False) or buttons.get('B', False):
            break

        obs, _, _, _, _ = env.step(action)

    shutdown_helper()

if __name__ == "__main__":
    main()