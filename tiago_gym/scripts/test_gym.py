from tiago_gym.tiago.tiago_gym import TiagoGym


env = TiagoGym(
        frequency=10,
        head_enabled=True,
        base_enabled=False,
        torso_enabled=False,
        right_arm_enabled=False,
        left_arm_enabled=False,
        right_gripper_type=None,#'robotiq2F-140',
        left_gripper_type=None, #'robotiq2F-85'
    )
env.reset()

for i in range(10):
    action = int(input('enter:'))
    env.step({'head': action})