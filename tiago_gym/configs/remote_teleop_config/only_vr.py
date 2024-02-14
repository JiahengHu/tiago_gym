from real_tiago.utils.general_utils import AttrDict

teleop_config = AttrDict(
    arm_left_controller='oculus',
    arm_right_controller='oculus',
    base_controller='oculus',
    torso_controller='oculus',

    interface_kwargs=AttrDict(
        oculus={},
        human_kpt={},
    )
)