from real_tiago.utils.general_utils import AttrDict

teleop_config = AttrDict(
    arm_left_controller=None,
    arm_right_controller='mobile_phone',
    base_controller=None,
    torso_controller=None,
    use_oculus=False,
    interface_kwargs=AttrDict(
        oculus={},
        human_kpt={},
        mobile_phone={},
    )
)