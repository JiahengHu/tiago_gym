from real_tiago.utils.general_utils import AttrDict
from real_tiago.utils.camera_utils import Camera

teleop_config = AttrDict(
    arm_left_controller='oculus',
    arm_right_controller='oculus',
    base_controller='human_kpt',
    torso_controller='human_kpt',

    interface_kwargs=AttrDict(
        oculus={},
        human_kpt=dict(set_ref=True),
    )
)