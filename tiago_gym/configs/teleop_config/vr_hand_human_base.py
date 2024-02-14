from real_tiago.utils.general_utils import AttrDict
from real_tiago.utils.camera_utils import Camera

cam = Camera(img_topic="/body_tracking_camera/color/image_raw", depth_topic="/body_tracking_camera/aligned_depth_to_color/image_raw")

teleop_config = AttrDict(
    arm_left_controller='oculus',
    arm_right_controller='oculus',
    base_controller='human_kpt',
    torso_controller='human_kpt',

    interface_kwargs=AttrDict(
        oculus={},
        human_kpt=dict(camera=cam, set_ref=True),
    )
)