import os
import time
import cv2
import copy
import rospy
import numpy as np
import imageio
from real_tiago.tiago.tiago_gym import TiagoGym    
from real_tiago.tiago.head import LookAtFixedPoint
from real_tiago.user_interfaces.teleop_policy import TeleopPolicy
from real_tiago.utils.camera_utils import Camera, flip_img, img_processing, depth_processing
from real_tiago.wrappers.policy_wrappers import RoboMimicPolicy

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

rospy.init_node('tiago_rollout_policy')
from real_tiago.configs.teleop_config.vr_hand_human_base import teleop_config
# from real_tiago.configs.teleop_config.only_human_kpts import teleop_config

device = TorchUtils.get_torch_device(try_to_use_cuda=True)

def record_teleop(save_vid=False):

    # agentview_left = Camera(img_topic="/agentview_left/color/image_raw",
    #                           depth_topic="/agentview_left/aligned_depth_to_color/image_raw",)
    
    # # flip the camera stream for the right hand side
    # agentview_right = Camera(img_topic="/agentview_right/color/image_raw",
    #                           depth_topic="/agentview_right/aligned_depth_to_color/image_raw",
    #                           img_post_proc_func=lambda x: flip_img(img_processing(x)),
    #                           depth_post_proc_func=lambda x: flip_img(depth_processing(x)))
    
    env = TiagoGym(
            frequency=10,
            head_policy=None,#LookAtFixedPoint(point=np.array([0.8, 0, 0.8])),
            base_enabled=teleop_config.base_controller is not None,
            torso_enabled=False,
            right_arm_enabled=teleop_config.arm_right_controller is not None,
            left_arm_enabled=teleop_config.arm_left_controller is not None,
            right_gripper_type='robotiq2F-140',
            left_gripper_type='robotiq2F-85',
            # external_cams={'agentview_left': agentview_left, 'agentview_right': agentview_right}
            ) 
    obs = env.reset(reset_arms=False)

    teleop = TeleopPolicy(teleop_config)
    teleop.start()

    def shutdown_helper():
        teleop.stop()
    rospy.on_shutdown(shutdown_helper)
    
    if save_vid:
        stamp = time.asctime().replace(" ", "_")
        robot_video = imageio.get_writer(f'/home/pal/Desktop/tiago_teleop/experiments/supp_videos/robot_{stamp}.mp4', fps=10)
        user_video =  imageio.get_writer(f'/home/pal/Desktop/tiago_teleop/experiments/supp_videos/user_{stamp}.mp4', fps=10)

    start_time = time.time()
    while not rospy.is_shutdown():# and time.time()-start_time < 30:
        action = teleop.get_action(obs)
        buttons = action.extra['buttons']
            
        n_obs, reward, done, info = env.step(action)
        done = buttons.get('A', False)

        if done:
            break
        
        obs = copy.deepcopy(n_obs)

        if save_vid:
            # robot_img = np.array(cv2.cvtColor(np.concatenate((obs['agentview_left_image'], obs['agentview_right_image']), axis=1).astype(np.float32), cv2.COLOR_BGR2RGB))
            # robot_video.append_data(robot_img)

            user_img = np.array(cv2.cvtColor(action.extra['overlayed_image'].astype(np.float32), cv2.COLOR_BGR2RGB))
            user_video.append_data(user_img)
            cv2.imshow('user', user_img)
            cv2.waitKey(1)

    if save_vid:
        robot_video.close()
        user_video.close()
    teleop.stop()

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_vid", action="store_true", help="create a video of rollout")
    args = parser.parse_args()

    record_teleop(args.save_vid)