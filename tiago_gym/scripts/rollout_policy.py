import os
import time
import cv2
import copy
import rospy
import numpy as np
import imageio
import torch
from tiago_gym.tiago.tiago_gym import TiagoGym    
from tiago_gym.utils.camera_utils import Camera, flip_img, img_processing, depth_processing
# from tiago_gym.wrappers.policy_wrappers import RoboMimicPolicy
from tiago_gym.env.hierarchical_env_wrapper import HierarchicalDiscreteEnv

rospy.init_node('tiago_rollout_policy')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# cover table with cloth-100 demos
DATA_STATS_PATH="~/Desktop/tiago_teleop/data/processed_cover_with_cloth_fixed_controller_n100/dataset_stats.pkl"

# slide chair 50 demos
# DATA_STATS_PATH="~/Desktop/tiago_teleop/data/processed_slide_chair/dataset_stats.pkl"

# serve bread 50 demos
# DATA_STATS_PATH="~/Desktop/tiago_teleop/data/processed_serve_bread/dataset_stats.pkl"

SINGLE_HAND=False
def rollout_policy(model_ckpt, save_vid=False):

    # load policy
    policy = None # TOdo: Load policy

    # agentview_left = Camera(img_topic="/agentview_left/color/image_raw",
    #                           depth_topic="/agentview_left/aligned_depth_to_color/image_raw",)
    
    # # flip the camera stream for the right hand side
    # agentview_right = Camera(img_topic="/agentview_right/color/image_raw",
    #                           depth_topic="/agentview_right/aligned_depth_to_color/image_raw",
    #                           img_post_proc_func=lambda x: flip_img(img_processing(x)),
    #                           depth_post_proc_func=lambda x: flip_img(depth_processing(x)))

    env = TiagoGym(
            frequency=10,
            head_enabled=True,
            base_enabled=True,
            right_arm_enabled=True,
            left_arm_enabled=(not SINGLE_HAND),
            right_gripper_type='robotiq2F-140',
            left_gripper_type='robotiq2F-85',
            # external_cams={'agentview_left': agentview_left, 'agentview_right': agentview_right}
            )

    # TODO: wrap the env in the hierarchical env wrapper
    # env = HierarchicalDiscreteEnv(env, ...

    obs, _ = env.reset() # We need to follow the gymnasium api now
    
    if save_vid:
        video = imageio.get_writer(f'/home/pal/Desktop/tiago_teleop/experiments/supp_videos/il_{time.asctime().replace(" ", "_")}.mp4', fps=10)

    while not rospy.is_shutdown():
        # resize images and align dims correctly
        for k in obs.keys():
            if 'image' in k or 'depth' in k:
                if 'depth' in k:
                    obs[k] = np.clip(obs[k], 0, 4000)/4000
                elif 'image' in k:
                    obs[k] = obs[k]/255

                obs[k] = np.array(cv2.resize(obs[k].astype(float), (180, 180)))
                if len(obs[k].shape) < 3: # happens in depth
                    obs[k] = obs[k][..., None]

                # print(k, obs[k].shape)

                obs[k] = obs[k].transpose(2, 0, 1)

        # TODO: These are for visualizing the current observations
        # color_img = np.concatenate((obs['agentview_left_image'], obs['agentview_right_image']), axis=2)
        # cv2.imshow('a', color_img.transpose(1, 2, 0))
        # cv2.waitKey(1)

        # policy_act = policy.get_action(obs)
        # if SINGLE_HAND:
        #     right_act = policy_act[3:10]
        # else:
        #     right_act = np.concatenate((policy_act[3:9], np.clip([policy_act[15]], 0, 1)))
        #     left_act = np.concatenate((policy_act[9:15], np.clip([policy_act[16]], 0, 1)))
        #     action['left'] = left_act
        #     print('left', left_act)
        # action['right'] = right_act
        # action['base'] = policy_act[:3]
        #
        # print('right', right_act)
        # print('base', policy_act[:3])
        # print()

        action = np.zeros(17)
            
        n_obs, reward,  done, info = env.step(action)
        done = False
        import ipdb; ipdb.set_trace()

        if done:
            break
        
        obs = copy.deepcopy(n_obs)

        if save_vid:
            img = np.array(cv2.cvtColor(np.concatenate((obs['agentview_left_image'], obs['agentview_right_image']), axis=1).astype(np.float32), cv2.COLOR_BGR2RGB))
            if buttons.get('RG', False) or buttons.get('LG', False):
                img[:, :, 1] = (img[:, :, 1] + 255)
            video.append_data(img)

    if save_vid:
        video.close()

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None, type=str, help="path to model checkpoint")
    parser.add_argument("--save_vid", action="store_true", help="create a video of rollout")
    args = parser.parse_args()

    rollout_policy(args.ckpt, args.save_vid)