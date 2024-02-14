import os
import time
import cv2
import copy
import rospy
import numpy as np
import h5py
from tiago_gym.tiago.tiago_gym import TiagoGym    
from tiago_gym.utils.camera_utils import Camera, flip_img, img_processing, depth_processing

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from tiago_gym.wrappers.policy_wrappers import RoboMimicPolicy

rospy.init_node('tiago_data_collect_dagger')

from telemoma.input_interface.teleop_policy import TeleopPolicy
# from telemoma.configs.only_vr import teleop_config
from telemoma.configs.vr_hand_human_base import teleop_config
# from telemoma.configs.only_human_kpts import teleop_config


device = TorchUtils.get_torch_device(try_to_use_cuda=True)

# cover table with cloth-50 demos
DATA_STATS_PATH="~/Desktop/tiago_teleop/data/processed_cover_with_cloth_fixed_controller_180x180/dataset_stats.pkl"


def process_obs_for_policy(obs):
    # resize images and align dims correctly
    obs_policy = copy.deepcopy(obs)
    for k in obs_policy.keys():
        if 'image' in k or 'depth' in k:
            if 'depth' in k:
                obs_policy[k] = np.clip(obs_policy[k], 0, 4000)/4000
            elif 'image' in k:
                obs_policy[k] = obs_policy[k]/255

            obs_policy[k] = np.array(cv2.resize(obs_policy[k].astype(float), (180, 180)))
            if len(obs_policy[k].shape) < 3: # happens in depth
                obs_policy[k] = obs_policy[k][..., None]
            
            obs_policy[k] = obs_policy[k].transpose(2, 0, 1)

    # color_img = np.concatenate((obs['agentview_left_image'], obs['agentview_right_image']), axis=2)
    # cv2.imshow('a', color_img.transpose(1, 2, 0))
    # cv2.waitKey(1)
            
    return obs_policy

def collect_trajectory(model_ckpt, render=False):
    
    # load policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=model_ckpt, device=device, verbose=True)
    policy = RoboMimicPolicy(
                    policy=policy,
                    data_stats_path=DATA_STATS_PATH
                )
    policy.start_episode()

    agentview_left = Camera(img_topic="/agentview_left/color/image_raw",
                              depth_topic="/agentview_left/aligned_depth_to_color/image_raw",)
    
    # flip the camera stream for the right hand side
    agentview_right = Camera(img_topic="/agentview_right/color/image_raw",
                              depth_topic="/agentview_right/aligned_depth_to_color/image_raw",
                              img_post_proc_func=lambda x: flip_img(img_processing(x)),
                              depth_post_proc_func=lambda x: flip_img(depth_processing(x)))

    env = TiagoGym(
                frequency=10,
                head_policy=None,#LookAtFixedPoint(point=np.array([0.8, 0, 0.8])),
                base_enabled=teleop_config.base_controller is not None,
                right_arm_enabled=teleop_config.arm_right_controller is not None,
                left_arm_enabled=teleop_config.arm_left_controller is not None,
                right_gripper_type='robotiq2F-140',
                left_gripper_type='robotiq2F-85',
                external_cams={'agentview_left': agentview_left, 'agentview_right': agentview_right})
    
    # rospy.on_shutdown(shutdown_helper)
    
    obs = env.reset()

    teleop = TeleopPolicy(teleop_config)
    teleop.start()
    def shutdown_helper():
        teleop.stop()
    rospy.on_shutdown(shutdown_helper)

    trajectory = {'obs':{}, 'actions': [], 'rewards': [], 'dones': [], 'intervention': []}
    for k in obs.keys():
        trajectory['obs'][k] = []
    
    while not rospy.is_shutdown():
        action = teleop.get_action(obs)
        buttons = action.extra['buttons']
        intervention = 0

        # rollout learning policy
        obs_policy = process_obs_for_policy(obs)

        policy_act = policy.get_action(obs_policy)
        right_act = np.concatenate((policy_act[3:9], np.clip([policy_act[15]], 0, 1)))
        left_act = np.concatenate((policy_act[9:15], np.clip([policy_act[16]], 0, 1)))
        print('right', right_act)
        print('left', left_act)
        print('base', policy_act[:3])
        print()

        if not buttons.get('RG', False):
            action['right'] = right_act
        if not buttons.get('LG', False):
            action['left'] = left_act
        
        if not (buttons.get('RG', False) or buttons.get('LG', False)):
            action['base'] = policy_act[:3]
            intervention = 1
        else:
            policy.start_episode()

        n_obs, reward,  done, info = env.step(action)
        done = buttons.get('A', False)
        
        trajectory['dones'].append(done)
        trajectory['actions'].append(np.concatenate((action['base'], action['right'][:-1], action['left'][:-1], action['right'][-1:], action['left'][-1:]), axis=0))
        # trajectory['actions'].append(np.concatenate((action['base'], action['right']), axis=0)) # for single hand tasks
        
        trajectory['intervention'].append(intervention)
        trajectory['rewards'].append(reward)
        for k in obs.keys():
            if 'agentview' in k:
                obs[k] = cv2.resize(obs[k].astype(float), (320, 180))
            elif 'tiago_head' in k:
                continue
                # obs[k] = cv2.resize(obs[k].astype(float), (640, 360))    
            trajectory['obs'][k].append(obs[k])

        if done:
            break
        if buttons.get('B', False):
            teleop.stop()
            return None
        
        obs = copy.deepcopy(n_obs)

        if render:
            print(obs['agentview_right_image'].shape)
            print(obs['agentview_left_image'].shape)
            print(obs['agentview_right_depth'].shape)
            print(obs['agentview_left_depth'].shape)
            print()

            color_img = np.concatenate((obs['agentview_left_image'], obs['agentview_right_image']), axis=1)/255
            depth_img = np.clip(np.concatenate((obs['agentview_left_depth'], obs['agentview_right_depth']), axis=1), 0, 4000)/4000

            cv2.imshow('cam', color_img)
            cv2.waitKey(1)
    teleop.stop()

    return trajectory

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default=None, type=str, help="path to save file")
    parser.add_argument("--ckpt", default=None, type=str, help="path to model checkpoint")
    parser.add_argument("--render", action="store_true", help="pass flag to render environment while data generation")
    args = parser.parse_args()

    data = collect_trajectory(args.ckpt, args.render)

    if (args.save_dir is not None) and (data is not None):
        os.makedirs(args.save_dir, exist_ok=True)
        demo_id = len(os.listdir(args.save_dir))
        save_path = os.path.join(args.save_dir, f'demo_{demo_id}.h5')

        f = h5py.File(save_path, 'w')

        for k in data.keys():
            print(f'Saving {k}')
            if k == 'obs':
                obs_grp = f.create_group('obs')

                for obs_k in data['obs'].keys():
                    print(f'Saving {obs_k}')
                    obs_grp.create_dataset(obs_k, data=data['obs'][obs_k])
            else:
                f.create_dataset(k, data=data[k])