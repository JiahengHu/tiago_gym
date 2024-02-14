import os
import time
import cv2
import copy
import rospy
import numpy as np
import h5py
from real_tiago.user_interfaces.teleop_policy import TeleopPolicy
from real_tiago.tiago.tiago_gym import TiagoGym    
from real_tiago.utils.camera_utils import Camera
from cv_bridge import CvBridge
from real_tiago.tiago.head import LookAtFixedPoint

rospy.init_node('tiago_data_collect')

def start_teleop(teleop_config, render=False):

    # agentview_left = Camera(img_topic="/agentview_left/color/image_raw",
    #                           depth_topic="/agentview_left/aligned_depth_to_color/image_raw",)
    
    # # flip the camera stream for the right hand side
    # agentview_right = Camera(img_topic="/agentview_right/color/image_raw",
    #                           depth_topic="/agentview_right/aligned_depth_to_color/image_raw",
    #                           img_post_proc_func=img_flip_processing,
    #                           depth_post_proc_func=depth_flip_processing)

    env = TiagoGym(
                frequency=10,
                head_policy=None,#LookAtFixedPoint(point=np.array([0.8, 0, 0.8])),
                base_enabled=teleop_config.base_controller is not None,
                torso_enabled=False,
                right_arm_enabled=teleop_config.arm_right_controller is not None,
                left_arm_enabled=teleop_config.arm_right_controller is not None,
                right_gripper_type='robotiq2F-140',
                left_gripper_type='robotiq2F-85',)
                # external_cams={'agentview_left': agentview_left, 'agentview_right': agentview_right})
    
    # rospy.on_shutdown(shutdown_helper)
    
    obs = env.reset()

    teleop = TeleopPolicy(teleop_config)
    teleop.start()
    def shutdown_helper():
        teleop.stop()
    rospy.on_shutdown(shutdown_helper)

    trajectory = {'obs':{}, 'actions': [], 'rewards': [], 'dones': []}
    for k in obs.keys():
        trajectory['obs'][k] = []
    
    start_time = time.time()
    while not rospy.is_shutdown():
        action = teleop.get_action(obs)
        buttons = action.extra['buttons']

        n_obs, reward,  done, info = env.step(action)
        done = buttons.get('A', False)
        
        trajectory['dones'].append(done)
        trajectory['actions'].append(np.concatenate((action['base'], action['right'][:-1], action['left'][:-1], action['right'][-1:], action['left'][-1:]), axis=0))
        trajectory['rewards'].append(reward)
        for k in obs.keys():
            if 'agentview' in k:
                obs[k] = cv2.resize(obs[k].astype(float), (320, 180))
            elif 'tiago_head' in k:
                continue
                obs[k] = cv2.resize(obs[k].astype(float), (640, 360))    
            trajectory['obs'][k].append(obs[k])

        if done:
            break
        if buttons.get('B', False):
            total_time = time.time() - start_time
            print(total_time)
            teleop.stop()
            return None
        
        obs = copy.deepcopy(n_obs)

        if 'overlayed_image' in action.extra.keys():
            img = action.extra['overlayed_image']
            if img is not None:
                cv2.imshow('img', img)
                cv2.waitKey(1)
        # if render:
        #     print(obs['agentview_right_image'].shape)
        #     print(obs['agentview_left_image'].shape)
        #     print(obs['agentview_right_depth'].shape)
        #     print(obs['agentview_left_depth'].shape)
        #     print()

        #     color_img = np.concatenate((obs['agentview_left_image'], obs['agentview_right_image']), axis=1)/255
        #     # depth_img = np.clip(np.concatenate((obs['agentview_left_depth'], obs['agentview_right_depth']), axis=1), 0, 4000)/4000

        #     cv2.imshow('cam', color_img)
        #     cv2.waitKey(1)
    
    total_time = time.time() - start_time
    teleop.stop()


    return trajectory, total_time

def main(args):
    
    if args.teleop == 'vr':
        from real_tiago.configs.teleop_config.only_vr import teleop_config
    elif args.teleop == 'vr_human_kpt':
        from real_tiago.configs.teleop_config.vr_hand_human_base import teleop_config
    elif args.teleop == 'human_kpt':
        from real_tiago.configs.teleop_config.only_human_kpts import teleop_config
    else:
        raise NotImplementedError

    if len(args.task) > 0:
        experiment_dir = os.path.join(os.environ['EXP_DIR'], f'user_study/{args.task}/{args.user}/{args.teleop}')
    else:
        raise NotImplementedError
        experiment_dir = os.path.join(os.environ['EXP_DIR'], f'user_study/{args.user}/{args.teleop}')
    os.makedirs(experiment_dir, exist_ok=True)

    data, total_time = start_teleop(teleop_config, render=args.render)

    print("==> Total time:", total_time)

    if data is not None:
        save_dir = experiment_dir
        demo_id = len(os.listdir(save_dir))
        save_path = os.path.join(save_dir, f'demo_{demo_id}.h5')

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

        f.attrs['total_time'] = total_time

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default=None, type=str, help="name of the user")
    parser.add_argument("--teleop", default=None, type=str, help="mode of teleperation")
    parser.add_argument("--task", default="", type=str, help="task name")
    parser.add_argument("--render", action="store_true", help="pass flag to render environment while data generation")
    args = parser.parse_args()

    main(args)