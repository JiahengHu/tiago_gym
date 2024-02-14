import os
import cv2
import time
import h5py
import numpy as np

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=None, type=str, help="path to model checkpoint")
    args = parser.parse_args()

    data_path = args.data_path
    if os.path.isdir(args.data_path):
        demos = []
        for f in os.listdir(data_path):
            print(f)
            demos.append(h5py.File(os.path.join(data_path, f), 'r'))
    else:
        f = h5py.File(data_path, 'r')['data']
        demos = [f[d] for d in f.keys()]
        
    for demo in demos:

        for k in demo.keys():
            if k == 'obs':
                print('obs')
                for obs_k in demo['obs'].keys():
                    print(f'\t{obs_k}', demo['obs'][obs_k][()].shape)
            else:
                print(k, demo[k][()].shape)

        # agentview_depth= demo['obs']['agentview_depth'][()].astype(float)
        agentview_right_depth = demo['obs']['agentview_right_depth'][()].astype(float)
        agentview_left_depth = demo['obs']['agentview_left_depth'][()].astype(float)
        depth_img = np.concatenate((agentview_left_depth, agentview_right_depth), axis=2)
        if np.max(depth_img) > 1:
            depth_img = np.clip(depth_img, 0, 4000)/4000

        agentview_right_image = demo['obs']['agentview_right_image'][()].astype(float)
        agentview_left_image = demo['obs']['agentview_left_image'][()].astype(float)
        color_img = np.concatenate((agentview_left_image, agentview_right_image), axis=2)/255
        img = color_img
        # agentview = np.clip(agentview_image, 0, 4000)/4000
        # tiago_head = np.clip(tiago_head_depth, 0, 4000)/4000

        # agentview = demo['obs']['agentview_image'][()].astype(float)/255
        # tiago_head = demo['obs']['agentview_right_image'][()].astype(float)/255
        
        # agentview[(agentview_depth>=2000), :3] = np.array([0, 1, 0])
        # tiago_head[(tiago_head_image>=4000)[..., 0], :3] = np.array([0, 1, 0]) 
        
        start_time = time.time()
        for i in range(len(img)):
            # cv2.imshow('exo', agentview[i])
            cv2.imshow('ego', img[i])
            cv2.waitKey(1)
            # input()
    # print('Episode time:', 10*(time.time() - start_time))
    