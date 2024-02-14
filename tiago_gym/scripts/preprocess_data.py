import h5py
import numpy as np
import json
import cv2

def find_action_mean_std(data):
    action_list = []
    for demo in data.keys():
        action_list.append(data[demo]['actions'][:])
    action_list = np.concatenate(action_list, axis=0)

    return action_list.mean(axis=0), action_list.std(axis=0)

def print_values(a):
    for i in a:
        print(f"{i:.8f}", end=', ')
    print()

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--output', type=str, default='preprocessed_data.h5')
    args = parser.parse_args()
    # Load data
    data = h5py.File(args.data_path, 'r')['data']
    output = h5py.File(args.output, 'w')
    grp = output.create_group('data')

    # action_mean, action_std = find_action_mean_std(data)
    
    # print('action mean')
    # print_values(action_mean)
    # print('action std')
    # print_values(action_std)
    
    for demo in data.keys():
        print(demo)
        ep_grp = grp.create_group(demo)
        ep_grp.attrs['num_samples'] = data[demo].attrs['num_samples']
        for k in data[demo].keys():
            if k == 'actions':
                action = data[demo][k][:]
                # action[:, :-2] = (action[:, :-2] - action_mean[:-2])/action_std[:-2] # last action is gripper action
                ep_grp.create_dataset(k, data=action)
            elif k == 'obs':
                obs_grp = ep_grp.create_group(k)
                for obs_key in data[demo][k].keys():
                    # if 'image' in obs_key or 'depth' in obs_key:
                    #     obs_data = data[demo][k][obs_key][:]
                    #     resized_obs_data = []
                    #     for img in obs_data:
                    #         resized_obs_data.append(cv2.resize(img, (84, 84)))
                    #     obs_grp.create_dataset(obs_key, data=np.array(resized_obs_data))
                    # else:
                    #     obs_grp.create_dataset(obs_key, data=data[demo][k][obs_key][:])
                    # print(f'\t{obs_key} {data[demo][k][obs_key][:].shape}')

                    obs = data[demo][k][obs_key][:]
                    if 'depth' in obs_key and len(obs.shape[1:]) == 2:
                        obs = obs[..., None]
                    obs_grp.create_dataset(obs_key, data=obs)
            else:
                # print(f'{k} {data[demo][k][:].shape}')
                ep_grp.create_dataset(k, data=data[demo][k][:])

    grp.attrs['num_demos'] = data.attrs['num_demos']
    grp.attrs['total'] = data.attrs['total']
    
    # new attrs
    # grp.attrs['action_mean'] = action_mean[:6]
    # grp.attrs['action_std'] = action_std[:6]
    grp.attrs['env_args'] = json.dumps(dict(env_name='tiago_mobile_pnp', env_kwargs={}, type=1), indent=4)
    