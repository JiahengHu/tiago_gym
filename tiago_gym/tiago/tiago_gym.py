import gym
import gymnasium
import time
import numpy as np
import rospy
from collections import OrderedDict
from gymnasium import spaces
from tiago_gym.tiago.tiago_core import Tiago
from tiago_gym.utils.general_utils import AttrDict

class TiagoGym(gymnasium.Env):

    def __init__(self,
                    frequency=10,
                    head_enabled=False,
                    base_enabled=False,
                    torso_enabled=False,
                    right_arm_enabled=True,
                    left_arm_enabled=True,
                    right_gripper_type=None,
                    left_gripper_type=None,
                    external_cams={}):
        
        super(TiagoGym).__init__()

        # rospy.init_node('tiago_gym')

        self.frequency = frequency
        self.head_enabled = head_enabled
        self.base_enabled = base_enabled
        self.torso_enabled = torso_enabled
        self.right_arm_enabled = right_arm_enabled
        self.left_arm_enabled = left_arm_enabled
        self.right_gripper_enabled = right_gripper_type is not None
        self.left_gripper_enabled = left_gripper_type is not None

        self.tiago = Tiago(
                        head_enabled=head_enabled,
                        base_enabled=base_enabled,
                        torso_enabled=torso_enabled,
                        right_arm_enabled=right_arm_enabled,
                        left_arm_enabled=left_arm_enabled,
                        right_gripper_type=right_gripper_type,
                        left_gripper_type=left_gripper_type
                    )

        self.cameras = OrderedDict()
        self.cameras['tiago_head'] = self.tiago.head.head_camera
        for cam_name in external_cams.keys():
            self.cameras[cam_name] = external_cams[cam_name]

        self.steps = 0

    @property
    def observation_space(self):
        ob_space = OrderedDict()

        ob_space['right'] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3+4+1,),
        )

        ob_space['left'] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3+4+1,),
        )
        ob_space['base_delta_pose'] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2+1,) # 2d x, y position delta, 1d z orientation delta
        )
        
        ob_space['base_velocity'] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2+1,) # 2d x, y linear velocity, 1d z angular velocity
        )

        ob_space['torso'] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1,)
        )

        ob_space['head'] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2,)
        )
        
        for cam in self.cameras.keys():
            
            ob_space[f'{cam}_image'] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=self.cameras[cam].img_shape,
            ) 

            ob_space[f'{cam}_depth'] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=self.cameras[cam].depth_shape,
            )

        return spaces.Dict(ob_space)

    @property
    def action_space(self):
        act_space = OrderedDict()
        if self.right_arm_enabled:
            r_arm_max = 0.1
            act_space['right'] = spaces.Box(
                low=-r_arm_max,
                high=r_arm_max,
                shape=(3+4+int(self.right_gripper_enabled),),
            )

        if self.left_arm_enabled:
            l_arm_max = 0.1
            act_space['left'] = spaces.Box(
                low=-l_arm_max,
                high=l_arm_max,
                shape=(3+4+int(self.left_gripper_enabled),),
            )

        if self.base_enabled:
            base_max = 0.3
            act_space['base'] = spaces.Box(
                low=-base_max,
                high=base_max,
                shape=(3,), # 2d x, y linear velocity, 1d z angular velocity
            )

        if self.torso_enabled:
            torso_max = 0.25
            act_space['torso'] = spaces.Box(
                low=-torso_max,
                high=torso_max,
                shape=(1,),
            )
        
        if self.head_enabled:
            # Originally, discrete actions for 0.3, -0.3
            act_space['head'] = spaces.Box(
                low=-0.3,
                high=0.3,
                shape=(2,),
            )

        return spaces.Dict(act_space)

    def _observation(self):
        # TODO: we are going to change the observations anyway
        observations = AttrDict({
            'right': np.r_[np.array(self.tiago.arms['right'].arm_pose, dtype=np.float32), np.array(self.tiago.right_gripper_pos, dtype=np.float32)],
            'left': np.r_[np.array(self.tiago.arms['left'].arm_pose, dtype=np.float32), np.array(self.tiago.left_gripper_pos, dtype=np.float32)],
            # 'base_delta_pose': np.array(self.tiago.base.get_delta_pose(), dtype=np.float32),
            # 'base_velocity': np.array(self.tiago.base.get_velocity(), dtype=np.float32),
            'torso': np.array([self.tiago.torso.get_torso_extension()], dtype=np.float32),
            'head': np.array(self.tiago.head.get_head_joints(), dtype=np.float32),
        })

        for cam in self.cameras.keys():
            observations[f'{cam}_image'] = np.array(self.cameras[cam].get_img(), dtype=np.float32)
            observations[f'{cam}_depth'] = np.array(self.cameras[cam].get_depth(), dtype=np.float32)

        return observations

    # TODO: This would return the things that high-level policy needs, such as images
    def get_additional_states(self, obs):
        return {
            "dummy": 0
        }

    def reset(self, *args, **kwargs):
        print('Resetting...')
        self.start_time = None
        self.end_time = None
        self.steps = 0

        self.tiago.reset(*args, **kwargs)
        rospy.sleep(1)
        return self._observation(), {}
    
    def step(self, action):

        if action is not None:
            self.tiago.step(action)
        
        self.end_time = time.time()
        if self.start_time is not None:
            # print('Idle time:', 1/self.frequency - (self.end_time-self.start_time))
            rospy.sleep(max(0., 1/self.frequency - (self.end_time-self.start_time)))
        self.start_time = time.time()

        obs = self._observation()
        rew = 0
        terminate = False
        truncate = False
        info = {}

        self.steps += 1

        return obs, rew, terminate, truncate, info

# wrap the TiagoGym as a gym environment with new action space
class TiagoGymWrapper(gym.Env):
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space, self.action_keys = self.get_action_dict()

    def step(self, action):
        action_dict = {}
        idx = 0
        for key, value in self.action_keys.items():
            action_dict[key] = action[idx:idx+value.shape[0]]
            idx += value.shape[0]
        obs, rew, terminate, truncate, info = self.env.step(action_dict)
        done = terminate or truncate
        return obs, rew, done, info

    def get_action_dict(self):
        action_dict = self.env.action_space
        combine_shape = 0
        combine_low = np.array([])
        combine_high = np.array([])
        action_keys = list(action_dict.keys())
        for key, value in action_dict.items():
            combine_shape += value.shape[0]
            combine_low = np.concatenate((combine_low, value.low))
            combine_high = np.concatenate((combine_high, value.high))
        combine_shape = (combine_shape,)
        combined_box = gym.spaces.Box(low=combine_low, high=combine_high, shape=combine_shape, dtype=np.float32)
        return combined_box, action_dict

    def reset(self):
        obs, _ = self.env.reset()
        return obs

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def configure(self, *args, **kwargs):
        return self.env.configure(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.env, name)