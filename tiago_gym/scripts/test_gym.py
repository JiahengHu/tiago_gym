import time
import numpy as np
import cv2
from tiago_gym.tiago.tiago_gym import TiagoGym
from tiago_gym.wrappers.env_wrappers import TiagoPointHeadWrapper, DisplayImageWrapper, SimAndRealUncertaintyAwareWrapper

env = TiagoGym(
        frequency=1,
        head_enabled=True,
        base_enabled=False,
        torso_enabled=False,
        right_arm_enabled=False,
        left_arm_enabled=False,
        right_gripper_type='robotiq2F-140',
        left_gripper_type='robotiq2F-85'
    )

exit(0)
# env = DisplayImageWrapper(TiagoPointHeadWrapper(env))
env = DisplayImageWrapper(SimAndRealUncertaintyAwareWrapper(env))
env.reset()
input()
for i in range(100):
    action = np.random.randint(1, 5)
    # action = int(input('enter:'))
    obs = env.step(action)[0]

    cv2.imshow('obs', obs['tiago_head_image']/255)
    cv2.waitKey(1)

    if i%10:
        env.reset()
        input()

exit(0)

### RL stuff
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO, DQN
from l2l.modules.real_feature_extractor import RealFeatureExtractor
from l2l.utils.general_utils import AttrDict

# model_config=AttrDict(
#     policy="MultiInputPolicy",
#     policy_kwargs=dict(
#         features_extractor_class=RealFeatureExtractor,
#         # features_extractor_kwargs=dict(features_dim=128),
#     ),
#     n_steps=50,
#     gamma=0.95,
#     learning_rate=1e-4,
#     # ent_coef=0.03,
#     batch_size=50,
#     # clip_range=0.05,
#     verbose=1
# )

# model = PPO(**model_config, env=Monitor(env))

model_config=AttrDict(
    policy="MultiInputPolicy",
    policy_kwargs=dict(
        features_extractor_class=RealFeatureExtractor,
        # features_extractor_kwargs=dict(features_dim=128),
    ),
    buffer_size=int(3e4),
    gamma=0.95,
    learning_rate=1e-4,
    learning_starts=50,
    target_update_interval=200,
    verbose=1
)

model = DQN(**model_config, env=Monitor(env))

model.learn(15e3)