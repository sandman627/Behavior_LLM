import random
import inspect
import numpy as np

import gymnasium as gym
from gymnasium import spaces

import metaworld

from stable_baselines3.common.env_checker import check_env


# ml10 = metaworld.ML10() # Construct the benchmark, sampling tasks
# mt10 = metaworld.MT10() # Construct the benchmark, sampling tasks


class BehaviorWrapper(gym.Wrapper):
    def __init__(self, env, skill_embedding_sequence):
        super().__init__(env)
        self.env = env
        self.skill_embedding_sequence = skill_embedding_sequence.flatten()
        
        # print(self.skill_embedding_sequence.shape)
        # exit()
        
        self.observation_space = spaces.Dict({
            "obs": env.observation_space,
            "skill": spaces.Box(low=-np.inf, high=np.inf, shape=self.skill_embedding_sequence.shape)
        })
        
    def step(self, action):
        next_state, reward, done, truncated, info = self.env.step(action)
        # modify ...
        
        # concate
        next_state={
            "obs": next_state,
            "skill": self.skill_embedding_sequence
        }
        
        return next_state, reward, done, truncated, info
    
    def reset(self, seed=None, options=None):
        self.curr_path_length = 0
        obs, info = super().reset()
        self._prev_obs = obs[:18].copy()
        obs[18:36] = self._prev_obs
        obs = np.float64(obs)
        
        obs={
            "obs": obs,
            "skill": self.skill_embedding_sequence
        }        
        
        return obs, info
    
    
class MetaWorldEnv(gym.Env):
    def __init__(self, env_cls: str, env, task, sparse_reward=True, max_steps=200, deterministic=True, seed=777):
        random.seed(seed)
        self.env_cls = env_cls
        self.env = env      # environment
        self.task = task    # list of tasks
        self.sparse_reward = sparse_reward # goal conditioned reward
        self.max_steps = max_steps
        self.deterministic = deterministic  # deterministic task
        self.observation_space = self.env.observation_space
        self.action_space = gym.spaces.Box(
            self.env.action_space.low,
            self.env.action_space.high,
            dtype=np.float32
        )
    def reset(self):
        task = self.task[0] if self.deterministic else random.choice(self.task)
        self.env.set_task(task)
        self.timesteps = 0
        return self.env.reset()
    def step(self, action):
        action = np.clip(action, -1, 1)
        obs, rew, done, info = self.env.step(action)
        self.timesteps += 1
        # Done if success / max step
        rewS = 1 if info['success'] else 0
        info['rewS'] = rewS
        info['is_success'] = [True] if info['success'] else [False]
        if info['success'] or self.timesteps == self.max_steps:
            done = True
        return obs, rewS, done, info
    def render(self, offscreen=True, camera_name='corner3', resolution=(640, 480)):
        
        print("Sdfsdfsd   ", inspect.getfullargspec(self.env.render))
        image = self.env.render(offscreen, camera_name, resolution)
        return image
    
    
    
if __name__=="__main__":
    print("Checking Environment!")
    # env = CustomEnv()
    # # It will check your custom environment and output additional warnings if needed
    # check_env(env)
    
    # print(metaworld.ML1.ENV_NAMES)  # Check out the available environments
    ml1 = metaworld.ML1('pick-place-v2') # Construct the benchmark, sampling tasks
    env = ml1.train_classes['pick-place-v2'](render_mode='rgb_array')  # Create an environment with task `pick_place`
    
    task = ml1.train_tasks[0]
    env.set_task(task)  # Set taskyes
    
    obs, info = env.reset()
    
    testenv = MetaWorldEnv('pick-place-v2', env, task)
    print(inspect.getfullargspec(env.render))
    # testenv.render()
