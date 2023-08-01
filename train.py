import os
import logging
import random
import time
from tqdm import tqdm

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium.wrappers import TimeLimit

import metaworld

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

from parameters import device

from behavior_framework import Front_Part
from behavior_models import Image_Captioning_Model, Behavior_LLM, SkillEmbedder
from behavior_policy import CustomCombinedExtractor, CustomMultiInputActorCriticPolicy
import metaworld_env




# Class for MetaWorld
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
        image = self.env.render(offscreen, camera_name, resolution)
        return image



def render(env:gym.Env, offscreen=True, camera_name='corner3', resolution=(640, 480)):
        image = env.render(offscreen, camera_name, resolution)
        return image





if __name__=="__main__":
    print("Training Behavior LLM framework!!")
    
    ## Environment declare and setting
    print(metaworld.ML1.ENV_NAMES)  # Check out the available environments
    ml1 = metaworld.ML1('pick-place-v2') # Construct the benchmark, sampling tasks
    env = ml1.train_classes['pick-place-v2'](render_mode='rgb_array')  # Create an environment with task `pick_place`

    print("Task List : ", ml1.train_tasks[0])
    
    # task = random.choice(ml1.train_tasks)
    task = ml1.train_tasks[0]
    env.set_task(task)  # Set task
    obs, info = env.reset()
    
    print("Task : ", task)
    print("OBS : ", obs.shape)
    
    env.render()
    exit()
    env = TimeLimit(env, max_episode_steps=1000)
    
    
    
    ## Get Behavior LLM
    test_model = Front_Part()
    test_obs = "./datasets/drawer_and_ball.png"
    # test_obs = f"I am in the kitchen. There are egg, flour, milk in the refrigerator and salt, suger in the cabinet. I want to make breakfast. what should I do?\n"
    test_instruction = "In this task, you have to analye the situation and present step by step procedure to solve the problem."

    ## Get skill embedding sequence from Behavior Model
    output = test_model(instruction=test_instruction, initial_obs=test_obs)
    print("Output : ",output.shape, output)

    ## concate OBS and skill embedding
    env = metaworld_env.BehaviorWrapper(env, output)

    ## Get Model
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,

    )
    model = PPO(policy="MultiInputPolicy", env=env, policy_kwargs=policy_kwargs)
    
    # ## Inference on Env
    # action, _states = model.predict(torch.from_numpy(obs))
    # print("action : ", action)
    # print("state : ", _states)
    
    # obs, reward, done, truncated, info = env.step(action)  # Step the environment with the sampled random action

    # print(f"action space : ", env.action_space)
    # print(f"observation space : ", env.observation_space)

    # print(f"task: {type(task)}\n\t", task)

    # print(f"action : {type(action)}\n\t", action.shape, action.dtype, action)
    # print(f"obs : {type(obs)}\n\t", obs.shape, obs.dtype, obs)

    # print(f"reward : {type(reward)}\n\t", reward)
    # print(f"done : {type(done)}\n\t", done)
    # print(f"info : {type(info)}\n\t", info)
    
    
    ## Learn the Model
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path="./logs/",
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    eval_callback = EvalCallback(
        eval_env=env, 
        best_model_save_path="./logs/best_model",
        log_path="./logs/results",
        eval_freq=500
    )
    
    callback = CallbackList([checkpoint_callback, eval_callback])
    
    # model.learn(total_timesteps=10000, callback=callback, progress_bar=True)
