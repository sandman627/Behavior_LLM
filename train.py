import os
import logging
import random
import time
from tqdm import tqdm

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
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
from behavior_policy import CustomCombinedExtractor, CustomCombinedMultiExtractor
import metaworld_env

# virtual display for headless mujoco
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()


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




def train_single_step_tasks():
    ## Get Behavior LLM
    test_model = Front_Part()

    ml10 = metaworld.ML10() # Construct the benchmark, sampling tasks

    training_envs = []
    for name, env_cls in ml10.train_classes.items():        
        env = env_cls(render_mode='rgb_array')
        task = random.choice([task for task in ml10.train_tasks if task.env_name == name])
        env.set_task(task)
        
        state, _ = env.reset()
        
        ## Get initial observation of environment
        initial_obs = env.render()
        Image.fromarray(initial_obs).save("test.png")
        
        ## get skill sequence embedding
        output = test_model(initial_obs=initial_obs, instruction=name).numpy()
        print(f"Name: {name}, Output : {output.shape}")        
        
        env = TimeLimit(env, max_episode_steps=1000)
        env = metaworld_env.BehaviorWrapper(env, name, output)
        
        training_envs.append(env)

    
    # Multi Task Env Wrapper
    env = metaworld_env.MultiTaskWrapper(training_envs)        
    
    ## Get Model
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
    )
    model = PPO(policy="MultiInputPolicy", env=env, policy_kwargs=policy_kwargs)
        
    ## Learn the Model
    checkpoint_callback = CheckpointCallback(
        save_freq=100,
        save_path="./logs/",
        name_prefix="bllm_rl_model",
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
    model.learn(total_timesteps=10000, callback=callback, progress_bar=True)
    exit()
    
    # Final Test
    model.load(path="logs/best_model/best_model.zip")
    pass




high_instruction_dict = dict()


def train_multi_step_tasks():
    ml10 = metaworld.ML10() # Construct the benchmark, sampling tasks
    
    ## hate this code
    # high_instruction_dict = dict()
    # for name, env_cls in ml10.train_classes.items():  
    #     name_num = len(high_instruction_dict)
    #     high_instruction_dict[name_num] = name 
        
    # print(high_instruction_dict)
    # exit()



    training_envs = []
    name_num = 0
    for name, env_cls in ml10.train_classes.items():        
        env = env_cls(render_mode='rgb_array')
        task = random.choice([task for task in ml10.train_tasks if task.env_name == name])
        env.set_task(task)
        
        state, _ = env.reset()
        
        ##### for checking #######
        initial_obs = env.render()
        Image.fromarray(initial_obs).save(os.path.join("logs/renders", name + '.' + "png"))
        ##########################
        
        env = TimeLimit(env, max_episode_steps=1000)
        env = metaworld_env.BehaviorMultiWrapper(env, name_num)
        env.reset()
        
        training_envs.append(env)
        name_num += 1

    # Multi Task Env Wrapper
    env = metaworld_env.MultiTaskWrapper(training_envs)        
    
    
    ## Get Model
    print("Getting Policy and Model")
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedMultiExtractor,
    )
    model = PPO(policy="MultiInputPolicy", env=env, policy_kwargs=policy_kwargs)
    exit()
        
    
    ## Learn the Model
    checkpoint_callback = CheckpointCallback(
        save_freq=10,
        save_path="./logs/",
        name_prefix="bllm_rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    eval_callback = EvalCallback(
        eval_env=env, 
        best_model_save_path="./logs/best_model",
        log_path="./logs/results",
        eval_freq=50
    )
    callback = CallbackList([checkpoint_callback, eval_callback])
    exit()
    model.learn(total_timesteps=100, callback=callback, progress_bar=True)
    
    # Final Test
    model.load(path="logs/best_model/best_model.zip")
    pass




if __name__=="__main__":
    print("Training Behavior LLM framework!!")
    
    train_single_step_tasks()
    # train_multi_step_tasks()
    
    exit()
    
    
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
    
    img = env.render()
    Image.fromarray(img).save("test.png")
    print(img)
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
    
    model.learn(total_timesteps=10000, callback=callback, progress_bar=True)
