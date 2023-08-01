import os
import logging
import random
import time
from tqdm import tqdm

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import gymnasium
from gymnasium.wrappers import TimeLimit

import metaworld

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

from parameters import device

from behavior_models import Image_Captioning_Model, Behavior_LLM, SkillEmbedder
from behavior_policy import CustomCombinedExtractor, CustomMultiInputActorCriticPolicy
import metaworld_env



class Front_Part(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Front Part
        self.image_cap_model = Image_Captioning_Model()
        self.behavior_LLM = Behavior_LLM()
        self.skill_embedder = SkillEmbedder()
        
        # Output size : torch.Size([1, 10, 768])
        pass


    def forward(self, instruction, initial_obs):
        
        image_cap = self.image_cap_model(initial_obs)
        print("image cap : ", image_cap)
        given_situation = image_cap + "I want to hide the ball"

        skill_description_seq = self.behavior_LLM(instruction, given_situation)
        print("skill des seq : ", skill_description_seq)
        
        skill_embedding_seq = self.skill_embedder(skill_description_seq)
        print("skill emb seq : ", skill_embedding_seq)
        
        return skill_embedding_seq.numpy()



# class Rear_Part(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.
#         pass
            
#     def forward(self, input):
        
#         return output



def concate_obs_and_subgoal(obs:torch.tensor, subgoal:numpy.array):
    subgoal = torch.from_numpy(subgoal)
    
    return 



if __name__=="__main__":
    print("Running Behavior LLM framework!!")
    
    ## Environment declare and setting
    print(metaworld.ML1.ENV_NAMES)  # Check out the available environments
    ml1 = metaworld.ML1('pick-place-v2') # Construct the benchmark, sampling tasks
    env = ml1.train_classes['pick-place-v2'](render_mode='rgb_array')  # Create an environment with task `pick_place`
    env = TimeLimit(env, max_episode_steps=1000)

    
    # img = env.render()
    # print(type(img))
    # exit()
    
    print("Task List : ", ml1.train_tasks[0])
    # task = random.choice(ml1.train_tasks)
    task = ml1.train_tasks[0]
    env.set_task(task)  # Set task
    obs, info = env.reset()
    print("Task : ", task)
    print("OBS : ", obs.shape)
    
    
    ## Get Behavior LLM
    test_model = Front_Part()
    test_obs = "./datasets/drawer_and_ball.png"
    # test_obs = f"I am in the kitchen. There are egg, flour, milk in the refrigerator and salt, suger in the cabinet. I want to make breakfast. what should I do?\n"
    test_instruction = "In this task, you have to analye the situation and present step by step procedure to solve the problem."

    ## Get skill embedding sequence from Behavior Model
    output = test_model(instruction=test_instruction, initial_obs=test_obs)
    print("Output : ",output.shape, output)
    env = metaworld_env.BehaviorWrapper(env, output)



    exit()

    ## Combine Observation and Skill Embeddings
    



    ## Get Model
    policy_kwargs = dict(
        

    )
    model = PPO(policy=CustomMultiInputActorCriticPolicy, env=env, policy_kwargs=policy_kwargs)
    
        
    # model = PPO(policy="MlpPolicy", env=env)
    
    
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
    
    # is_not_complite = True
    # while is_not_complite:
    #     action, _states = model.predict(torch.from_numpy(obs))
    #     obs, rewards, dones, truncated, info = env.step(action)
    #     print("dones : ", dones)
    #     # env.render("human")