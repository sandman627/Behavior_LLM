import os
import random
import numpy as np
import imageio

import torch as th

from PIL import Image
import PIL.ImageDraw as ImageDraw
import matplotlib.pyplot as plt  


import gymnasium as gym
from gymnasium.wrappers import TimeLimit

import metaworld

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

from behavior_framework import Front_Part

from behavior_policy import CustomCombinedExtractor, CustomCombinedMultiExtractor
import metaworld_env


# virtual display for headless mujoco
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()



'''
assembly-v2
basketball-v2
button-press-topdown-v2
button-press-topdown-wall-v2
button-press-v2
button-press-wall-v2
coffee-button-v2
coffee-pull-v2
coffee-push-v2
dial-turn-v2
disassemble-v2
door-close-v2
door-open-v2
drawer-close-v2
drawer-open-v2
faucet-open-v2
faucet-close-v2
hammer-v2
handle-press-side-v2
handle-press-v2
handle-pull-side-v2
handle-pull-v2
lever-pull-v2
peg-insert-side-v2
pick-place-wall-v2
pick-out-of-hole-v2
reach-v2
push-back-v2
push-v2
pick-place-v2
plate-slide-v2
plate-slide-side-v2
plate-slide-back-v2
plate-slide-back-side-v2
peg-unplug-side-v2
soccer-v2
stick-push-v2
stick-pull-v2
push-wall-v2
reach-wall-v2
shelf-place-v2
sweep-into-v2
sweep-v2
window-open-v2
window-close-v2
'''


def test_single_step_tasks():
    ## Get Behavior LLM
    test_model = Front_Part()

    ml45 = metaworld.ML45() # Construct the benchmark, sampling tasks
    

    for name, env_cls in ml45.train_classes.items():
        if name == "button-press-v2":
            print("name : ", name)
            env = env_cls(render_mode='rgb_array')
            task = random.choice([task for task in ml45.train_tasks if task.env_name == name])
            env.set_task(task)
            
            state, _ = env.reset()
            
            ## Get initial observation of environment
            initial_obs = env.render()
            Image.fromarray(initial_obs).save("test_"+name+".png")
            
            ## get skill sequence embedding
            output = test_model(initial_obs=initial_obs, instruction=name).numpy()
            print(f"Name: {name}, Output : {output.shape}")        
            
            env = TimeLimit(env, max_episode_steps=1000)
            env = metaworld_env.BehaviorWrapper(env, name, output)
            break
        

    # Multi Task Env Wrapper
    # env = metaworld_env.MultiTaskWrapper(training_envs)        
    
    ## Get Model
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
    )
    model = PPO(policy="MultiInputPolicy", env=env, policy_kwargs=policy_kwargs)
    
    # Load the model
    print("Loading Model")
    model.load(path="logs/bllm_rl_model_10200_steps.zip")
    

    # Test Inference
    print("Testing on Environment starts")
    images = []
    
    obs, _ = env.reset()
    images.append(env.render())
    
    max_timestep = 300
    done = False
    while not done and max_timestep:
        action, _ = model.predict(
            observation=obs
        )
        
        obs, reward, done, truncated, info = env.step(action)
        
        images.append(env.render())
        
        max_timestep -= 1
        
        if max_timestep % 100 ==0:
            print("timestep left : ", max_timestep)
        
        
    gif_filename = os.path.join("logs/gifs", name + '.' + "gif")
    gif_config = {
        'loop':1,
        'duration': 1/60        
    }
    imageio.mimsave(gif_filename, images, format='gif', **gif_config)    
    
    
    
    pass





  
  
  
  
if __name__=="__main__":
    print("Testing Behavior LLM!")
    
    test_single_step_tasks()
    
    pass
