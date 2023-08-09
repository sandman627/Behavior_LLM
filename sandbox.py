



max_it = 10
done = False
while not done and max_it:
    print(f"done: {done}, it: {max_it}")
    
    if max_it == 4:
        done = True
    
    max_it -= 1
    





exit()

import numpy as np
import gymnasium as gym


ll = [1,2,3,4]
print(ll)
ll.append(None)
print(ll)


# print(np.zeros((1, *tuple(None)), dtype='<U'))

# ttt = gym.spaces.Text(5)
# print(ttt.shape)
# print(ttt.sample())



dd = dict()
print(dd)
dd["a"] = 4
print(dd)
dd[1] = "b"
print(dd)

gg = {
    'a':1,
    'b':2,
    # 'c':3,
    'd':4
}

print(gg)


exit()

import numpy as np


a = [[[1,2],[3,4]]]
a = np.arange(72).reshape((2,4,9))
ar_a = np.array(a)
print(np.shape(ar_a))
print(ar_a)


arr = np.reshape(ar_a, (-1, 4, 18))
print(np.shape(arr))
print(arr)

print("padding test")
max_len = 6
ppp = max_len - np.shape(ar_a)[1]
pad_arr = np.pad(ar_a, pad_width=((0, 0),(0, ppp), (0,0)), mode='constant', constant_values=0)
print(ar_a)
print(pad_arr)
print(f"{np.shape(ar_a)} to {np.shape(pad_arr)}")





def test_f1(**kwargs):
    print("F 1")
    for k,v in kwargs.items():
        print(f"f1 {k}: {v}")
    return test_f2(**kwargs)

def test_f2(**kwargs):
    print("F 2")
    for k,v in kwargs.items():
        print(f"f2 {k}: {v}")
    return 0


tata = {
    "name" : "jjj",
    "age" : 12,
    "etc" : "dumdum"
}

test_f1(**tata)
print(type(tata))
print(type(**tata))

'''
import mujoco_py
import os

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sandman/.mujoco/mujoco210/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/.mujoco/mujoco210/bin
# export MUJOCO_PY_MUJOCO_PATH=/workspace/.mujoco/mujoco210


mj_path = mujoco_py.utils.discover_mujoco()
xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)

print(sim.data.qpos)
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

sim.step()
print(sim.data.qpos)
# [-2.09531783e-19  2.72130735e-05  6.14480786e-22 -3.45474715e-06
#   7.42993721e-06 -1.40711141e-04 -3.04253586e-04 -2.07559344e-04
#   8.50646247e-05 -3.45474715e-06  7.42993721e-06 -1.40711141e-04
#  -3.04253586e-04 -2.07559344e-04 -8.50646247e-05  1.11317030e-04
#  -7.03465386e-05 -2.22862221e-05 -1.11317030e-04  7.03465386e-05
#  -2.22862221e-05]

'''

'''
import metaworld
import random

ml10 = metaworld.ML10() # Construct the benchmark, sampling tasks

training_envs = []
for name, env_cls in ml10.train_classes.items():
    env = env_cls()
    task = random.choice([task for task in ml10.train_tasks
                        if task.env_name == name])
    env.set_task(task)
    training_envs.append(env)


print(training_envs)

for env in training_envs:
    obs = env.reset()  # Reset environment
    a = env.action_space.sample()  # Sample an action
    obs, reward, done, info = env.step(a)  # Step the environment with the sampled random action
    print("obs : ", obs)
    print("reward : ", reward)
    print("done : ", done)
    print("info : ", info)

'''
  
###############################


# from ai2thor.controller import Controller
# controller = Controller()
# event = controller.step("MoveAhead")


'''
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
                            # these are ordered dicts where the key : value
                            # is env_name : env_constructor

import numpy as np

door_open_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["door-open-v2-goal-observable"]
door_open_goal_hidden_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN["door-open-v2-goal-hidden"]

env = door_open_goal_hidden_cls()
env.reset()  # Reset environment
a = env.action_space.sample()  # Sample an action
obs, reward, done, info = env.step(a)  # Step the environment with the sampled random action
assert (obs[-3:] == np.zeros(3)).all() # goal will be zeroed out because env is HiddenGoal

# You can choose to initialize the random seed of the environment.
# The state of your rng will remain unaffected after the environment is constructed.
env1 = door_open_goal_observable_cls(seed=5)
env2 = door_open_goal_observable_cls(seed=5)

env1.reset()  # Reset environment
env2.reset()
a1 = env1.action_space.sample()  # Sample an action
a2 = env2.action_space.sample()
next_obs1, _, _, _ = env1.step(a1)  # Step the environment with the sampled random action

next_obs2, _, _, _ = env2.step(a2)
assert (next_obs1[-3:] == next_obs2[-3:]).all() # 2 envs initialized with the same seed will have the same goal
assert not (next_obs2[-3:] == np.zeros(3)).all()   # The env's are goal observable, meaning the goal is not zero'd out

env3 = door_open_goal_observable_cls(seed=10)  # Construct an environment with a different seed
env1.reset()  # Reset environment
env3.reset()
a1 = env1.action_space.sample()  # Sample an action
a3 = env3.action_space.sample()
next_obs1, _, _, _ = env1.step(a1)  # Step the environment with the sampled random action
next_obs3, _, _, _ = env3.step(a3)

assert not (next_obs1[-3:] == next_obs3[-3:]).all() # 2 envs initialized with different seeds will have different goals
assert not (next_obs1[-3:] == np.zeros(3)).all()   # The env's are goal observable, meaning the goal is not zero'd out
'''


# import metaworld
# import random

# print(metaworld.ML1.ENV_NAMES)  # Check out the available environments

# ml1 = metaworld.ML1('pick-place-v2') # Construct the benchmark, sampling tasks

# env = ml1.train_classes['pick-place-v2']()  # Create an environment with task `pick_place`
# task = ml1.train_tasks[0]
# env.set_task(task)  # Set task


# obs = env.reset()  # Reset environment
# a = env.action_space.sample()  # Sample an action
# obs, reward, done, truncated, info = env.step(a)  # Step the environment with the sampled random action



# print(f"action space : ", env.action_space)
# print(f"observation space : ", env.observation_space)

# print(f"task: {type(task)}\n\t", task)

# print(f"a : {type(a)}\n\t", a.shape, a.dtype, a)
# print(f"obs : {type(obs)}\n\t", obs.shape, obs.dtype, obs)


# print(f"reward : {type(reward)}\n\t", reward)
# print(f"done : {type(done)}\n\t", done)
# print(f"info : {type(info)}\n\t", info)
