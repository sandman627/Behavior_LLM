import os
import torch

import metaworld

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback


import behavior_framework
from behavior_policy import CustomCombinedExtractor

# virtual display for headless mujoco
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()


# Check GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())










def single_inference(model_loc:str="logs/best_model/best_model.zip"):
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
    )
    model = PPO(policy="MultiInputPolicy", env=env, policy_kwargs=policy_kwargs)
    
    pass







if __name__=="__main__":
    print("Running Behavior LLM prototype!!")
    single_inference()