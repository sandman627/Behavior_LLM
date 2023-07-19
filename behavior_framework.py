import os
import logging
import random
import time
from tqdm import tqdm

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from behavior_models import Image_Captioning_Model, Behavior_LLM, SkillEmbedder


class FrontPart(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.image_cap_model = Image_Captioning_Model()
        self.behavior_LLM = Behavior_LLM()
        self.skill_embedder = SkillEmbedder()

    def forward(self, instruction, initial_obs):
        
        image_cap = self.image_cap_model(initial_obs)
        print("image cap : ", image_cap)
        given_situation = image_cap + "I want to hide the ball"

        skill_description_seq = self.behavior_LLM(instruction, given_situation)
        print("skill des seq : ", skill_description_seq)
        
        skill_embedding_seq = self.skill_embedder(skill_description_seq)
        print("skill emb seq : ", skill_embedding_seq)
        return skill_embedding_seq



if __name__=="__main__":
    print("Running Behavior LLM framework!!")
    
    test_model = FrontPart()
    
    test_obs = "./datasets/drawer_and_ball.png"
    # test_obs = f"I am in the kitchen. There are egg, flour, milk in the refrigerator and salt, suger in the cabinet. I want to make breakfast. what should I do?\n"
    
    test_instruction = "In this task, you have to analye the situation and present step by step procedure to solve the problem."
        
    output = test_model(instruction=test_instruction, initial_obs=test_obs)
    print(output)