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

import parameters
from parameters import device


class FrontPart(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        # Front Part
        self.image_cap_model = Image_Captioning_Model()
        self.behavior_LLM = Behavior_LLM()
        self.skill_embedder = SkillEmbedder()
        
        # Rear Part
        
        pass

    def forward(self, instruction, initial_obs):
        
        image_cap = self.image_cap_model(initial_obs)
        print("image cap : ", image_cap)
        given_situation = image_cap + "I want to hide the ball"

        skill_description_seq = self.behavior_LLM(instruction, given_situation)
        print("skill des seq : ", skill_description_seq)
        
        skill_embedding_seq = self.skill_embedder(skill_description_seq)
        print("skill emb seq : ", skill_embedding_seq)
        return skill_embedding_seq
