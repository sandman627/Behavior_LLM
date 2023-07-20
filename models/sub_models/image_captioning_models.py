from typing import Union

import os
import logging
import random
import time
from tqdm import tqdm
from PIL import Image

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BertConfig, BertTokenizer, BertModel
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import GenerationConfig

from lavis.models import load_model_and_preprocess


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BLIP_Captioning_Model(nn.Module):
    """
    Image Captioning Model for getting Natural Language Caption from given Image
    """
    def __init__(self) -> None:
        super().__init__()
        self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)
        
        
    def forward(self, initial_obs):
        if os.path.isfile(initial_obs):
            raw_image = Image.open(initial_obs).convert("RGB")
        else:
            raw_image = initial_obs
            
        image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        
        image_cap = self.model.generate({"image": image})
        return image_cap

