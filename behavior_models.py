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
import torchvision

from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from transformers import BertConfig, BertTokenizer, BertModel
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import GenerationConfig

# from datasets import Image

from parameters import device


class Prompt_Model(nn.Module):
    """
    Prompt Model for merging Initial Observation and High-level Instruction
    Data Pre-processing such as Image or large dataset
    """
    def __init__(self) -> None:
        super().__init__()
        
        
    def forward(self, input):
        output = input
        return output



class Image_Captioning_Model(nn.Module):
    """
    Image Captioning Model for getting Natural Language Caption from given Image
    """
    def __init__(self) -> None:
        super().__init__()
        self.processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")
        
    def forward(self, initial_obs):
        raw_image = self.read_img(initial_obs)
        pixel_values = self.processor(images=raw_image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values=pixel_values, max_length=50)
        image_cap = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return image_cap

    def read_img(self, initial_obs):
        if os.path.isfile(initial_obs):
            raw_image = Image.open(initial_obs).convert("RGB")
        else:
            raw_image = initial_obs
        # raw_image = torchvision.io.read_image(initial_obs)
        return raw_image



class Behavior_LLM(nn.Module):
    """
    LLM for outputing sequence of skill description.
    Gets Initial Observation, Instruction for Input
    ex:
        Input: Image(metaworld with Drawer, Cube and Robot arm), Instruction(Hide the Cube)
        Output:
        1. Open Drawer
        2. Pick cube and Place it inside the Drawer
        3. Close Drawer    
    """
    
    def __init__(self) -> None:
        super().__init__()      
        
        self.tokenizer = LlamaTokenizer.from_pretrained("wordcab/llama-natural-instructions-7b")
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token_id = (0)    
            
        self.llm_model = LlamaForCausalLM.from_pretrained(
            "wordcab/llama-natural-instructions-7b",
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map=0,
        )
        
        self.llm_model.eval()
        if torch.__version__ >= "2":
            self.llm_model = torch.compile(self.llm_model)
        print("Model loaded")
        
        self.prompt_template = {
            "prompt": "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
            "response": "### Response:"    
        }

        self.generation_config = GenerationConfig(
            temperature=0.2,
            top_p=0.75,
            top_k=40,
            num_beams=4,
        )        
        
               
    def forward(self, instruction, observation):
        print("High Instruction : ", instruction)        
        print("Text observation : ", observation)

        prompt = self.generate_prompt(instruction, observation,)
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        input_ids = inputs["input_ids"].to(self.llm_model.device)
                
        gen_outputs = self.llm_model.generate(
            input_ids=input_ids,
            generation_config=self.generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=100,
        )
        s = gen_outputs.sequences[0]
        output = self.tokenizer.decode(s, skip_special_tokens=True)
        
        response = self.get_response(output)
        return response
               
        
    def generate_prompt(
        self,
        definition: str,
        inputs: str,
        targets: Union[None, str] = None,
    ) -> str:
        """Generate a prompt from instruction and input."""
        res = self.prompt_template["prompt"].format(
            instruction=definition, input=inputs
        )

        if targets:
            res = f"{res}{targets}"

        return res

    def get_response(self, output: str) -> str:
        """Get the response from the output."""
        return output.split(self.prompt_template["response"])[1].strip()
        


class SkillEmbedder(nn.Module):
    """
    Frozen LLM for getting Embedding features with Skill Description Sequence
    
    """
    def __init__(self) -> None:
        super().__init__()
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        
        # Freeze Skill Embedder, yet gradient still flows
        for p in self.model.parameters():
            p.requires_grad=False
        

    def forward(self, skill_descriptions):
        inputs = self.tokenizer(skill_descriptions, return_tensors='pt')
        outputs = self.model(**inputs)  # (batch_size, sequence_length, hidden_size)
        skill_embeddings = outputs[0]  # last hidden state of bert model
        return skill_embeddings
        