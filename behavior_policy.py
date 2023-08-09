from typing import Any, Dict, List, Optional, Type, Union

import numpy as np

import torch as th
from torch import nn
import torch.nn as nn

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, CombinedExtractor

from stable_baselines3.common.type_aliases import Schedule

from behavior_framework import Front_Part
from behavior_models import Image_Captioning_Model





class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key== "obs":
                extractors[key] = nn.Linear(subspace.shape[0], 16)
                total_concat_size += 16
            elif key == "skill":
                # Run through a simple MLP
                extractors[key] = nn.Linear(subspace.shape[0], 16)
                total_concat_size += 16

        self.extractors = nn.ModuleDict(extractors)


        # Update the features dim manually
        self._features_dim = total_concat_size
        

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)






class CustomCombinedMultiExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=1)
        # print("CustomCombinedMultiExtractor")

        # hate this code
        self.high_instruction_dict = {
            0: 'reach-v2', 
            1: 'push-v2', 
            2: 'pick-place-v2', 
            3: 'door-open-v2', 
            4: 'drawer-close-v2', 
            5: 'button-press-topdown-v2', 
            6: 'peg-insert-side-v2', 
            7: 'window-open-v2', 
            8: 'sweep-v2', 
            9: 'basketball-v2'
        }

        # Encoder for initial Skill Embedding
        ## This part contains Larga Language Model. SO it is frozen. Might make errors
        self.skill_embedder = Front_Part()
        
        # Encoder for image observation
        self.image_encoder = Image_Captioning_Model() # this model outputs Text, so watch out for the use

        # obs embedder
        obs_subspace = observation_space.spaces["obs"]
        self.obs_embedder = nn.Linear(obs_subspace.shape[0], 16)
        
        # skill seq encoder
        self.skill_seq_encoder = nn.Linear(128, 16)


    def forward(self, observations) -> th.Tensor:
        print("SDFsdfsddfs")
        encoded_tensor_list = []

        # Get Skill Sequence Embedding
        if observations["initial_step"] == 0:
            print("obs high inst: ", self.high_instruction_dict["high_instruction"])
            
            self.skill_sequence_embeddings = self.skill_embedding_padding(
                self.skill_embedder(
                    observations["image_obs"], 
                    self.high_instruction_dict[observations["high_instruction"]]
                )
            )
            print("skill seq embeddings: ", self.skill_sequence_embeddings)
            exit()
        assert self.skill_sequence_embeddings != None, "No Skill embeddings!!!!"
                    
        # Get State
        encoded_tensor_list.append(self.obs_embedder(observations["obs"]))
        
        # Get Skill Embedding
        encoded_tensor_list.append(self.skill_seq_encoder(self.skill_sequence_embeddings))
        
        return th.cat(encoded_tensor_list, dim=1)
    
    def skill_embedding_padding(self, skill_embedding, max_seq_len:int=256):
        pad_size = max_seq_len - np.shape(skill_embedding)[1]
        padded_skill_emb = np.pad(skill_embedding, pad_width=((0,0), (0,pad_size), (0,0)), mode='constant', constant_values=0)
        return padded_skill_emb
    




class CustomMultiInputActorCriticPolicy(MultiInputActorCriticPolicy):
    def __init__(
        self, 
        observation_space: Dict, 
        action_space:spaces.Space, 
        lr_schedule: Schedule, 
        net_arch: List[int] | Dict[str, List[int]] | None = None, 
        activation_fn: type[nn.Module] = nn.Tanh, 
        ortho_init: bool = True, 
        use_sde: bool = False, 
        log_std_init: float = 0, 
        full_std: bool = True, 
        use_expln: bool = False, 
        squash_output: bool = False, 
        features_extractor_class: type[BaseFeaturesExtractor] = ..., 
        features_extractor_kwargs: Dict[str, Any] | None = None, 
        share_features_extractor: bool = True, 
        normalize_images: bool = True, 
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam, 
        optimizer_kwargs: Dict[str, Any] | None = None
    ):
        super().__init__(
            observation_space, 
            action_space, 
            lr_schedule, 
            net_arch, 
            activation_fn, 
            ortho_init, 
            use_sde, 
            log_std_init, 
            full_std, 
            use_expln, 
            squash_output, 
            features_extractor_class, 
            features_extractor_kwargs, 
            share_features_extractor, 
            normalize_images, 
            optimizer_class, 
            optimizer_kwargs
        )






if __name__=="__main__":
    print("Running Behavior Policy")
    
    # policy_kwargs = dict(
    #     features_extractor_class=CustomCNN,
    #     features_extractor_kwargs=dict(features_dim=128),
    # )
    # model = PPO("CnnPolicy", "BreakoutNoFrameskip-v4", policy_kwargs=policy_kwargs, verbose=1)
    # model.learn(1000)