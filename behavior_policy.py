from typing import Any, Dict, List, Optional, Type, Union
import torch as th
from torch import nn
import torch.nn as nn

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, CombinedExtractor

from stable_baselines3.common.type_aliases import Schedule



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