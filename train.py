import os
import logging
import random
import time
from tqdm import tqdm

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from behavior_models import Behavior_LLM, SkillEmbedder
from behavior_framework import FrontPart



class Behavior_Model_Trainer(object):
    def __init__(self) -> None:
        pass
    
    



if __name__=="__main__":
    print("Training Behavior LLM framework!!")
    
    # Create Tensors to hold input and outputs.
    x = torch.linspace(-math.pi, math.pi, 2000)
    y = torch.sin(x)
    
    # Construct our model by instantiating the class defined above
    model = FrontPart()

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters (defined 
    # with torch.nn.Parameter) which are members of the model.
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)


    for t in range(2000):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        loss = criterion(y_pred, y)
        if t % 100 == 99:
            print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()