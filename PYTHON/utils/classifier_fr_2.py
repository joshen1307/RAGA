import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
from tqdm import tqdm

from models import HMT
#from plots import *
from FRDEEP import FRDEEPN, FRDEEPF

def classification_procedure():
    model = torch.load('model_1.out')
    net = HMT()
    return net