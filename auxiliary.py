import torch
import numpy as np

def A(x):
    return (x[1] * torch.sin(np.pi * x[0]))

def psy_trial(x, net_out):
    return A(x) + x[0] * (1 - x[0]) * x[1] * (1 - x[1]) * net_out