import random
import torch


x = torch.normal(mean=0., std=1., size=(2,3))
print(x)
print(x.shape)