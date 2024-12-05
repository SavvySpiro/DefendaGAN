import visualize as vis
import utils
import network
import comparison
import torch
import dataset
import config
from torch import nn, optim
from torch.autograd.variable import Variable



model_path = "/home/hanna/Documents/DefendaGAN/generator_dandy-disco-321.pth"
attacker = utils.load_model(network.GeneratorNetGroup(1), model_path)

print("succ")