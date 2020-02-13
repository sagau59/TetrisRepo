

# Source code : https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

from tetrisClass import Tetris
from dqnClass import ReplayMemory, DQN

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torchsummary import summary


# if gpu is to be used
device = torch.device("cpu")

#game = Tetris(nb_rows=32,nb_cols=16)
game = Tetris(nb_rows=8,nb_cols=6)

######################################################################
# Replay Memory
# -------------

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


######################################################################

resize = T.Compose([T.ToTensor()])

def get_screen():
    # GetScreen output 1*1*15*10 au lieu de 1*3*40*90
    return resize(game.board).unsqueeze(0).to(device).type(torch.float)


######################################################################

def select_action(state,eps_threshold=1):
    sample = random.random()
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


######################################################################
# Training loop

######################################################################

init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space

n_actions = 4
#PATH = 'C:/Users/sagau/Desktop/Kaggle/TetrisRepo/models/model1_2.pth'
policy_net = DQN(screen_height, screen_width, n_actions,layers=20)
policy_net.eval()
policy_net = DQN(screen_height, screen_width, n_actions,layers=20).to(device)

PATH = 'C:/Users/sagau/Google Drive/smaller1.pth'
model_dict = torch.load(PATH,map_location=torch.device('cpu'))
policy_net.load_state_dict(model_dict['state_dict'])
    
######################################################################
# Play with model !
sleep_time = 0.2
game = Tetris(nb_rows=8,nb_cols=6)
done = game.generate_block(choice=3)
rows = 0
for t in count():
#    for t in range(200):
    state = get_screen()
    action = select_action(state,0)
#    print(action.item())
    game.play_active_block(action.item())
    if game.block_reached_end():
        rows = rows + game.clear_rows()
        done = game.generate_block(choice=3)
    else:
        game.move_active_block_down()
        done = False
    print(game)
    plt.pause(sleep_time)
    if done:
        print(t,rows)
        break


