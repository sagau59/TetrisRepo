# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 21:04:39 2020

@author: sagau
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 16:08:49 2020

@author: sagau
"""

# Source code : https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

from tetrisClass import Tetris
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
game = Tetris()

######################################################################
# Replay Memory
# -------------

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


######################################################################
# Trois réseaux : fuckall
# Deux réseaux : fuckall
class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        STRIDE = 1
        KERNEL_SIZE = 2
        self.conv1 = nn.Conv2d(1, 16, kernel_size=KERNEL_SIZE, stride=STRIDE)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=KERNEL_SIZE, stride=STRIDE)
        self.bn2 = nn.BatchNorm2d(32)
#        self.conv3 = nn.Conv2d(32, 32, kernel_size=KERNEL_SIZE, stride=STRIDE)
#        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size = KERNEL_SIZE, stride = STRIDE):
            return (size - (kernel_size - 1) - 1) // stride  + 1
#        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
#        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        convw = conv2d_size_out(conv2d_size_out(w))
        convh = conv2d_size_out(conv2d_size_out(h))
#        convw = conv2d_size_out(w)
#        convh = conv2d_size_out(h)
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
#        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


######################################################################

resize = T.Compose([T.ToPILImage(),
                    T.ToTensor()])

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

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def plot_durations(save_fig=False):
    plt.figure(2)
    plt.clf()
    if len(episode_durations) < 2000:
        chunk = 10
    elif len(episode_durations) < 10000:
        chunk = 50
    else:
        chunk = 100
    duration_split = list(chunks(episode_durations,chunk))
    duration_split = [sum(i) / len(i) for i in duration_split]
    x = chunk*np.array((range(len(duration_split))))
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(x,duration_split)
    # Take 1000 episode averages and plot them too

    if save_fig:
        plt.savefig('DQN_me4_state_screen')
    else:
        plt.pause(0.001)
        plt.show()

######################################################################
# Training loop

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


######################################################################

init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space

n_actions = 4
policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
print(summary(policy_net, input_size=(1, screen_height, screen_width)))
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(300000)

episode_durations = []
point_list = []
steps_done = 0
BATCH_SIZE = 128
#GAMMA = 0.999
#GAMMA = 0.9
GAMMA = 0.99
EPS_START = 0.95
EPS_END = 0.1
EPS_DECAY = 10000
TARGET_UPDATE = 10
starting_time = time.time()
num_episodes = 1000000
i_episode = 0

while i_episode < num_episodes:
    # Initialize the environment and state
    game = Tetris()
    if i_episode < 50000:
        done = game.generate_block(choice=3) # Juste carré début
    else:
        done = game.generate_block()
    state = get_screen()
    total_reward = 0
    if i_episode < EPS_DECAY:
        eps_threshold = EPS_START - (EPS_START - EPS_END) * (i_episode / EPS_DECAY)
    else:
        eps_threshold = EPS_END
    
    for t in count():
        # Select and perform an action
        steps_done += 1
        action = select_action(state,eps_threshold)
        game.play_active_block(action.item())
        if game.block_reached_end():
            rows_cleared = game.clear_rows()
            if i_episode < 50000:
                done = game.generate_block(choice=3)
            else:
                done = game.generate_block()
        else:
            game.move_active_block_down()
            rows_cleared = 0
            done = False
            
        if done:
            reward = -10
        else:
            reward = 1 + rows_cleared - game.get_min_row()/10
        total_reward += reward
        reward = torch.tensor([reward], device=device).type(torch.float)

        # Observe new state
#        if not done:
#            next_state = get_screen()
#        else:
#            next_state = None
        next_state = get_screen()
        
        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            point_list.append(game.points)
            plot_durations()
            print('Steps :',steps_done
                  ,'\nEpisode :',i_episode
                  ,'\nTotal reward :',total_reward
                  ,'\nPoints :',game.points
                  ,'\nTotal Points :',sum(point_list)
                  ,'\nEpsilon :',eps_threshold
                  ,'\nTime elapsed :',(time.time()-starting_time)/60
                  ,'\nDuration :',t+1)
            if i_episode % 10 == 0:
                print(game)
#            print(game)
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    i_episode += 1

print('Complete')

######################################################################
# Play with model !
sleep_time = 0.2
def play_agent():
    game = Tetris()
    game.generate_block()
    print(game)
    time.sleep(sleep_time)
    state = get_screen()
    for t in count():
        # Select and perform an action
        action = select_action(state)
        game.play_active_block(action.item())
        print(game)
        time.sleep(sleep_time)
        if not game.block_reached_end():
            game.move_active_block_down()
            print(game)
            time.sleep(sleep_time)
            if not game.block_reached_end():
                done = False
            else:
                done = game.generate_block()
                print(game)
                time.sleep(sleep_time)
        else:
            done = game.generate_block()
            print(game)
            time.sleep(sleep_time)
        
        if done:
            break
    
#play_agent()