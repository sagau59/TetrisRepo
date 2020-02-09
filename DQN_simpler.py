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

from simplerTetrisClass import SimplerTetris
from dqnClass import ReplayMemory, DQN
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torchsummary import summary


# if gpu is to be used
device = torch.device("cpu")

#game = Tetris(nb_rows=32,nb_cols=16)
game = SimplerTetris()

######################################################################
# Replay Memory
# -------------

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


######################################################################

resize = T.Compose([T.ToTensor()])

def get_screen():
    return resize(game.board).unsqueeze(0).to(device).type(torch.float)

######################################################################

def select_action(state,eps_threshold=1):
    if torch.rand(1)[0] > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def plot_durations(save_fig=False):
    plt.figure(2,figsize=(15,10))
    plt.clf()
    if len(episode_durations) < 2000:
        chunk = 10
    elif len(episode_durations) < 10000:
        chunk = 50
    else:
        chunk = 100
    duration_split = list(chunks(episode_durations,chunk))
    duration_split = [sum(i) / len(i) for i in duration_split]
    total_reward_split = list(chunks(total_reward_list,chunk))
    total_reward_split = [sum(i) / len(i) for i in total_reward_split]
    point_list_split = list(chunks(point_list,chunk))
    point_list_split = [sum(i) / len(i) for i in point_list_split]
    x = chunk*np.array((range(len(duration_split))))
    
    plt.subplot(3, 1, 1)
    plt.ylabel('Duration')
    plt.plot(x,duration_split)
    
    plt.subplot(3, 1, 2)
    plt.ylabel('Reward')
    plt.plot(x,total_reward_split)

    plt.subplot(3, 1, 3)
    plt.ylabel('Points')
    plt.plot(x,point_list_split)

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
policy_net = DQN(screen_height, screen_width, n_actions,layers=10).to(device)
target_net = DQN(screen_height, screen_width, n_actions,layers=10).to(device)
PATH = 'C:/Users/sagau/Google Drive/simplermodel2.pth'
optimizer = optim.Adam(policy_net.parameters(),lr=1e-4)

load_mode = False
if load_mode:
    model_dict = torch.load(PATH,map_location=torch.device('cpu'))
    i_episode = model_dict['epoch']
    optimizer.load_state_dict(model_dict['optimizer'])
    policy_net.load_state_dict(model_dict['state_dict'])
    target_net.load_state_dict(model_dict['state_dict'])
    episode_durations = model_dict['episode_durations']
    total_reward_list = model_dict['total_reward_list']
    point_list = model_dict['point_list']
    plot_durations()
else:     
    i_episode = 0
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    episode_durations = []
    total_reward_list = []
    point_list = []

memory = ReplayMemory(1000000)
steps_done = 0
BATCH_SIZE = 128
GAMMA = 0.9
EPS_START = 0.95
EPS_END = 0.1
EPS_DECAY = 10000
TARGET_UPDATE = 10
decay_rate = -(math.log(EPS_END) - math.log(EPS_START))/EPS_DECAY
starting_time = time.time()
num_episodes = 1000000

while i_episode < num_episodes:
    # Initialize the environment and state
    game = SimplerTetris()
    done = game.generate_block()
    state = get_screen()
    total_reward = 0
    if i_episode < EPS_DECAY:
        eps_threshold = EPS_START*math.exp(-decay_rate*i_episode)
    else:
        eps_threshold = EPS_END
    
    for t in count():
        # Select and perform an action
        steps_done += 1
        action = select_action(state,eps_threshold)
        game.play_active_block(action.item())
        active_min_row = game.get_min_row()/50.0
        if game.block_reached_end():
            sparsity = sum(np.where(game.board.sum(axis=1) > 1,1,0))/10
            down_reward = -1 + game.get_min_row()/10.0
            rows_cleared = game.clear_rows()
            done = game.generate_block()
            
        else:
            sparsity = 0
            down_reward = 0.2
            rows_cleared = 0
            done = False
            game.move_active_block_down()

        if done:
            reward = -5
        else:
            reward = 50*rows_cleared \
                + active_min_row \
                + down_reward \
                - sparsity
                
        total_reward += reward
        reward = torch.tensor([reward], device=device).type(torch.float)

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
            total_reward_list.append(total_reward)
            mod_i = 10
            if i_episode % mod_i == mod_i-1 and i_episode >= mod_i-1:
                print(game)
                plot_durations()
                print('Steps :',steps_done
                      ,'\nEpisode :',i_episode
                      ,'\n100 reward avg :',sum(total_reward_list[-mod_i:])/mod_i
                  ,'\n100 points avg :',sum(point_list[-mod_i:])/mod_i
                  ,'\nTotal Points :',sum(point_list)
                  ,'\nEpsilon :',eps_threshold
                  ,'\nTime elapsed :',(time.time()-starting_time)/60
                  ,'\nDuration :',t+1)
                model_dict = {
                        'epoch': i_episode
                        ,'state_dict': target_net.state_dict()
                        ,'optimizer': optimizer.state_dict()
                        ,'episode_durations':episode_durations
                        ,'total_reward_list':total_reward_list
                        ,'point_list':point_list
                }
                torch.save(model_dict, PATH)
                time.sleep(1)
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    i_episode += 1
    

######################################################################
# Play with model !
#sleep_time = 0.2
#game = Tetris()
#game.generate_block()
#print(game)
#time.sleep(sleep_time)
#state = get_screen()
#for t in count():
#    # Select and perform an action
#    action = select_action(state)
#    game.play_active_block(action.item())
#    print(game)
#    time.sleep(sleep_time)
#    if not game.block_reached_end():
#        game.move_active_block_down()
#        print(game)
#        time.sleep(sleep_time)
#        if not game.block_reached_end():
#            done = False
#        else:
#            done = game.generate_block()
#            print(game)
#            time.sleep(sleep_time)
#    else:
#        done = game.generate_block()
#        print(game)
#        time.sleep(sleep_time)
#    
#    if done:
#        break
    