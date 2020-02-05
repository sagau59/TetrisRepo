# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 08:38:43 2020

@author: sagau
"""
import torch.nn.functional as F
import torch.nn as nn
import random
from collections import namedtuple

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
class DQN(nn.Module):

    def __init__(self, h, w, outputs, layers=3):
        super(DQN, self).__init__()
        self.layers = layers
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        
        if self.layers == 1:
            self.conv1 = nn.Conv2d(1, 128, kernel_size=2, stride=1)
            self.bn1 = nn.BatchNorm2d(128)
            convw = conv2d_size_out(w,2,1)
            convh = conv2d_size_out(h,2,1)
            linear_input_size = convw * convh * 128
            self.head = nn.Linear(linear_input_size, outputs)
        
        elif self.layers == 2:
            self.conv1 = nn.Conv2d(1, 16, kernel_size=2, stride=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1)
            self.bn2 = nn.BatchNorm2d(32)
            convw = conv2d_size_out(conv2d_size_out(w,2,1),2,1)
            convh = conv2d_size_out(conv2d_size_out(h,2,1),2,1)
            linear_input_size = convw * convh * 32
            self.head = nn.Linear(linear_input_size, outputs)
        
        elif self.layers == 3:
            self.conv1 = nn.Conv2d(1, 16, kernel_size=2, stride=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.conv3 = nn.Conv2d(32, 32, kernel_size=2, stride=1)
            self.bn3 = nn.BatchNorm2d(32)
            convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w,2,1),2,1),2,1)
            convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h,2,1),2,1),2,1)
            linear_input_size = convw * convh * 32
            self.head = nn.Linear(linear_input_size, outputs)
            
        elif self.layers == 4:
            # https://github.com/jaybutera/tetrisRL
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
            self.bn2 = nn.BatchNorm2d(32)
            self.lin1 = nn.Linear(480, 256)
            self.head = nn.Linear(256, outputs)

    def forward(self, x):
        if self.layers == 1:
            x = F.relu(self.bn1(self.conv1(x)))
        
        elif self.layers == 2:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
        
        elif self.layers == 3:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
        
        elif self.layers == 4:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.lin1(x.view(x.size(0), -1)))
            
        return self.head(x.view(x.size(0), -1))
    
    