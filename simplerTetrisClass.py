# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 20:52:35 2020

@author: sagau
"""

import numpy as np
import random
import math

NB_ROWS = 7
NB_COLS = 4

class SimplerTetris:
    def __init__(self,nb_rows=NB_ROWS,nb_cols=NB_COLS):
        self.nb_rows = nb_rows
        self.nb_cols = nb_cols
        self.board = np.zeros((self.nb_rows,self.nb_cols)).astype(int)
        self.starting_col = int(math.floor((self.nb_cols-1)/2))
        self.active_block = None
        self.active_block_zone_xy = None
        self.lists = [[0,0],[0,1],[1,0],[1,1]]
        self.points = 0
        self.duration = 0
        #Rotation works like this :
        # The shape of the block is put on a square of 2*2 for square
        # 4*4 for the line and 3*3 for the other blocks.
        # The shape as 1 and the other as 0.
        # The square is rotated with np.rot90
        # The 0 is removed for the square
        # Then I compute the difference betwwen the orignal 4 and 
        # The remaining 4. I add this difference to the active block
    
    def __str__(self):  
        board = np.where(self.board == 1,'X','-')
        return '_'*(2*self.nb_rows + 2) + '\n' +\
                  '\n'.join(['| ' + '  '.join(map(str,board[i])) + ' |' for i in range(self.nb_rows)]) +\
                  '\n' + '_'*(2*self.nb_rows + 2)

    
    def generate_block(self):
        #choice = random.randint(0,5)
        #choice = 3
        #Line
        self.active_block = [[i,self.starting_col] for i in range(2)]
        self.active_block_zone_xy = [1,3]
        new_board = np.copy(self.board)
        
        for ele in self.active_block:
            new_board[ele[0]][ele[1]] += 1
        
        #Check if game is lost
        if 2 in new_board:
            return True
        else:
            self.board = new_board
            return False  

# Plus lent    
#    def generate_block2(self):
#        self.active_block = [[i,self.starting_col] for i in range(2)]
#        self.active_block_zone = np.array([[0,1]
#                                          ,[0,1]])
#        self.active_block_zone_x = np.array([0,1])
#        self.active_block_zone_y = np.array([self.starting_col,self.starting_col])
#        
##        self.active_block_zone_x = np.array([0,0,1,1])
##        self.active_block_zone_y = np.array([self.starting_col,self.starting_col+1,self.starting_col,self.starting_col+1])
#        
#        self.board[self.active_block_zone_x,self.active_block_zone_y] +=1
#        #Check if game is lost
#        if 2 in self.board:
#            self.board[self.active_block_zone_x,self.active_block_zone_y] -=1
#            return True
#        else:
#            return False  
        
        
    def move_active_block_down(self):
        #Remove block where it was
        for ele in self.active_block:
            self.board[ele[0]][ele[1]] += -1
        #Compute where block will go
        self.active_block = [[ele[0]+1,ele[1]] for ele in self.active_block]
        #Move block
        for ele in self.active_block:
            self.board[ele[0]][ele[1]] += 1
      
        
    def playable_move(self):
        moves = [0] # 0 : do nothing
        board_without_active = np.copy(self.board)
        for ele in self.active_block:
            board_without_active[ele[0]][ele[1]] += -1
        #Check left
        if min([i[1] for i in self.active_block]) > 0:
            if 1 not in [board_without_active[ele[0],ele[1]-1] for ele in self.active_block]:
                moves.append(1)
        #Check right
        if max([i[1] for i in self.active_block]) < self.nb_cols-1:
            if 1 not in [board_without_active[ele[0],ele[1]+1] for ele in self.active_block]:
                moves.append(2)
        #Todo : Check rotation
        active_block_rotated = np.copy(self.active_block)
        if self.active_block_zone_xy == [1,3]:
            active_block_rotated[0] += np.array(self.lists[3]) - np.array(self.lists[1])
            active_block_rotated[1] += np.array(self.lists[2]) - np.array(self.lists[3])
        elif self.active_block_zone_xy == [3,2]:
            active_block_rotated[0] += np.array(self.lists[2]) - np.array(self.lists[3])
            active_block_rotated[1] += np.array(self.lists[0]) - np.array(self.lists[2])
        elif self.active_block_zone_xy == [2,0]:
            active_block_rotated[0] += np.array(self.lists[0]) - np.array(self.lists[2])
            active_block_rotated[1] += np.array(self.lists[1]) - np.array(self.lists[0])
        elif self.active_block_zone_xy == [0,1]:
            active_block_rotated[0] += np.array(self.lists[1]) - np.array(self.lists[0])
            active_block_rotated[1] += np.array(self.lists[3]) - np.array(self.lists[1])
            
        if max([i[0] for i in active_block_rotated]) < self.nb_rows \
                and max([i[1] for i in active_block_rotated]) < self.nb_cols \
                and min([i[1] for i in active_block_rotated]) > 0:
            if 1 not in [board_without_active[ele[0],ele[1]] for ele in active_block_rotated]:
                moves.append(3)
        
        return moves
        
    
    def play_active_block(self,move):
        #Check if block can go to new place
        possible_moves = self.playable_move()
        if move in possible_moves:
            #Remove block where it was
            for ele in self.active_block:
                self.board[ele[0]][ele[1]] += -1
            #Compute where block will go   
            if move == 1: #Left
                self.active_block = [[ele[0],ele[1]-1] for ele in self.active_block]
            elif move == 2: #Right            
                self.active_block = [[ele[0],ele[1]+1] for ele in self.active_block]
            elif move == 3: #Rotate
                if self.active_block_zone_xy == [1,3]:
                    self.active_block[0] += np.array(self.lists[3]) - np.array(self.lists[1])
                    self.active_block[1] += np.array(self.lists[2]) - np.array(self.lists[3])
                    self.active_block_zone_xy = [3,2]
                elif self.active_block_zone_xy == [3,2]:
                    self.active_block[0] += np.array(self.lists[2]) - np.array(self.lists[3])
                    self.active_block[1] += np.array(self.lists[0]) - np.array(self.lists[2])
                    self.active_block_zone_xy = [2,0]
                elif self.active_block_zone_xy == [2,0]:
                    self.active_block[0] += np.array(self.lists[0]) - np.array(self.lists[2])
                    self.active_block[1] += np.array(self.lists[1]) - np.array(self.lists[0])
                    self.active_block_zone_xy = [0,1]
                elif self.active_block_zone_xy == [0,1]:
                    self.active_block[0] += np.array(self.lists[1]) - np.array(self.lists[0])
                    self.active_block[1] += np.array(self.lists[3]) - np.array(self.lists[1])
                    self.active_block_zone_xy = [1,3] 
            #Move block
            for ele in self.active_block:
                self.board[ele[0]][ele[1]] += 1

    
    def get_bottom_elements(self):
        active_cols = list(set([i[1] for i in self.active_block]))
        return [[max([i[0] for i in self.active_block if i[1] == j]),j] for j in active_cols]
    
    def get_min_row(self):
        bottom_elements = self.get_bottom_elements()
        return min([i[0] for i in bottom_elements])
    
    def block_reached_end(self):
        bottom_elements = self.get_bottom_elements()
        #End of board
        if max([i[0] for i in bottom_elements]) == self.nb_rows-1:
            return True
        #Block next to active block
        elif 1 in [self.board[j[0]][j[1]] for j in [[i[0]+1,i[1]] for i in bottom_elements]]:
            return True
        #Not the end of board
        else:
            return False
    
    
    def clear_rows(self):
        sum_block_rows = self.board.sum(axis=1)
        rows_cleared = 0
        for i in range(self.nb_rows):
            if sum_block_rows[i] == self.nb_cols:
                self.board = np.vstack((
                                        np.zeros(self.nb_cols).astype(int)
                                        ,self.board[:i]
                                        ,self.board[i+1:]
                                      ))
                self.points += 1
                rows_cleared += 1
        return rows_cleared
    
    
    def play_game(self,verbose=0):
        while self.generate_block() == False:
            if verbose > 0:
                print(self)
            while True:
                if verbose > 1:
                    print(self)
                self.play_active_block(random.randint(0,3))
                if self.block_reached_end():
                    break
                self.move_active_block_down()
                if self.block_reached_end():
                    break
                    
            self.clear_rows()
        print('Number of points : ' + str(self.points))
     
        
    def play_n_moves(self,n=10):
        nb_turns = 0
        self.generate_block()
        while True and nb_turns <= n:
            move = random.randint(0,3)
            print('Move : ' + str(move))
            self.play_active_block(move)
            nb_turns = nb_turns + 1
            if self.block_reached_end():
                break
            self.move_active_block_down()
            if self.block_reached_end():
                break
        self.clear_rows()
        print('Number of points : ' + str(self.points))
    
    def play_manually(self):
        while self.generate_block() == False:
            while True:
                print(self)
                print('Move :\n0 : Nothing\n1 : Left\n2 : Right\n3 : Rotation')
                self.duration += 1
                move = input()
                if move == '':
                    move = 0
                elif move == 9:
                    break
                else:
                    
                    move = int(move)
                self.play_active_block(move)
                if self.block_reached_end():
                    break
                self.move_active_block_down()
#                if self.block_reached_end():
#                    break
            self.clear_rows()
            if move == 9:
                break
        print('Number of points : ' + str(self.points))
            
#random.seed(10)
if __name__ == '__main__':
    game = SimplerTetris()
    game.play_manually()
#game.play_n_moves(10)
#game.playable_move()
##print(game)
##game.move_active_block()
##print(game)
##game.block_reached_end()
#
#game.play_game(verbose=2)
#print(game)
#game.generate_block()



#game = SimplerTetris()
#game.generate_block()
#game.playable_move()




