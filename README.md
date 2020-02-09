# TetrisRepo
This repo has been a journey to me for understanding reinforcement learning. I wanted to create a robot able to master the Tetris game, being a Tetris fan myself.

I started by coding my own Tetris class, which is in *tetrisClass.py*. Then, I looked up [this](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) tutorial on deep Q-Learning with PyTorch and decided to give it a shot with my own class. I used Google Colab for the free GPU and even with that calculation power, I had trouble making my bot learn to clear the game.

I then decided to make the game easier so I could figure out if my bot could win a game. I then coded the SimplerTetris class, which is a game with 7 rows and 4 columns and only blocks that are 2 tiles long. The code that beats the game is in *DQN_simpler.py*. After 5000 episodes, this agent looks like this :

![alt text](https://github.com/sagau59/TetrisRepo/blob/master/images/simpler.gif)

