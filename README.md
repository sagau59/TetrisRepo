# TetrisRepo
This repo has been a journey to me for understanding reinforcement learning. I wanted to create a robot able to master the Tetris game, being a Tetris fan myself.

I started by coding my own Tetris class, which is in *tetrisClass.py*. Then, I looked up [this](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) tutorial on deep Q-Learning with PyTorch and decided to give it a shot with my own class. I used Google Colab for the free GPU and even with that calculation power, I had trouble making my bot learn to clear the game.

# Simpler Tetris

I then decided to make the game easier so I could figure out if my bot could win a game. I then coded the SimplerTetris class, which is a game with 7 rows and 4 columns and only blocks that are 2 tiles long. The code that beats the game is in *DQN_simpler.py*. After 5,000 episodes, this agent looks like this :

![alt text](https://github.com/sagau59/TetrisRepo/blob/master/images/simpler.gif)

You can play with the model with *playSimplerModel.py*.

# Smaller Tetris - squares only

With a model able to beat my simpler game, I then tried to beat a smaller version of the game. Instead of a normal 15 rows and 10 columns, I tried to beat the game with 8 rows and 6 columns. The make the training easier, I started training the agent with only the square blocks. When the agent beats the game, I will build another agent based on this agent using transfer learning to beat the game. The code that beats the game is in *DQN_smaller.py* and *playSmallerModel.py* plays the game. After 30,000 episodes only with squares, the agent looks like this :

![alt text](https://github.com/sagau59/TetrisRepo/blob/master/images/smaller.gif)

# Smaller Tetris - all blocks

After training my model to beat the game only with squares, I tried to use my pre-trained model to learn to beat the smaller game with all blocks. After 350,000 episodes for 72 hours of training on a GPU, but model still did not beat the game, only achieving a couple of points in average.

![alt text](https://github.com/sagau59/TetrisRepo/blob/master/images/transfer.gif)
