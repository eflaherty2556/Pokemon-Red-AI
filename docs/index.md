# Pokemon-Red-AI

## What is this repository?

This project is a reinforcement-learning-based AI that plays Pokemon Red for the GameBoy. 

<img src="https://raw.githubusercontent.com/JFlaherty347/Pokemon-Red-AI/master/images/overworld.png" alt="game" width="200">
<img src="https://raw.githubusercontent.com/JFlaherty347/Pokemon-Red-AI/master/images/battle.png" alt="game" width="200">

## Motivation

One key point of interest around this project revolves around Pokemon Red being an RPG. In contrast to games which AI has mastered like tic-tac-toe or breakout, the state space of Pokemon Red is massive. With the main story taking an average player over 25 hours to complete, the game is certainly a daunting task for current approaches to AI. In the face of this challenge, we hope to gain a deeper understanding of reinforcement learning and potentially discover new strategies for tackling complex AI problems.

## Experiments

### A2C vs DQN

In order to achieve the best performance, we wanted to look into a few of the most popular reinforcement learning algorithms. One major advantage of A2C was that it supported vectorized environments to allow for much faster computation, whereas DQN cannot due to how Q-values are calculated. This ultimately meant that the amount of time for a single timestep of training to occur was far shorter for A2C than DQN. Ultimately, it was concluded that given an equal amount of time, A2C will perform far better because it can train for more timesteps than DQN for any period. It is difficult to determine whether DQN would perform better given an equal amount of timesteps, however, the amount of training time that it would have taken to reach and level of competency for the agent would have taken several days. 

### Image-based observations vs Ram-based observations

Two observation options are available for the environment: image-based and ram-based. When comparing the two we found a much faster rate of improvement with RAM observations. To achieve similar performance with image observations, the input image could be pre-processed to simplify the data into something simpler. Even then, with the amount of time dedicated to creating a better performance with image observations, where performance would probably at best be slightly better than RAM, other more important improvements could be made. Hence, we have decided to use RAM observations for now. This in effect made it easier to decide between CNN and MLP policies for the agent, since CNN performs much with images.

### With Pretraining vs Without Pretraining

Currently, we are looking into whether pretraining provides a noticeable improvement in performance for the agent. With non-expert human demonstrations, we hope to see the agent demonstrate better pathing to new locations. Imitation learning also may lead to much faster improvement since the agent will start by examining a somewhat successful strategy and build upon it to achieve competency. One other possible benefit that may come with imitation learning is that the agent may perform better with the non-linear route of the game and finding certain key items that are necessary for progression.

## What's next

After we determine pretraining we hope to examine the agent's performance and improve upon its weaknesses. Some general improvements that don't depend upon the agent are improvements to the reward function, longer pretraining demonstrations, and longer training periods.
