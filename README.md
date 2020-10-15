# ai_pj1-2
# Contest@unimelb

retrieve from https://gitlab.eng.unimelb.edu.au/zhouhuiw/comp90054-pacman (the original repository)

## Team Information
- team name: halation
- team member:
  - Zhouhui Wu 963830
  - Xinze Li 964135
  - Dongsheng Xie 963832

## Introduction

We have developed 3 different agents using 3 methods. The 3 agents are approximate q learning agent, Monte Carlo Tree Search - only agent and Monte Carlo Tree Search - Search agent. The Monte Carlo Tree Search - Search agent is used for contest.

## Agent description

### 1. Approximate q learning agent


In the approximate q learning agent, we use approximate q learning method to build the pacman agent. This agent is inside the file **approQLearningTeam.py**.  This agent is **not** used for the contest.

We select some features ourselves and train our pacman agent by playing thousands of games against the baseline team. By training the model, our agent learnt to eat food and escape from ghost's attacking.

The approximate q learning agent is the first agent we made, after finding that our MCTS agent performed better, we then focus on the MCTS method, and spent less to improve the approximate q learning agent. As a result, it may run with poor performance.


### 2. Monte Carlo Tree Search - only agent (MCTS-only agent)

In the Monte Carlo Tree Search - only agent (MCTS-only agent), we use only Monte Carlo Tree Search method to build our pacman agent. This agent is inside the file **mcts-only.py**. This agent is **not** used for the contest.

#### (1) Pacman

We use Monte Carlo Tree Search method to construct the pacman agent. In each turn, we train our pacman agent with thousands of simulation. Due to the time limit, we cannot simulate the whole gaming in each simulation; instead, we simulated the game in depth of 5, and evaluate the final state it arrives at. The simulation depth 5 is chosen elaborately as the pacman Agent cannot observe its enemy if the Manhattan Distance larger than 5, in this case, simulation of depth 5 could be reasonable, not too much optimistic. In each simulation, Agent's action is chosen randomly from all its feasible moves. We model the enemies' move if we can observe them. The Baseline Defensive Agent is used when modelling the enemy's ghosts. The Baseline Defensive Agent is not performing well; however, when it can observe the Pacman, it will chase it and trying to eat it. This property is useful for us: we want our agent to learn how to run away when observing enemy chasing it.

#### (2) Ghost

For the Ghost part, we use a decision tree to make the ghost chase the pacman and protect foods. It simply selects the action minimising the distance between itself and enemy pacman if the ghost observes the enemy; otherwise, it selects the action minimising the distance between itself and its border, in this case, it can protect food from being eaten by pacman.


### 3. Monte Carlo Tree Search - Search agent (MCTS-Search agent)

In the Monte Carlo Tree Search - Search agent (MCTS-Search agent), we use both Monte Carlo Tree Search method and search method to build our pacman agent, and search method to build the ghost agent.
This agent is used for **contest** and it is coded inside the file **myTeam.py**.

#### (1) Pacman

The pacman is made using Monte Carlo Tree Search method and search method. The Monte Carlo Tree Search method used is similar to the  MCTS-only agent; however, with a better simulation and evaluation function, the new agent performs better. For the evaluation, we select better feature and design better reward for each situation. For simulation, we remove moves causing pacman die in each turn from action list. In this case, all moves make the pacman survive and pacman only choose from action making it survive. By improving the simulation, our pacman may be less conservative when enemy ghost is around and still trying to eat food.

The pacman agent also uses search method. During the game, if the pacman is blocked by enemy ghost, it will find a short way to another border to attack using search method.

#### (2) Ghost

The Ghost is constructed using decision tree and search method. Similar to the MCTS-only agent, it uses a decision tree to decide what to do. It uses search method to find the shortest way to capture enemy if enemy pacman invades. The reason for using search method instead of maze distance is that the maze distance provided is the "global maze distance", which sometimes lead the ghost to enemy's home and makes the ghost at risk. We use search to find a "in home short distance" and in this way, the ghost will always stay at home and would not be at risk.



## Methods

### 1. Monte Carlo Tree Search

We use Monte Carlo Tree Search method to construct our MCTS-only Agent and MCTS-Search Agent. 

In each turn, we train our pacman agent with thousands of simulation. Due to the time limit, we cannot simulate the whole gaming process in each simulation; therefore, we simulated the game in depth of 5, and evaluate the final state it arrives at. The simulation depth 5 is chosen elaborately as the pacman Agent cannot observe its enemy if the Manhattan Distance larger than 5, in this case, simulation of depth 5 could be reasonable, not too much optimistic. In each simulation, Agent's action is chosen randomly from all its feasible moves. We model the enemies' move if we can observe them. The Baseline Defensive Agent is used when modelling the enemy's ghosts. This Agent is not performing well; however, when it can observe the Pacman, it will chase it and trying to eat it. This property is useful for us: we want our agent to learn how to run away when observing enemy chasing it.

### 2. Search

We use the search method in the MCST-Search agent. This is used in both pacman agent and ghost agent. For a pacman agent, it uses the search method to find a short way to border for attacking; the ghsot agent uses the search method to find a short way to capture enemy pacman.

Although the CaptureAgent class already provided "maze distance" class and we can look up the "maze distance" between any positions, we found it is not always helpful.  The default maze distance method only compute the global maze distance and sometimes following that route lead the ghost to enemy's home, which makes the ghost agent at risk; our search method find way inside our home, in this case, the ghost agent will always stay at home and would not be at risk. And this approach improve the ghost's performance.


### 3. Approximate Q Learning

We use the Approximate Q Learning for our approximate q learning agent. We design some features by ourselves and try training our model to learn weight for each feature. This might be a good method as when playing the game, we are less likely to exceed time limits: we have already learnt the feature for each weight and can compute Q value easily. We use the approximate q learning method first; however, in the end, we gave up using this method for our pacman agent, as we can hardly choose good feathers to learn, and the performance did not meet our expectation. Also, our agent took too much time on training and converge slowly. We then focus on the Monte Carlo Tree Search method instead.

## Challenges

One challenge we faced during the implementation is to build a proper evaluation function for our Monte Carlo Tree Search method. It is essential for our Monte Carlo Tree Search agent as we cannot run the whole game in simulation phase and hence require an evaluation function to tell us if the new state is good enough. It is hard to find proper features and combine them in a good way. We watched some games and used domin knowledge to improve our evaluation function. The new function performed better comparing to our early one and could get a high rank position in pre-contests.



## Possible Improvement

Currently, we select and expand node randomly in the expansion phase. One possible improvement we could make in the future is that we can use the UCB1 method to select nodes to expand. This might make our agent's performance better. This method we have not implemented in our agent now, as this might make the coding harder, and we do not have much time for this.

Another possible improvement we could make is that we can use some techniques to inform opponents' action in our defensive agent. Simply chasing opponent agent is not a good idea as the pacman can often run away easily. Inferring the opponent's action and position when defending our food might be better. However, this might be hard. Maybe we should involve some technique from Markov Decision Processes, Q Learning or Plan Recognition.

Evaluation function is also a part we could improve in the future. It is essential for our Monte Carlo Tree Search agent as we cannot run the whole game in the simulation phase and hence require some evaluation function to tell us if the new state is good enough. It is hard to find proper features and combine them in a good way to evaluate the new states. We have tried and changed our evaluation function many time but still not able to build a good enough evaluation function for our Monte Carlo Tree Search agent. Currently we use a decission tree to evaluate the state; however, it does not always performing well. This rule based approach might be done better after later involving some general techniques.

## Analysis

In the analysis, we compare the performance in 4 different situation. The following table shows the performance in 4 situation.

|  | MCTS-only Agent | MCTS-Search Agent | Approximate Q Learning Agent |
| ------ | ------ | ------ | ------ |
| Eating Efficiency When No Enemy Around| Low | Medium | High |
| Returning Back Frequency When Already Eating Some Food | High | Medium | Low |
| Scaping From Enemy Capture | High | High | Medium |
| Eating Efficiency When Enemy Around| Low | High | Low |

### 1. Eating Efficiency When No Enemy Around

When there is no enemy around the pacman, it should eat as much as possible to gain a high score. We compare the eating efficiency in this situation. The Approximate Q Learning Agent performed best as it eats as much as possible, this might because our approximate q learning's evaluation give high reward for eating and MCTS agent give a higher reward for returning home comparing to storing food. Our MCTS-only agent and MCTS-Search agent done worse in this situation. Our MCTS-Search agent performed better than MCTS-only Agent, this might because the evaluation function is better.


### 2. Returning Back Frequency When Already Eating Some Food

When pacman has eaten some food, it should return back and store its food. However, it should not return back often as it will waste time on the way; it should also not returning back seldom as it may be in risk latter and once it is killed, all food will return back to enemy's home. Our approximate q learning agent return home seldom as returning home's reward is smaller than eating's reward. Our MCTS-Only agent return home frequently as the reward of returning home is higher. And the MCTS-Search agent return home not so frequent as the reward for returning home is medium.

### 3. Scaping From Enemy Capture

When there is enemy ghost nearby, the pacman should escape from the ghost. Our MCTS-only agent and MCTS-Search agent performed better comparing to the approximate q learning agent. This might because the MCTS method takes some further steps into account and will not choose action causing pacman to die in the further few steps. So they could perform better and avoid being eaten.

### 4. Eating Efficiency When Enemy Around

When there is enemy ghost around the pacman, pacman should eat as much as possible and keep from been eaten by ghost. Our approximate q learning agent and MCTS-only agent seem performed conservatively, they tried to keep away from the ghost and stop eating anymore. Our MCTS-Search agent perfomed better, it kept eating food and stayed away from the ghost. This better performance might result from the better simulation in MCTS-Search agent.

## Video


The presentation is made by 3 people together:

| Time | Part | Speaker |
| ------ | ------ | ------ |
| 0:00 - 1:20 | Introduction & Method Describption | Zhouhui Wu |
| 1:21 - 1: 55 | Challenge | Xinze Li |
| 1:56 - 2:52 | Improvement Could Make| Xinze Li |
| 2:53 - 4:44 | Demo | Dongsheng Xie |


Link to Youtube: https://www.youtube.com/watch?v=Y8dd82SVpHY&feature=youtu.be


## Rank

| Date | halation | Top | Medium | Basic | Comment |
| ------ | ------ | ------ | ------ | ------ | ------ |
| 10-3 | 27 | 16 | 28 | 45 | First Submission: Bug in Defensive Agent (7/120 failed) |
| 10-4 | 21 | 14 | 30 | 60 | Bug Fixed |
| 10-5 | 13 | 20 | 30 | 58 | Defensive Agent Improved |
| 10-6 | 76 | 26 | 49 | 61 | Start-Up Time Exceeded: Print too much data when start-up (145/154 failed) |
| 10-7 | 14 | 22 | 60 | 77 | Print function commented |
| 10-8 | 9 | 21 | 60 | 55 | Nothing Changed |
| 10-9 | 18 | 32 | 48 | 78 | Offensive Agent Improved: Only One Map Tested |
| 10-10 | 13 | 28 | 75 | 103 | Nothing changed: Only One Map Tested |
| 10-11 | 3 | 38 | 86 | 90 | Nothing changed: Only One Map Tested |
| 10-12 | 2 | 47 | 90 | 125 | Nothing changed: Only One Map Tested |
| 10-13 | 14 | 62 | 116 | 114 | Nothing changed: Only One Map Tested |
| 10-14 | - | - | - | - | No result released |
| 10-15 | 6 | 76 | 122 | 138 | Nothing changed: Only One Map Tested |
