# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
import capture
import math

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'MonteCarloTreeSearchCaptureAgent', second = 'DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########
class BasicCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
 
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != util.nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveReflexAgent(BasicCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}

class DefensiveReflexAgent(BasicCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

class MonteCarloTreeSearchCaptureAgent(BasicCaptureAgent):
  def __init__( self, index, timeForComputing = .1 ):
    self.index = index
    self.red = None
    self.agentsOnTeam = None
    self.distancer = None
    self.observationHistory = []
    self.timeForComputing = timeForComputing
    self.display = None
    self.alpha=0.2
    self.discount=0.8
    self.epsilon=0.2
    self.weights = util.Counter()
    self.features = util.Counter()
    self.simulationNumber=70
    self.simulationDepth=3
    self.nearestFood=None
    self.mode='Offensive' # Offensive or Defensive
    if self.index==0 or self.index==2:
      self.enemyPacmanIndex=1
      self.enemyGhostIndex=3
    else:
      self.enemyPacmanIndex=0
      self.enemyGhostIndex=2
    self.enemyPacman=OffensiveReflexAgent(self.enemyPacmanIndex)
    self.enemyGhost=DefensiveReflexAgent(self.enemyGhostIndex)
    self.containFood=0

  def setNearestFood(self,state):
    foodList=self.getFood(state).asList()
    distance=float('inf')
    for food in foodList:
      if self.getMazeDistance(self.start,food)<distance:
        self.nearestFood=food
        distance=self.getMazeDistance(self.start,food)

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    if self.index==0 or self.index==1:
      self.mode='Offensive'
    else:
      self.mode='Defensive'
    self.setNearestFood(gameState)

  def chooseAction(self, state):
    if self.start==state.getAgentPosition(self.index):
      self.containFood=0
      self.setNearestFood(state)

    if self.mode=='Offensive':
      result=util.Counter()
      nums=util.Counter()
      actions=self.getActions(state)
      foods=self.containFood
      nearestFood=self.nearestFood
      for i in range(self.simulationNumber):
        self.containFood=foods
        self.nearestFood=nearestFood
        (reward,action)=self.simulation(state,0)
        result[action]+=reward
        nums[action]+=1
        print action,reward
      bestAction,bestActionReward=None,-float('inf')
      for action in actions:
        oneReward=result[action]/nums[action]
        print '============================>',oneReward,action
        if oneReward>bestActionReward:
          bestActionReward=oneReward
          bestAction=action
      print '============== one step =============='
      # print self.containFood
      self.containFood=foods
      self.nearestFood=nearestFood
      nextState=state.generateSuccessor(self.index,bestAction)
      oldFoodList,newFoodList=self.getFood(state).asList(),self.getFood(nextState).asList()
      if len(oldFoodList)>len(newFoodList):
        self.containFood+=1
        self.setNearestFood(nextState)
      if self.getScore(nextState)>self.getScore(state):
        self.containFood=0
      # if self.start==nextState.getAgentPosition(self.index):
      #   self.containFood=0
      #   self.setNearestFood(state)
      return bestAction

    else: # defensive
      # unimplement yet
      return 'West'


  def simulation(self,state,depth):
    if depth<self.simulationDepth:
      actions=self.getActions(state)
      action = random.choice(actions)
      nextState=state.generateSuccessor(self.index,action)
      if nextState.getAgentPosition(self.enemyGhostIndex)!=None:
        self.enemyGhost.registerInitialState(nextState)
        enemyGhostAction=self.enemyGhost.chooseAction(nextState)
        nextState=nextState.generateSuccessor(self.enemyGhostIndex,enemyGhostAction)
        if nextState.getAgentPosition(self.index)==self.start:
          return (-100,action)
        reward=self.computeRewardWithOpponentMovement(state,nextState,action)
        self.setNearestFood(nextState)
        return (self.simulation(nextState,depth+1)[0]+reward,action)
      else:
        if nextState.getAgentPosition(self.index)==self.start:
          return (-100,action)
        reward=self.computeReward(state,nextState,action)
        self.setNearestFood(nextState)
        return (self.simulation(nextState,depth+1)[0]+reward,action)
    else:
      return (self.getScore(state),None)

  def getActions(self,state):
    legalActions = state.getLegalActions(self.index)
    legalActions.remove('Stop')
    return legalActions
  
  def computeRewardWithOpponentMovement(self,state,nextState,action):
    reward,oldFoodList,newFoodList=0,self.getFood(state).asList(),self.getFood(nextState).asList()
    oldShortDistance,newShortDistance=self.getMazeDistance(self.nearestFood,state.getAgentPosition(self.index)),self.getMazeDistance(self.nearestFood,nextState.getAgentPosition(self.index))
    oldEnemyPacmanPosition,oldEnemyGhostPosition,oldPacmanPosition,newPacmanPosition=state.getAgentPosition(self.enemyPacmanIndex),state.getAgentPosition(self.enemyGhostIndex) ,state.getAgentPosition(self.index),nextState.getAgentPosition(self.index)
    if newPacmanPosition==self.start:
      # been eaten by enemy
      self.containFood=0
      reward=-200
    else:
      # survive
      if self.getScore(state)<self.getScore(nextState):
          # get higher score <=> storing food in home
          self.containFood=0
          reward+=200
      else:
        # still wondering without storing food
        if self.containFood<3:
          # eat few food, might focus on eating more food
          if len(oldFoodList)>len(newFoodList):
            # eat a food
            self.containFood+=1
            reward+=60
          else:
            # not eat a food
            if oldShortDistance<newShortDistance:
              # move away from food
              if oldEnemyGhostPosition!=None:
                if self.getMazeDistance(oldEnemyGhostPosition,oldPacmanPosition)<=2:
                  # when enemy is coming, escape will not result in negative reward
                  reward+=0
                else:
                  # enemy far away from pacman, but move away from food
                  reward-=50
              else:
                # no enemy nearby, but move away from food
                reward-=50
            else:
              # move toward food
              reward+=20
        else:
          # eat much food, should go back and storing them
          if self.red:
            if action=='West':
              # home direction
              reward+=50
            elif action=='East':
              # not home direction
              reward-=10
          else:
            if action=='East':
              # home direction
              reward+=50
            elif action=='West':
              # not home direction
              reward-=10
    return reward

  def computeReward(self,state,nextState,action):
    reward,oldFoodList,newFoodList=0,self.getFood(state).asList(),self.getFood(nextState).asList()
    enemyPacmanPosition,enemyGhostPosition,oldPacmanPosition,newPacmanPosition=state.getAgentPosition(self.enemyPacmanIndex),state.getAgentPosition(self.enemyGhostIndex) ,state.getAgentPosition(self.index),nextState.getAgentPosition(self.index)
    oldShortDistance,newShortDistance=self.getMazeDistance(self.nearestFood,state.getAgentPosition(self.index)),self.getMazeDistance(self.nearestFood,nextState.getAgentPosition(self.index))
    if enemyPacmanPosition==None:
      enemyPacmanPosition=nextState.getAgentPosition(self.enemyPacmanIndex)
    if enemyGhostPosition==None:
      enemyGhostPosition=nextState.getAgentPosition(self.enemyGhostIndex)

    if enemyGhostPosition!=None:
      if self.getMazeDistance(enemyGhostPosition,newPacmanPosition)==1:
        # will be eaten by enemy
        self.containFood=0
        reward=-200
        return reward

    if newPacmanPosition==self.start:
      # been eaten by enemy
      self.containFood=0
      reward=-200
    else:
      # survive
      if self.getScore(state)<self.getScore(nextState):
          # get higher score <=> storing food in home
          self.containFood=0
          reward+=200
      else:
        # still wondering without storing food
        if self.containFood<3:
          # eat few food, might focus on eating more food
          if len(oldFoodList)>len(newFoodList):
            # eat a food
            self.containFood+=1
            reward+=60
          else:
            # not eat a food
            if oldShortDistance<newShortDistance:
              # move away from food
              if enemyGhostPosition!=None:
                if self.getMazeDistance(enemyGhostPosition,oldPacmanPosition)<=2:
                  # when enemy is coming, escape will not result in negative reward
                  reward+=0
                else:
                  # enemy far away from pacman, but move away from food
                  reward-=50
              else:
                # no enemy nearby, but move away from food
                reward-=50
            else:
              # move toward food
              reward+=20
        else:
          # eat much food, should go back and storing them
          if self.red:
            if action=='West':
              # home direction
              reward+=50
            elif action=='East':
              # not home direction
              reward-=10
          else:
            if action=='East':
              # home direction
              reward+=50
            elif action=='West':
              # not home direction
              reward-=10
    return reward




    