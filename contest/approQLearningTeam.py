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
               first = 'OffensiveQLearningAgent', second = 'DefensiveReflexAgent'):
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
class ReflexCaptureAgent(CaptureAgent):
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

class OffensiveReflexAgent(ReflexCaptureAgent):
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

class DefensiveReflexAgent(ReflexCaptureAgent):
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
    

def generateLegalActions(actions):
  newActions=[]
  for action in actions:
    if action!='Stop':
      newActions.append(action)
  return newActions

class OffensiveQLearningAgent(CaptureAgent):
 
  def __init__( self, index, timeForComputing = .1 ):
    self.index = index
    self.red = None
    self.agentsOnTeam = None
    self.distancer = None
    self.observationHistory = []
    self.timeForComputing = timeForComputing
    self.display = None
    self.values = util.Counter()
    self.trainNumber=0
    self.alpha=0.2
    self.discount=0.8
    self.epsilon=0.1
    self.epsilon=0
    self.weights = util.Counter()
    self.features = util.Counter()
    self.targetTrainingNumber=50
    self.totalFood=0
    self.weights["bias"] = 1
    self.weights["ghost-1-step-away"]=1
    # self.weights["#-of-ghosts-1-step-away"]=1
    self.weights["eats-food"] = 100
    self.weights["closest-food"]= -1000

    self.features["bias"] = 1.0
    self.features["ghost-1-step-away"]=1
    # self.features["#-of-ghosts-1-step-away"]=0
    self.features["eats-food"] = 1.0
    self.features["closest-food"]= 1.0
    self.border=[]
    self.boundary=0

  def getQValue(self, state, action):
    nextState=state.generateSuccessor(self.index,action)
    self.updateFeatures(nextState)
    # if self.trainNumber==self.targetTrainingNumber:
    #   print 'features: ',self.features
    features=self.features.copy()
    if len(self.weights)!=0 and len(features)!=0:
      return features*self.weights
    return 0

  def getWeights(self):
    return self.weights

  def updateFeatures(self,nextState):
    # self.features["#-of-ghosts-1-step-away"]=0
    self.features["eats-food"]=0
    newFoods=self.getFood(nextState)
    newShortDistance=0
    newFoodList = newFoods.asList()    
    if len(newFoodList) > 0:
      myPos = nextState.getAgentPosition(self.index)
      newShortDistance = min([self.getMazeDistance(myPos, food) for food in newFoodList])
    self.features["eats-food"] = self.totalFood-len(newFoodList)
    self.features["closest-food"] = newShortDistance/10.0
    if self.red:
      if nextState.getAgentPosition(3)!=None:
        if self.getMazeDistance(nextState.getAgentPosition(3),nextState.getAgentPosition(self.index))==1:
          self.features["ghost-1-step-away"]=1
    else:
      if nextState.getAgentPosition(1)!=None:
        if self.getMazeDistance(nextState.getAgentPosition(1),nextState.getAgentPosition(self.index))==1:
          self.features["ghost-1-step-away"]=1

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.weights["bias"] = 1
    self.weights["ghost-1-step-away"]=1
    # self.weights["#-of-ghosts-1-step-away"]=1
    self.weights["eats-food"] = 100
    self.weights["closest-food"]= -1000

    self.features["bias"] = 1.0
    self.features["ghost-1-step-away"]=1
    # self.features["#-of-ghosts-1-step-away"]=0
    self.features["eats-food"] = 1.0
    foods=self.getFood(gameState)
    shortDistance=0
    foodList = foods.asList()    
    if len(foodList) > 0:
      myPos = gameState.getAgentPosition(self.index)
      shortDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
    self.totalFood=len(foodList)
    if self.red:
      self.boundary = (gameState.data.layout.width - 2) / 2
    else:
      self.boundary = ((gameState.data.layout.width - 2) / 2) + 1
    for i in range(gameState.data.layout.height):
      if gameState.hasWall(self.boundary, i):
        continue
      self.border.append((self.boundary, i))
    self.features["closest-food"] = shortDistance/10.0
    self.train(gameState)

  def computeValueFromQValues(self, state):
    legalActions, bestQ = state.getLegalActions(self.index), -float('inf')
    legalActions=generateLegalActions(legalActions)
    if len(legalActions) == 0:
      return 0
    for action in legalActions:
      bestQ = max(bestQ, self.getQValue(state, action))
    return bestQ

  def computeActionFromQValues(self, state):
    legalActions, bestQ, action = state.getLegalActions(self.index), -float('inf'), None
    legalActions=generateLegalActions(legalActions)
    if len(legalActions) == 0:
      return action
    for legalAction in legalActions:
      qValue = self.getQValue(state, legalAction)
      # print legalAction,qValue
      if qValue > bestQ:
        action = legalAction
        bestQ = qValue
    return action

  def getActionByQValues(self, state):
    legalActions = state.getLegalActions(self.index)
    legalActions=generateLegalActions(legalActions)
    action = None
    if len(legalActions) == 0:
      return action
    if util.flipCoin(self.epsilon):
      action = random.choice(legalActions)
    else:
      action = self.computeActionFromQValues(state)
      # print '----------------------------'
    return action

  def update(self, state, action, nextState, reward):
    qValue=self.getQValue(state,action)
    alpha=self.alpha
    gamma=self.discount
    maxQk= self.computeValueFromQValues(nextState)
    self.updateFeatures(nextState)
    difference=(reward+gamma*maxQk)-qValue
    temp=self.features.copy()
    if difference!=0:
      temp.divideAll(1.0/(alpha*difference))
      self.weights+=temp
      

  def train(self,state):
    while self.trainNumber < self.targetTrainingNumber:
      self.singleTrain(state)
      self.trainNumber+=1
      print self.trainNumber
      print self.weights
    # self.epsilon=0 # no more exploring
    print 'weights: ',self.weights

  def singleTrain(self,gameState):
    state=gameState.deepCopy()
    trainingAgent=[]
    agentIndex,i=self.index,0
    trainingAgent.append(OffensiveQLearningAgent(agentIndex))
    trainingAgent[0].start = gameState.getAgentPosition(trainingAgent[0].index)
    CaptureAgent.registerInitialState(trainingAgent[0], gameState)
    trainingAgent[0].epsilon=0.05
    enemyPacmanIndex=0
    enemyGhostIndex=2
    ourGhostIndex=3
    if self.red:
      enemyPacmanIndex=1
      enemyGhostIndex=3
      ourGhostIndex=2
    trainingAgent.append(OffensiveReflexAgent(enemyPacmanIndex))
    trainingAgent[1].registerInitialState(state)
    trainingAgent.append(DefensiveReflexAgent(ourGhostIndex))
    trainingAgent[2].registerInitialState(state)
    trainingAgent.append(DefensiveReflexAgent(enemyGhostIndex))
    trainingAgent[3].registerInitialState(state)
    while i<100 and not state.isOver():
      ourPacmanAction=trainingAgent[0].getActionByQValues(state)
      nextState=state.generateSuccessor(agentIndex,ourPacmanAction)
      # print nextState.getAgentPosition(self.index)
      reward=self.computeRewardForOffensiveAgent(state,nextState,trainingAgent[0].index,i*4)
      self.update(state,ourPacmanAction,nextState,reward)
      if nextState.isOver():
        break
      state=nextState

      enemyPacmanAction=trainingAgent[1].chooseAction(state)
      nextState=state.generateSuccessor(trainingAgent[1].index,enemyPacmanAction)
      if nextState.isOver():
        break
      state=nextState

      ourGhostAction=trainingAgent[2].chooseAction(state)
      nextState=state.generateSuccessor(trainingAgent[2].index,ourGhostAction)
      if nextState.isOver():
        break
      state=nextState

      enemyGhostAction=trainingAgent[3].chooseAction(state)
      nextState=state.generateSuccessor(trainingAgent[3].index,enemyGhostAction)
      if nextState.isOver():
        break
      state=nextState
      i+=1
    # exit()

  def computeRewardForOffensiveAgent(self,previousState,nextState,index,step):
    # enemy agent distance not included
    endStep=400
    reward=-20
    oldFoods=self.getFood(previousState)
    newFoods=self.getFood(nextState)
    enemyGhostIndex=2
    if self.red:
      enemyPacmanIndex=3
    oldShortDistance=0
    oldFoodList = oldFoods.asList()
    enemyPosition,oldPacmanPosition,nextPacmanPosition=None,None,None
    oldPacmanToHomeDistance,newPacmanToHomeDistance=0,0
    oldPacmanPosition = previousState.getAgentPosition(index)
    if len(oldFoodList) > 0: # This should always be True,  but better safe than sorry
      
      oldShortDistance = min([self.getMazeDistance(oldPacmanPosition, food) for food in oldFoodList])
      enemyPosition = previousState.getAgentPosition(enemyGhostIndex)
    newShortDistance=0
    newFoodList = newFoods.asList()
    nextPacmanPosition = nextState.getAgentPosition(index)
    if len(newFoodList) > 0: # This should always be True,  but better safe than sorry
      
      newShortDistance = min([self.getMazeDistance(nextPacmanPosition, food) for food in newFoodList])
      if enemyPosition==None:
        enemyPosition = nextState.getAgentPosition(enemyGhostIndex)

    distanceToOldEnemy,distanceToNextEnemy=0,0
    if self.getScore(nextState)>self.getScore(previousState):
      return 200
    if enemyPosition!=None:
      oldDistanceToEnemy=self.getMazeDistance(enemyPosition, oldPacmanPosition)
      nextDistanceToEnemy=self.getMazeDistance(enemyPosition, nextPacmanPosition)
      if oldDistanceToEnemy>nextDistanceToEnemy:
        if nextDistanceToEnemy<=1:
          return -50
      else:
        return 30
    if self.features["eats-food"]>=3:
      oldPacmanToHomeDistance = min([self.getMazeDistance(oldPacmanPosition, spot) for spot in self.border])
      newPacmanToHomeDistance = min([self.getMazeDistance(nextPacmanPosition, spot) for spot in self.border])
      if oldPacmanToHomeDistance<newPacmanToHomeDistance:
        return -50
      # print height,width
    if len(newFoodList)<len(oldFoodList):
      return 50
    else:
      if newShortDistance<oldShortDistance:
        return 0
      else:
        return -40
    

    if nextState.isOver() or step>=endStep:
      if self.getScore(nextState)<=0:
        if self.features["eats-food"]>0:
          return -500
        return -50
      else:
        return 1000
    return reward


  def chooseAction(self, gameState):
    action=self.getActionByQValues(gameState)
    nextState=gameState.generateSuccessor(self.index,action)
    # reward=self.computeRewardForOffensiveAgent(gameState,nextState,self.index,0)
    # print nextState.getAgentPosition(self.index)
    # self.update(gameState,action,nextState,reward)
    return action

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


