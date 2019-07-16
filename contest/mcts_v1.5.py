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

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'MonteCarloTreeSearchCaptureAgent', second = 'MonteCarloTreeSearchCaptureAgent'):
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
    self.simulationNumber=1000
    self.simulationDepth=4
    self.nearestFood=None
    self.nearestBorder=None
    self.powerOn=False
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
    self.boundary=0
    self.border=[]
    self.powerLeft=0
    self.enemyPowerOn=False
    self.enemyPowerLeft=0
    self.enemyPacmanPosition=None
    self.enemyGhostPosition=None
    self.bestActions=[]
    self.history=[]

  def setNearestBorder(self,state):
    distance=float('inf')
    position=state.getAgentPosition(self.index)
    for spot in self.border:
      if self.getMazeDistance(position,spot)<distance:
        self.nearestBorder=spot
        distance=self.getMazeDistance(position,spot)

  def setBetterBorder(self,state):
    capsule=self.getCapsulesYouAreDefending(state)
    if len(capsule)>0:
      distance=float('inf')
      position=capsule[0]
      for spot in self.border:
        if self.getMazeDistance(position,spot)<distance:
          self.nearestBorder=spot
          distance=self.getMazeDistance(position,spot)
    else:
      distance=float('inf')
      position=state.getAgentPosition(self.index)
      for spot in self.border:
        if self.getMazeDistance(position,spot)<distance:
          self.nearestBorder=spot
          distance=self.getMazeDistance(position,spot)

  def setNearestFood(self,state):
    foodList=self.getFood(state).asList()
    distance=float('inf')
    position=state.getAgentPosition(self.index)
    for food in foodList:
      if self.getMazeDistance(position,food)<distance:
        self.nearestFood=food
        distance=self.getMazeDistance(position,food)

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    if self.index==0 or self.index==1:
      self.mode='Offensive'
    else:
      self.mode='Defensive'
    self.setNearestFood(gameState)
    if self.red:
      self.boundary = (gameState.data.layout.width - 2) / 2
    else:
      self.boundary = ((gameState.data.layout.width - 2) / 2) + 1
    for i in range(gameState.data.layout.height):
      if gameState.hasWall(self.boundary,i):
        continue
      self.border.append((self.boundary,i))
    self.setNearestBorder(gameState)

  def getActions(self,state):
    legalActions = state.getLegalActions(self.index)
    legalActions.remove('Stop')
    return legalActions

  def chooseAction(self, state):

    # return to start place
    if self.start==state.getAgentPosition(self.index):
      self.containFood=0
      self.setNearestFood(state)
      if self.mode=='Offensive':
        self.setNearestBorder(state)
      else:
        self.setBetterBorder(state)

    # check enemy power left
    if self.enemyPowerLeft==0:
      if self.index==2 or self.index==3:
        # switch to defence mode
        self.enemyPowerOn=False
        self.mode='Defensive'
    else:
      self.enemyPowerLeft-=1

    # assign mode to 2 agents
    if self.index==0 or self.index==1:
      # decide agent mode
      if self.getScore(state)>=6:
        # already gain a high score: switch to defensive mode
        self.mode='Defensive'
      else:
        # score is low currently: switch to offensive mode
        self.mode='Offensive'

    # offensive agent
    if self.mode=='Offensive':
      if self.powerLeft>25:
        self.powerLeft-=1
        actions=self.getActions(state)
        for action in actions:
          nextState=state.generateSuccessor(self.index,action)
          if self.getMazeDistance(state.getAgentPosition(self.index),self.nearestFood)>self.getMazeDistance(nextState.getAgentPosition(self.index),self.nearestFood):
            oldFoodList,newFoodList=self.getFood(state).asList(),self.getFood(nextState).asList()
            if len(oldFoodList)>len(newFoodList):
              self.containFood+=1
              self.setNearestFood(nextState)
              self.setNearestBorder(nextState)
            return action
        return action
      if len(self.history)>3:
        oldPosition=self.history[len(self.history)-2][0]
        oldTargetFood=self.history[len(self.history)-2][1]
        if self.nearestFood==oldTargetFood:
          if self.getMazeDistance(oldTargetFood,oldPosition)-self.getMazeDistance(self.nearestFood,state.getAgentPosition(self.index))<1:
            self.changeTargetFood(state)
      result=util.Counter()
      nums=util.Counter()
      actions=self.getActions(state)
      foods=self.containFood
      nearestFood=self.nearestFood
      timeStart=time.time()
      for i in range(self.simulationNumber):
        timeEnd=time.time()
        if timeEnd-timeStart>0.600:
          break
        self.containFood=foods
        self.nearestFood=nearestFood
        record={}
        (reward,action)=self.simulation(state,foods,state,0)
      #   if result[action]<reward:
      #     result[action]=reward
      # bestAction=result.argMax()
        result[action]+=reward
        nums[action]+=1
      bestAction,bestActionReward=None,-float('inf')
      for action in actions:
        oneReward=result[action]/nums[action]
        if oneReward>bestActionReward:
          bestActionReward=oneReward
          bestAction=action

      self.containFood=foods
      self.nearestFood=nearestFood
      self.history.append((state.getAgentPosition(self.index),self.nearestFood))
      nextState=state.generateSuccessor(self.index,bestAction)
      oldFoodList,newFoodList,oldCapsule,newCapsule=self.getFood(state).asList(),self.getFood(nextState).asList(),self.getCapsules(state),self.getCapsules(nextState)
      if len(oldFoodList)>len(newFoodList):
        self.containFood+=1
        self.setNearestFood(nextState)
        self.setNearestBorder(nextState)
      if len(oldCapsule)>len(newCapsule):
        self.powerOn=True
        self.powerLeft=40
        # self.bestActions=self.findMaxFoodPathIn15Steps(nextState)
      else:
        if self.powerOn:
          self.powerLeft-=1
          if self.powerLeft==0:
            self.powerOn=False
      if self.getScore(nextState)>self.getScore(state):
        self.containFood=0
      # if self.start==nextState.getAgentPosition(self.index):
      #   self.containFood=0
      #   self.setNearestFood(state)
      return bestAction

    else: # defensive
      lastState=self.getPreviousObservation()
      if lastState!=None:
        oldCapsule,newCapsule=self.getCapsulesYouAreDefending(lastState),self.getCapsulesYouAreDefending(state)
        if len(oldCapsule)>len(newCapsule):
          self.enemyPowerLeft=40
          self.enemyPowerOn=True
          self.mode='Offensive'
          return self.chooseAction(state)


        if state.getAgentPosition(self.enemyPacmanIndex)!=None:
          enemyPacmanPosition=state.getAgentPosition(self.enemyPacmanIndex)
          actions=state.getLegalActions(self.index)
          shortestDistance=float('inf')
          bestAction='Stop'
          for action in actions:
            nextState=self.getSuccessor(state,action)
            if nextState.getAgentPosition(self.index)[0]>self.boundary:
              continue
            if self.getMazeDistance(nextState.getAgentPosition(self.index),enemyPacmanPosition)<shortestDistance:
              shortestDistance=self.getMazeDistance(nextState.getAgentPosition(self.index),enemyPacmanPosition)
              bestAction=action
          return bestAction
        else:
          oldFood,newFood=self.getFoodYouAreDefending(lastState),self.getFoodYouAreDefending(state)
          if len(oldFood.asList())>len(newFood.asList()):
            # food been eaten
            enemyPacmanPosition=None
            if self.red:
              for i in range(self.boundary):
                for j in range(state.data.layout.height):
                  if oldFood[i][j]!=newFood[i][j]:
                    enemyPacmanPosition=(i,j)
            else:
              for i in range(self.boundary,state.data.layout.width):
                for j in range(state.data.layout.height):
                  if oldFood[i][j]!=newFood[i][j]:
                    enemyPacmanPosition=(i,j)
            self.enemyPacmanPosition=enemyPacmanPosition
            actions=state.getLegalActions(self.index)
            shortestDistance=float('inf')
            bestAction='Stop'
            for action in actions:
              nextState=self.getSuccessor(state,action)
              if self.getMazeDistance(nextState.getAgentPosition(self.index),enemyPacmanPosition)<shortestDistance:
                shortestDistance=self.getMazeDistance(nextState.getAgentPosition(self.index),enemyPacmanPosition)
                bestAction=action
            return bestAction
          else:
            if self.enemyPacmanPosition!=None:
              actions=state.getLegalActions(self.index)
              shortestDistance=float('inf')
              bestAction='Stop'
              for action in actions:
                nextState=self.getSuccessor(state,action)
                if nextState.getAgentPosition(self.index)[0]>self.boundary:
                  continue
                if self.getMazeDistance(nextState.getAgentPosition(self.index),self.enemyPacmanPosition)<shortestDistance:
                  shortestDistance=self.getMazeDistance(nextState.getAgentPosition(self.index),self.enemyPacmanPosition)
                  bestAction=action
              return bestAction
            else:
              actions=state.getLegalActions(self.index)
              shortestDistance=float('inf')
              bestAction='Stop'
              for action in actions:
                nextState=self.getSuccessor(state,action)
                if nextState.getAgentPosition(self.index)[0]>self.boundary:
                  continue
                if self.getMazeDistance(nextState.getAgentPosition(self.index),self.nearestBorder)<shortestDistance:
                  shortestDistance=self.getMazeDistance(nextState.getAgentPosition(self.index),self.nearestBorder)
                  bestAction=action
              return bestAction
      else:
        actions=state.getLegalActions(self.index)
        shortestDistance=float('inf')
        bestAction='Stop'
        for action in actions:
          nextState=self.getSuccessor(state,action)
          if nextState.getAgentPosition(self.index)[0]>self.boundary:
              continue
          if self.getMazeDistance(nextState.getAgentPosition(self.index),self.nearestBorder)<shortestDistance:
            shortestDistance=self.getMazeDistance(nextState.getAgentPosition(self.index),self.nearestBorder)
            bestAction=action
        return bestAction



  def findMaxFoodPathIn15Steps(self,state):
    # eat as many food as pacman can in 15 steps
    # heuristic search
    return self.aStarSearch(state)


  def getDirection(self,previousNode,endState,startState,relations):
    currentState=endState
    directions=[]
    while currentState!=startState:
      directions.insert(0,relations[(previousNode[currentState],currentState)])
      currentState=previousNode[currentState]
    return directions

  def aStarSearch(self,state):
    startState=state
    visited=set()
    visited.add(startState)
    cost={}
    relations={}
    previousNode={}
    resultDirection=[]
    queue=util.PriorityQueue()
    queue.push(startState,self.foodHeuristic(startState))
    queueItem=set()
    queueItem.add(startState)
    cost[startState]=self.foodHeuristic(startState)
    while not queue.isEmpty():
      currentState=queue.pop()
      queueItem.remove(currentState)
      if self.isGoalState(state,currentState):
        resultDirection=self.getDirection(previousNode,currentState,state,relations)
        return resultDirection
      else:
        visited.add(currentState)
        for action in currentState.getLegalActions(self.index):
          nextState=currentState.generateSuccessor(self.index,action)

          actionList=self.getDirection(previousNode,currentState,state,relations)

          soFarCost=len(actionList)+self.foodHeuristic(nextState)+1
          if nextState not in visited and nextState not in queueItem:
            cost[nextState]=soFarCost
            queue.push(nextState,cost[nextState])
            queueItem.add(nextState)
            previousNode[nextState]=currentState
            relations[(currentState,nextState)]=action
            visited.add(nextState)
          elif nextState in queueItem and cost[nextState]>soFarCost:
            cost[nextState]=soFarCost
            queue.update(nextState,cost[nextState])
            previousNode[nextState]=currentState
            relations[(currentState,nextState)]=action
    return resultDirection

  def isGoalState(self,state,currentState):
    oldFoodList=self.getFood(state).asList()
    newFoodList=self.getFood(currentState).asList()
    if len(oldFoodList)-len(newFoodList)==1:
      return True
    return False

  def foodHeuristic(self,state):
    foods=self.getFood(state)
    foodsList=foods.asList()
    longDistance=-float('inf')
    for food in foodsList:
      distance=self.getMazeDistance(state.getAgentPosition(self.index),food)
      if distance>longDistance:
        longDistance=distance
    if len(foodsList)==0:
      return 0
    return longDistance

  def changeTargetFood(self,state):
    foods=self.getFood(state).asList()
    totalDistance=0
    for food in foods:
      if food==self.nearestFood:
        foods.remove(food)
        continue
      totalDistance+=self.getMazeDistance(state.getAgentPosition(self.index),food)
    averageDistance=totalDistance/(len(foods)-1)
    for food in foods:
      if self.getMazeDistance(state.getAgentPosition(self.index),food)<averageDistance:
        foods.remove(food)
    self.nearestFood=random.choice(foods)

  def simulation(self,startState,startContainFood,state,depth):
    if depth<self.simulationDepth:
      actions=self.getActions(state)
      action = random.choice(actions)
      nextState=state.generateSuccessor(self.index,action)
      if nextState.getAgentPosition(self.enemyGhostIndex)!=None and nextState.getAgentPosition(self.enemyPacmanIndex)!=None:
        # can observe all enemy
        if self.getMazeDistance(nextState.getAgentPosition(self.enemyPacmanIndex),nextState.getAgentPosition(self.index))<10:
          enemyPacmanAgent=DefensiveReflexAgent(self.enemyPacmanIndex)
          enemyPacmanAgent.registerInitialState(nextState)
          enemyPacmanAction=enemyPacmanAgent.chooseAction(nextState)
          nextState=nextState.generateSuccessor(self.enemyPacmanIndex,enemyPacmanAction)

        self.enemyGhost.registerInitialState(nextState)
        enemyGhostAction=self.enemyGhost.chooseAction(nextState)
        nextState=nextState.generateSuccessor(self.enemyGhostIndex,enemyGhostAction)

        if self.haveEatFood(state,nextState):
          self.containFood+=1
        if self.haveReturnBack(state,nextState):
          self.containFood=0
        if nextState.getAgentPosition(self.index)==self.start:
          return (-1000,action)
        return (self.simulation(startState,startContainFood,nextState,depth+1)[0],action)

      if nextState.getAgentPosition(self.enemyGhostIndex)!=None and nextState.getAgentPosition(self.enemyPacmanIndex)==None:
        # can observe enemy ghost
        self.enemyGhost.registerInitialState(nextState)
        enemyGhostAction=self.enemyGhost.chooseAction(nextState)
        nextState=nextState.generateSuccessor(self.enemyGhostIndex,enemyGhostAction)

        if self.haveEatFood(state,nextState):
          self.containFood+=1
        if self.haveReturnBack(state,nextState):
          self.containFood=0
        if nextState.getAgentPosition(self.index)==self.start:
          return (-1000,action)
        return (self.simulation(startState,startContainFood,nextState,depth+1)[0],action)

      if nextState.getAgentPosition(self.enemyGhostIndex)==None and nextState.getAgentPosition(self.enemyPacmanIndex)!=None:
        # can observe enemy pacman agent
        if self.getMazeDistance(nextState.getAgentPosition(self.enemyPacmanIndex),nextState.getAgentPosition(self.index))<10:
          enemyPacmanAgent=DefensiveReflexAgent(self.enemyPacmanIndex)
          enemyPacmanAgent.registerInitialState(nextState)
          enemyPacmanAction=enemyPacmanAgent.chooseAction(nextState)
          nextState=nextState.generateSuccessor(self.enemyPacmanIndex,enemyPacmanAction)

        if self.haveEatFood(state,nextState):
          self.containFood+=1
        if self.haveReturnBack(state,nextState):
          self.containFood=0
        if nextState.getAgentPosition(self.index)==self.start:
          return (-1000,action)
        return (self.simulation(startState,startContainFood,nextState,depth+1)[0],action)

      if nextState.getAgentPosition(self.enemyGhostIndex)==None and nextState.getAgentPosition(self.enemyPacmanIndex)==None:
        # cannot observe any enemy
        if self.haveEatFood(state,nextState):
          self.containFood+=1
        if self.haveReturnBack(state,nextState):
          self.containFood=0
        if nextState.getAgentPosition(self.index)==self.start:
          return (-1000,action)
        return (self.simulation(startState,startContainFood,nextState,depth+1)[0],action)
    else:
      return (self.evaluation(startState,startContainFood,state),None)

  def haveReturnBack(self,state,nextState):
    if self.getScore(state)<self.getScore(nextState):
      return True
    return False

  def haveEatFood(self,state,nextState):
    oldFoodList,newFoodList=self.getFood(state).asList(),self.getFood(nextState).asList()
    if len(oldFoodList)>len(newFoodList):
      return True
    return False

  def evaluation(self,state,startContainFood,nextState):
    reward,oldFoodList,newFoodList=0,self.getFood(state).asList(),self.getFood(nextState).asList()
    oldCapsule,newCapsule=self.getCapsules(state),self.getCapsules(nextState)
    oldBorderDistance,newBorderDistance=self.getMazeDistance(self.nearestBorder,state.getAgentPosition(self.index)),self.getMazeDistance(self.nearestBorder,nextState.getAgentPosition(self.index))
    oldShortDistance,newShortDistance=self.getMazeDistance(self.nearestFood,state.getAgentPosition(self.index)),self.getMazeDistance(self.nearestFood,nextState.getAgentPosition(self.index))
    oldEnemyPacmanPosition,newEnemyPacmanPosition,oldEnemyGhostPosition,oldPacmanPosition,newPacmanPosition=state.getAgentPosition(self.enemyPacmanIndex),nextState.getAgentPosition(self.enemyPacmanIndex),state.getAgentPosition(self.enemyGhostIndex) ,state.getAgentPosition(self.index),nextState.getAgentPosition(self.index)
    if newPacmanPosition==self.start:
      # been eaten by enemy
      reward=-1000
    else:
      # survive
      if len(oldCapsule)>len(newCapsule):
        # eat capsule
        reward+=500
      if self.getScore(state)<self.getScore(nextState):
          # get higher score <=> storing food in home
          reward+=100*self.containFood
      else:
        # still wondering without storing food
        if startContainFood<3:
          # eat few food, might focus on eating more food
          if len(oldFoodList)>len(newFoodList):
            # eat a food
            reward+=70*(self.containFood-startContainFood)
          else:
            # not eat a food
            if oldShortDistance<newShortDistance:
              # move away from food
              if oldEnemyGhostPosition!=None and oldEnemyPacmanPosition!=None:
                if self.getMazeDistance(oldEnemyGhostPosition,oldPacmanPosition)<2 or self.getMazeDistance(oldEnemyPacmanPosition,oldPacmanPosition)<2:
                  # when enemy is coming, escape will not result in negative reward
                  return 0
                else:
                  # enemy far away from pacman, but move away from food
                  return -50*(newShortDistance-oldShortDistance)

              # move away from food
              if oldEnemyGhostPosition==None and oldEnemyPacmanPosition!=None:
                if self.getMazeDistance(oldEnemyPacmanPosition,oldPacmanPosition)<2:
                  # when enemy is coming, escape will not result in negative reward
                  return 0
                else:
                  # enemy far away from pacman, but move away from food
                  return -50*(newShortDistance-oldShortDistance)

              # move away from food
              if oldEnemyGhostPosition!=None and oldEnemyPacmanPosition==None:
                if self.getMazeDistance(oldEnemyGhostPosition,oldPacmanPosition)<2:
                  # when enemy is coming, escape will not result in negative reward
                  return 0
                else:
                  # enemy far away from pacman, but move away from food
                  return -50*(newShortDistance-oldShortDistance)

                # no enemy nearby, but move away from food
              return -50*(newShortDistance-oldShortDistance)

            else:
              # move toward food
              return 15*(oldShortDistance-newShortDistance)
        else:

          if len(oldFoodList)>len(newFoodList):
            # eat a food
            reward+=40*(self.containFood-startContainFood)

          if newBorderDistance<oldBorderDistance:
            reward+=10*(oldBorderDistance-newBorderDistance)


          # # eat much food, should go back and storing them
          # reward-=10*(newBorderDistance-oldBorderDistance)
    return reward







