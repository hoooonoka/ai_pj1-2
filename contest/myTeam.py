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
import heapq


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='MonteCarloTreeSearchCaptureAgent', second='MonteCarloTreeSearchCaptureAgent'):
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
                dist = self.getMazeDistance(self.start, pos2)
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
        features['successorScore'] = -len(foodList)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
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


class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)


class MonteCarloTreeSearchCaptureAgent(BasicCaptureAgent):
    def __init__(self, index, timeForComputing=.1):
        self.index = index
        self.red = None
        self.agentsOnTeam = None
        self.distancer = None
        self.observationHistory = []
        self.timeForComputing = timeForComputing
        self.display = None
        self.alpha = 0.2
        self.discount = 0.8
        self.epsilon = 0.2
        self.weights = util.Counter()
        self.features = util.Counter()
        self.simulationNumber = 1000
        self.simulationDepth = 5
        self.nearestFood = None
        self.nearestBorder = None
        self.powerOn = False
        self.mode = 'Offensive'  # Offensive or Defensive
        if self.index == 0 or self.index == 2:
            self.enemyPacmanIndex = 1
            self.enemyGhostIndex = 3
        else:
            self.enemyPacmanIndex = 0
            self.enemyGhostIndex = 2
        self.enemyPacman = OffensiveReflexAgent(self.enemyPacmanIndex)
        self.enemyGhost = DefensiveReflexAgent(self.enemyGhostIndex)
        self.containFood = 0
        self.boundary = 0
        self.border = []
        self.powerLeft = 0
        self.enemyPowerOn = False
        self.enemyPowerLeft = 0
        self.enemyPacmanPosition = None
        self.enemyGhostPosition = None
        self.bestActions = []
        self.history = []
        self.lastPacmanPosition = None
        self.moveToOtherEntrance = False
        self.pathToBorder = []
        self.shortDistanceToBorder = {}
        self.moveToOneEntrance = False
        self.paths = {}
        self.alley = {}
        self.alleyPosition={}

    def getOppositeAction(self, action):
        if action == 'West':
            return 'East'
        if action == 'East':
            return 'West'
        if action == 'North':
            return 'South'
        if action == 'South':
            return 'North'
        return 'Stop'

    def getLastAction(self, lastState, currentState):
        lastPosition = lastState.getAgentPosition(self.index)
        currentPosition = currentState.getAgentPosition(self.index)
        if currentPosition == self.start:
            # been eaten
            return None
        if currentPosition[0] - lastPosition[0] == 1:
            return 'East'
        if currentPosition[0] - lastPosition[0] == -1:
            return 'West'
        if currentPosition[1] - lastPosition[1] == 1:
            return 'North'
        if currentPosition[1] - lastPosition[1] == -1:
            return 'South'
        return 'Stop'

    def setNearestBorder(self, state):
        distance = float('inf')
        position = state.getAgentPosition(self.index)
        for spot in self.border:
            if self.getMazeDistance(position, spot) < distance:
                self.nearestBorder = spot
                distance = self.getMazeDistance(position, spot)

    def changeAnotherEntrance(self, state):
        distance = -float('inf')
        position = state.getAgentPosition(self.index)
        for spot in self.border:
            if self.getMazeDistance(position, spot) > distance:
                self.nearestBorder = spot
                distance = self.getMazeDistance(position, spot)
        print self.nearestBorder

    def setBetterBorder(self, state):
        capsule = self.getCapsulesYouAreDefending(state)
        if len(capsule) > 0:
            distance = float('inf')
            for position in capsule:
                for spot in self.border:
                    if self.paths[(position, spot)] < distance:
                        self.nearestBorder = spot
                        distance = self.paths[(position, spot)]
        else:
            self.nearestBorder = self.border[len(self.border) / 2]

    def setNearestFood(self, state):
        foodList = self.getFood(state).asList()
        distance = float('inf')
        position = state.getAgentPosition(self.index)
        for food in foodList:
            if self.getMazeDistance(position, food) < distance:
                self.nearestFood = food
                distance = self.getMazeDistance(position, food)

    def setPathToBorder(self, gameState):
        for spot in self.border:
            for anotherSpot in self.border:
                if spot == anotherSpot:
                    continue
                else:
                    actionList = self.findShortPathToBorder(gameState, spot, anotherSpot)
                    print spot, anotherSpot, actionList
                    self.shortDistanceToBorder[(spot, anotherSpot)] = actionList

    def getDirection(self, previousNode, endState, startState, relations):
        currentState = endState
        directions = []
        while currentState != startState:
            directions.insert(0, relations[(previousNode[currentState], currentState)])
            currentState = previousNode[currentState]
        return directions

    def isGoalState(self, state, goal):
        if state == goal:
            return True
        return False

    def heuristic(self, position, goal):
        distance = self.getMazeDistance(position, goal)
        return distance

    def precomputeHomeMazeDistance(self, state,startTime):
        distances = {}
        layout = state.data.layout
        allNodes=[]
        if self.red:
            for i in range(self.boundary+1):
                for j in range(state.data.layout.height):
                    if not state.hasWall(i,j):
                        allNodes.append((i,j))
        else:
            for i in range(self.boundary,state.data.layout.width):
                for j in range(state.data.layout.height):
                    if not state.hasWall(i,j):
                        allNodes.append((i,j))
        for source in allNodes:
            dist = {}
            closed = {}

            endTime=time.time()
            if endTime-startTime>=13:
                print 'time closed: use default maze distance'
                distances=self.distancer._distances
                break

            for node in allNodes:
                dist[node] = 1000
            queue = util.PriorityQueue()
            queue.push(source, 0)
            dist[source] = 0
            while not queue.isEmpty():
                node = queue.pop()
                if node in closed:
                    continue
                closed[node] = True
                nodeDist = dist[node]
                adjacent = []
                x, y = node
                if not layout.isWall((x, y + 1)):
                    adjacent.append((x, y + 1))
                if not layout.isWall((x, y - 1)):
                    adjacent.append((x, y - 1))
                if not layout.isWall((x + 1, y)):
                    adjacent.append((x + 1, y))
                if not layout.isWall((x - 1, y)):
                    adjacent.append((x - 1, y))
                for other in adjacent:
                    if not other in dist:
                        continue
                    oldDist = dist[other]
                    newDist = nodeDist + 1
                    if newDist < oldDist:
                        dist[other] = newDist
                        queue.push(other, newDist)
            for target in allNodes:
                distances[(target, source)] = dist[target]
        self.paths = distances
        # print self.paths


    def findShortPathToBorder(self, gameState, position, anotherPosition):
        visited = set()
        visited.add(position)
        cost = {}
        relations = {}
        previousNode = {}
        resultDirection = []
        queue = PriorityQueue()
        queue.push(position, self.heuristic(position, anotherPosition))
        queueItem = set()
        queueItem.add(position)
        cost[position] = self.heuristic(position, anotherPosition)
        while not queue.isEmpty():
            currentState = queue.pop()
            queueItem.remove(currentState)
            if self.isGoalState(currentState, anotherPosition):
                resultDirection = self.getDirection(previousNode, currentState, position, relations)
                return resultDirection
            else:
                visited.add(currentState)
                actions = ['West', 'East', 'South', 'North']
                for action in actions:
                    nextState = self.nextStep(currentState, action)
                    if gameState.hasWall(nextState[0], nextState[1]):
                        continue
                    if not self.positionInHome(nextState):
                        continue
                    actionList = self.getDirection(previousNode, currentState, position, relations)
                    soFarCost = len(actionList) + self.heuristic(nextState, anotherPosition) + 1
                    if nextState not in visited and nextState not in queueItem:
                        cost[nextState] = soFarCost
                        queue.push(nextState, cost[nextState])
                        queueItem.add(nextState)
                        previousNode[nextState] = currentState
                        relations[(currentState, nextState)] = action
                        visited.add(nextState)
                    elif nextState in queueItem and cost[nextState] > soFarCost:
                        cost[nextState] = soFarCost
                        queue.update(nextState, cost[nextState])
                        previousNode[nextState] = currentState
                        relations[(currentState, nextState)] = action
        return resultDirection

    def positionInHome(self, position):
        if self.red:
            if position[0] > self.boundary:
                return False
            return True
        else:
            if position[0] < self.boundary:
                return False
            return True

    def getAvailableAction(self,position,state):
        actions=[]
        nextPosition=self.nextStep(position,'South')
        if nextPosition[0]>=0 and nextPosition[0]<state.data.layout.width and nextPosition[1]>=0 and nextPosition[1]<state.data.layout.height:
            if not state.hasWall(nextPosition[0],nextPosition[1]):
                actions.append('South')
        nextPosition=self.nextStep(position,'North')
        if nextPosition[0]>=0 and nextPosition[0]<state.data.layout.width and nextPosition[1]>=0 and nextPosition[1]<state.data.layout.height:
            if not state.hasWall(nextPosition[0],nextPosition[1]):
                actions.append('North')
        nextPosition=self.nextStep(position,'West')
        if nextPosition[0]>=0 and nextPosition[0]<state.data.layout.width and nextPosition[1]>=0 and nextPosition[1]<state.data.layout.height:
            if not state.hasWall(nextPosition[0],nextPosition[1]):
                actions.append('West')
        nextPosition=self.nextStep(position,'East')
        if nextPosition[0]>=0 and nextPosition[0]<state.data.layout.width and nextPosition[1]>=0 and nextPosition[1]<state.data.layout.height:
            if not state.hasWall(nextPosition[0],nextPosition[1]):
                actions.append('East')
        # if self.red:
        #     nextPosition=self.nextStep(position,'South')
        #     if nextPosition[0]>=0 and nextPosition[0]<=self.boundary and nextPosition[1]>=0 and nextPosition[1]<state.data.layout.height:
        #         if self.positionInHome(nextPosition) and not state.hasWall(nextPosition[0],nextPosition[1]):
        #             actions.append('South')
        #     nextPosition=self.nextStep(position,'North')
        #     if nextPosition[0]>=0 and nextPosition[0]<=self.boundary and nextPosition[1]>=0 and nextPosition[1]<state.data.layout.height:
        #         if self.positionInHome(nextPosition) and not state.hasWall(nextPosition[0],nextPosition[1]):
        #             actions.append('North')
        #     nextPosition=self.nextStep(position,'West')
        #     if nextPosition[0]>=0 and nextPosition[0]<=self.boundary and nextPosition[1]>=0 and nextPosition[1]<state.data.layout.height:
        #         if self.positionInHome(nextPosition) and not state.hasWall(nextPosition[0],nextPosition[1]):
        #             actions.append('West')
        #     nextPosition=self.nextStep(position,'East')
        #     if nextPosition[0]>=0 and nextPosition[0]<=self.boundary and nextPosition[1]>=0 and nextPosition[1]<state.data.layout.height:
        #         if self.positionInHome(nextPosition) and not state.hasWall(nextPosition[0],nextPosition[1]):
        #             actions.append('East')
        # else:
        #     nextPosition=self.nextStep(position,'South')
        #     if nextPosition[0]<state.data.layout.width and nextPosition[0]>=self.boundary and nextPosition[1]>=0 and nextPosition[1]<state.data.layout.height:
        #         if self.positionInHome(nextPosition) and not state.hasWall(nextPosition[0],nextPosition[1]):
        #             actions.append('South')
        #     nextPosition=self.nextStep(position,'North')
        #     if nextPosition[0]<state.data.layout.width and nextPosition[0]>=self.boundary and nextPosition[1]>=0 and nextPosition[1]<state.data.layout.height:
        #         if self.positionInHome(nextPosition) and not state.hasWall(nextPosition[0],nextPosition[1]):
        #             actions.append('North')
        #     nextPosition=self.nextStep(position,'West')
        #     if nextPosition[0]<state.data.layout.width and nextPosition[0]>=self.boundary and nextPosition[1]>=0 and nextPosition[1]<state.data.layout.height:
        #         if self.positionInHome(nextPosition) and not state.hasWall(nextPosition[0],nextPosition[1]):
        #             actions.append('West')
        #     nextPosition=self.nextStep(position,'East')
        #     if nextPosition[0]<state.data.layout.width and nextPosition[0]>=self.boundary and nextPosition[1]>=0 and nextPosition[1]<state.data.layout.height:
        #         if self.positionInHome(nextPosition) and not state.hasWall(nextPosition[0],nextPosition[1]):
        #             actions.append('East')
        return actions

    def precomputeAlley(self,state):
        for i in range(state.data.layout.width):
            for j in range(state.data.layout.height):
                if not state.hasWall(i,j):
                    actions=self.getAvailableAction((i,j),state)
                    if len(actions)==1:
                        # one action can be performed: in alley
                        self.check4Directions((i,j),actions[0],state)


    def check4Directions(self,position,action,state):
        direction=self.getOppositeAction(action)
        alleyPosition={position}
        position=self.nextStep(position,action)
        actions=self.getAvailableAction(position,state)
        alleyLength=1
        while len(actions)==2:
            alleyPosition.add(position) 
            direction=self.getOppositeAction(action)
            actions.remove(direction)
            action=actions[0]
            position=self.nextStep(position,action)
            actions=self.getAvailableAction(position,state)
            alleyLength+=1
        direction=self.getOppositeAction(action)
        self.alley[(position,direction)]=alleyLength
        self.alleyPosition[(position,direction)]=alleyPosition

    def isDeadState(self,state):
        pacmanPosition=state.getAgentPosition(self.index)
        if self.positionInHome(pacmanPosition):
            return False
        # 1 step away from opponent ghost
        enemy = self.getOpponents(state)
        ghosts = []
        for oneEnemy in enemy:
            if not state.getAgentState(oneEnemy).isPacman:
                ghosts.append(oneEnemy)
        for ghost in ghosts:
            enemyGhostPosition=state.getAgentPosition(ghost)
            if enemyGhostPosition!=None:
                isScared = (state.getAgentState(ghost).scaredTimer>0)
                if self.getMazeDistance(pacmanPosition,enemyGhostPosition)==1 and not isScared:
                    return True

        # ghost and pacman in alley
        alley=self.getAlley(self.index,state)
        if alley!=None:
            # in alley
            for ghost in ghosts:
                enemyGhostPosition=state.getAgentPosition(ghost)
                if enemyGhostPosition!=None:
                    isScared = (state.getAgentState(ghost).scaredTimer>0)
                    if self.getMazeDistance(alley,enemyGhostPosition)<=self.getMazeDistance(pacmanPosition,alley) and not isScared:
                        # ghost close to entrance
                        return True

        return False

    def isDeadInstantly(self,state):
        pacmanPosition=state.getAgentPosition(self.index)
        if pacmanPosition==self.start:
            return True
        enemy = self.getOpponents(state)
        ghosts = []
        for oneEnemy in enemy:
            if not state.getAgentState(oneEnemy).isPacman:
                ghosts.append(oneEnemy)
        for ghost in ghosts:
            enemyGhostPosition=state.getAgentPosition(ghost)
            if enemyGhostPosition!=None:
                isScared = (state.getAgentState(ghost).scaredTimer>0)
                if self.getMazeDistance(pacmanPosition,enemyGhostPosition)==1 and not isScared:
                    return True
        return False

    def getAlley(self,agentIndex,state):
        position=state.getAgentPosition(agentIndex)
        if position==None:
            return None
        for alley in self.alley.keys():
            for spot in self.alleyPosition[alley]:
                if position==spot:
                    return alley[0]
        return None


    def findAvailableActions(self,state):
        allActions=self.getActions(state)
        actions=[]
        for action in allActions:
            nextState=state.generateSuccessor(self.index,action)
            if self.isDeadState(nextState):
                continue
            if nextState.getAgentPosition(self.index)==self.start:
                continue
            actions.append(action)
        if actions==[]:
            return ['Stop']
        return actions

    def nextStep(self, position, action):
        if action == 'West':
            return (position[0] - 1, position[1])
        if action == 'East':
            return (position[0] + 1, position[1])
        if action == 'South':
            return (position[0], position[1] - 1)
        if action == 'North':
            return (position[0], position[1] + 1)

    def registerInitialState(self, gameState):
        startTime = time.time()
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        if self.index == 0 or self.index == 1:
            self.mode = 'Offensive'
        else:
            self.mode = 'Defensive'
        self.setNearestFood(gameState)
        if self.red:
            self.boundary = (gameState.data.layout.width - 2) / 2
        else:
            self.boundary = ((gameState.data.layout.width - 2) / 2) + 1
        for i in range(gameState.data.layout.height):
            if gameState.hasWall(self.boundary, i):
                continue
            self.border.append((self.boundary, i))
        self.setNearestBorder(gameState)
        self.precomputeAlley(gameState)
        self.precomputeHomeMazeDistance(gameState,startTime)
        # draw alley - debug use
        # for alley in self.alleyPosition.values():
        #     for spot in alley:
        #         print spot
        #         self.debugDraw(spot,[1,0,0])
        endTime = time.time()
        print 'startUp: ',endTime - startTime  
        # print self.alley
        # exit()

    def getActions(self, state):
        legalActions = state.getLegalActions(self.index)
        legalActions.remove('Stop')
        return legalActions

    def chooseAction(self, state):
        timeBegin = time.time()
        # return to start place
        if self.start == state.getAgentPosition(self.index):
            self.containFood = 0
            self.enemyPowerOn = False
            self.setNearestFood(state)
            if self.mode == 'Offensive':
                self.setNearestBorder(state)
            else:
                self.setBetterBorder(state)

        # check enemy power left
        if self.enemyPowerLeft == 0:
            if self.index == 2 or self.index == 3:
                # switch to defence mode
                self.enemyPowerOn = False
                if self.isInYourHome(state):
                    self.mode = 'Defensive'
        else:
            self.enemyPowerLeft -= 1

        # assign mode to 2 agents
        if self.index == 0 or self.index == 1:
            # decide agent mode
            if self.getScore(state) >= 20:
                lastState = self.getPreviousObservation()
                if lastState != None:
                    oldCapsule, newCapsule = self.getCapsulesYouAreDefending(
                        lastState), self.getCapsulesYouAreDefending(state)
                    if len(oldCapsule) > len(newCapsule):
                        self.mode = 'Offensive'
                    # already gain a high score: switch to defensive mode
                    else:
                        self.mode = 'Defensive'
                else:
                    self.mode = 'Defensive'
            else:
                # score is low currently: switch to offensive mode
                self.mode = 'Offensive'

        #################################### offensive agent ##################################

        # eat enemy if it is only one step away from you
        if self.mode == 'Offensive':
            if self.isInYourHome(state):
                if state.getAgentState(self.enemyPacmanIndex).isPacman:
                    if state.getAgentPosition(self.enemyPacmanIndex) != None:
                        if self.getMazeDistance(state.getAgentPosition(self.index),
                                                state.getAgentPosition(self.enemyPacmanIndex)) == 1:
                            actions = self.getActions(state)
                            for action in actions:
                                nextState = state.generateSuccessor(self.index, action)
                                if self.getMazeDistance(nextState.getAgentPosition(self.index), state.getAgentPosition(
                                        self.enemyPacmanIndex)) == 0 and self.isInYourHome(nextState):
                                    return action
                if state.getAgentState(self.enemyGhostIndex).isPacman:
                    if state.getAgentPosition(self.enemyGhostIndex) != None:
                        if self.getMazeDistance(state.getAgentPosition(self.index),
                                                state.getAgentPosition(self.enemyGhostIndex)) == 1:
                            actions = self.getActions(state)
                            for action in actions:
                                nextState = state.generateSuccessor(self.index, action)
                                if self.getMazeDistance(nextState.getAgentPosition(self.index), state.getAgentPosition(
                                        self.enemyGhostIndex)) == 0 and self.isInYourHome(nextState):
                                    return action

            # power on: super pacman
            # if self.powerLeft > 30:
            #     self.powerLeft -= 1
            #     actions = self.getActions(state)
            #     for action in actions:
            #         nextState = state.generateSuccessor(self.index, action)
            #         if self.getMazeDistance(state.getAgentPosition(self.index),
            #                                 self.nearestFood) > self.getMazeDistance(
            #             nextState.getAgentPosition(self.index), self.nearestFood):
            #             oldFoodList, newFoodList = self.getFood(state).asList(), self.getFood(nextState).asList()
            #             if len(oldFoodList) > len(newFoodList):
            #                 self.containFood += 1
            #                 self.setNearestFood(nextState)
            #                 self.setNearestBorder(nextState)
            #             return action
            #     return action

            # go to another entrance
            if self.moveToOtherEntrance == True:
                if self.isGoalState(state.getAgentPosition(self.index), self.nearestBorder):
                    self.moveToOtherEntrance = False
                    self.setNearestFood(state)
                else:
                    actions = self.getActions(state)
                    bestAction=None
                    for action in actions:
                        nextState = state.generateSuccessor(self.index, action)
                        if self.isInYourHome(nextState):
                            if self.paths[nextState.getAgentPosition(self.index), self.nearestBorder] < self.paths[
                                state.getAgentPosition(self.index), self.nearestBorder]:
                                return action
                    self.moveToOtherEntrance=False
                    self.setNearestFood(state)

            # change another route if pacman cannot attack in one entrance: go to another entrance and try attack
            if len(self.history) > 8:
                if self.isInYourHome(state):
                    oldPosition = self.history[len(self.history) - 7][0]
                    oldTargetFood = self.history[len(self.history) - 7][1]
                    if self.nearestFood == oldTargetFood:
                        if self.getMazeDistance(oldPosition, state.getAgentPosition(self.index)) < 3:
                            if self.getMazeDistance(oldTargetFood, oldPosition) - self.getMazeDistance(self.nearestFood,
                                                                                                       state.getAgentPosition(
                                                                                                               self.index)) < 3:
                                self.moveToOtherEntrance = True
                                self.changeAnotherEntrance(state)
                                actions = self.getActions(state)
                                for action in actions:
                                    nextState = state.generateSuccessor(self.index, action)
                                    if self.isInYourHome(nextState):
                                        if self.paths[nextState.getAgentPosition(self.index), self.nearestBorder] < \
                                                self.paths[state.getAgentPosition(self.index), self.nearestBorder]:
                                            return action
                                self.moveToOtherEntrance=False
                                self.setNearestFood(state)

            ###################################### find best action###########################################
            result = util.Counter()
            nums = util.Counter()
            actions = self.findAvailableActions(state)
            # print actions
            # time.sleep(2) 
            foods = self.containFood
            nearestFood = self.nearestFood
            currentPosition = state.getAgentPosition(self.index)
            timeStart = time.time()

            for i in range(self.simulationNumber):
                timeEnd = time.time()
                if timeEnd - timeStart > 0.800:
                    break
                self.containFood = foods
                self.nearestFood = nearestFood
                record = {}
                (reward, action, depth) = self.simulation(state, foods, state, 0)
                if reward != -1000:
                    reward = reward / depth

                    nextPosition = state.generateSuccessor(self.index, action).getAgentPosition(self.index)
                    if nextPosition == currentPosition and reward >= 0:
                        reward *= 0.8

                    result[action] += reward
                    nums[action] += 1
            bestAction, bestActionReward = None, -float('inf')
            for action in actions:
                if nums[action] == 0:
                    continue
                oneReward = result[action] / nums[action]
                if oneReward > bestActionReward:
                    bestActionReward = oneReward
                    bestAction = action
            if bestAction==None:
                # if enemy agent is rational, our pacman agent will die
                # however, we should not assume that all enemy are rational
                # therefore, we should still try our best to survive
                actions=self.getActions(state)
                surviveActions=[]
                for action in actions:
                    nextState=state.generateSuccessor(self.index,action)
                    if not self.isDeadInstantly(nextState):
                        surviveActions.append(action)
                if len(surviveActions)==0:
                    # no actions can help pacman survive
                    bestAction=random.choice(actions)
                else:
                    bestAction=random.choice(surviveActions)
            ################################# update information ##################################################
            self.containFood = foods
            self.nearestFood = nearestFood
            self.history.append((state.getAgentPosition(self.index), self.nearestFood))
            nextState = state.generateSuccessor(self.index, bestAction)
            self.setNearestBorder(nextState)
            oldFoodList, newFoodList, oldCapsule, newCapsule = self.getFood(state).asList(), self.getFood(
                nextState).asList(), self.getCapsules(state), self.getCapsules(nextState)
            if len(oldFoodList) > len(newFoodList):
                self.containFood += 1
                self.setNearestFood(nextState)

            if len(oldCapsule) > len(newCapsule):
                self.powerOn = True
                self.powerLeft = 40
                self.setNearestFood(nextState)
            else:
                if self.powerOn:
                    self.powerLeft -= 1
                    if self.powerLeft == 0:
                        self.powerOn = False

            if self.getScore(nextState) > self.getScore(state):
                self.containFood = 0
                self.setNearestFood(nextState)

            timeFinish = time.time()
            print timeFinish - timeBegin
            return bestAction


        else:
            ########################################## defensive agent ############################################
            if not self.isInYourHome(state):
                actions = self.getActions(state)
                for action in actions:
                    nextState = self.getSuccessor(state, action)
                    if self.getMazeDistance(nextState.getAgentPosition(self.index),
                                            self.nearestBorder) < self.getMazeDistance(
                        state.getAgentPosition(self.index), self.nearestBorder):
                        return action

            lastState = self.getPreviousObservation()

            if lastState != None:
                # if it is not initial state
                oldCapsule, newCapsule = self.getCapsulesYouAreDefending(lastState), self.getCapsulesYouAreDefending(
                    state)
                if len(oldCapsule) > len(newCapsule):
                    # capsule has been eaten
                    self.enemyPowerLeft = 40
                    self.enemyPowerOn = True
                    for oneCapsule in oldCapsule:
                        if oneCapsule not in newCapsule:
                            self.lastPacmanPosition = oneCapsule

                if self.enemyPowerOn:
                    return self.enemyPowerOnDecision(state,lastState)

                # get enemt and action
                enemy = self.getOpponents(state)
                actions = self.getActions(state)
                oldCapsule, newCapsule = self.getCapsulesYouAreDefending(lastState), self.getCapsulesYouAreDefending(
                    state)
                # load current food
                oldFood, newFood = self.getFoodYouAreDefending(lastState), self.getFoodYouAreDefending(state)
                # put enemys into invaders list
                invaders = []
                for oneEnemy in enemy:
                    if state.getAgentState(oneEnemy).isPacman:
                        invaders.append(oneEnemy)
                # initionalise bestaction stop
                bestAction = 'Stop'
                # declear the shortest distance to border
                shortDistanceToBorder = float('inf')
                # if there is not enemy, the pacman will go to boder
                if len(invaders) == 0:
                    # no invaders
                    # go to best border
                    self.setBetterBorder(state)
                    distance = self.paths[(state.getAgentPosition(self.index), self.nearestBorder)]
                    for action in actions:
                        nextState = self.getSuccessor(state, action)
                        if not self.isInYourHome(nextState):
                            continue
                        if self.paths[(nextState.getAgentPosition(self.index), self.nearestBorder)] < distance:
                            bestAction = action
                            if shortDistanceToBorder > self.paths[
                                (nextState.getAgentPosition(self.index), self.nearestBorder)]:
                                shortDistanceToBorder = self.paths[
                                    (nextState.getAgentPosition(self.index), self.nearestBorder)]

                    return bestAction

                bestPosition = None
                # declear the distance to enemy
                shortDistanceToEnemy = float('inf')

                for invader in invaders:
                    print invader

                    if state.getAgentPosition(invader) != None:
                        # can observe this enemy
                        distance = self.paths[(state.getAgentPosition(invader), state.getAgentPosition(self.index))]

                        enemyToBorderShortestDistance = float('inf')
                        closestBorderPosition = None

                        # get the shortest distance to border
                        for border in self.border:
                            enemyToBorderDistance = self.paths[(state.getAgentPosition(invader), border)]
                            if enemyToBorderDistance < enemyToBorderShortestDistance:
                                enemyToBorderShortestDistance = enemyToBorderDistance
                                closestBorderPosition = border
                        denfencerToClosestBorderPositionDistance = self.paths[
                            (state.getAgentPosition(self.index), closestBorderPosition)]

                        ###
                        self.nearestBorder = closestBorderPosition
                        ####

                        for action in actions:
                            nextState = self.getSuccessor(state, action)
                            if not self.isInYourHome(nextState):
                                continue
                            if self.paths[
                                (nextState.getAgentPosition(self.index), state.getAgentPosition(invader))] == 0:
                                return action

                            # next action to border distance
                            nextStateDenfencerToClosestBorderPositionDistance = self.paths[
                                (nextState.getAgentPosition(self.index), closestBorderPosition)]

                            if nextStateDenfencerToClosestBorderPositionDistance < enemyToBorderShortestDistance:
                                if self.paths[(
                                state.getAgentPosition(invader), nextState.getAgentPosition(self.index))] < distance:

                                    # near to enemy agent
                                    if shortDistanceToEnemy > self.paths[
                                        (state.getAgentPosition(invader), nextState.getAgentPosition(self.index))]:
                                        shortDistanceToEnemy = self.paths[
                                            (state.getAgentPosition(invader), nextState.getAgentPosition(self.index))]

                                        bestAction = action
                                        bestPosition = state.getAgentPosition(invader)
                            else:
                                if self.paths[(closestBorderPosition, nextState.getAgentPosition(
                                        self.index))] < denfencerToClosestBorderPositionDistance:
                                    bestAction = action
                                    bestPosition = state.getAgentPosition(invader)
                if bestAction != 'Stop':
                    nextState = self.getSuccessor(state, action)
                    self.lastPacmanPosition = bestPosition

                    return bestAction
                else:
                    # enemy exist
                    # cannot observe any enemy

                    if self.red:
                        for i in range(self.boundary+1):
                            for j in range(state.data.layout.height):
                                if oldFood[i][j] != newFood[i][j]:
                                    # get the position where enemy eat
                                    self.lastPacmanPosition = (i, j)
                    else:
                        for i in range(self.boundary, state.data.layout.width):
                            for j in range(state.data.layout.height):
                                if oldFood[i][j] != newFood[i][j]:
                                    self.lastPacmanPosition = (i, j)

                    if self.lastPacmanPosition == None:
                        bestAction = 'Stop'
                        shortDistanceToBorder = float('inf')
                        actions = self.getActions(state)
                        distance = self.paths[(self.nearestBorder, state.getAgentPosition(self.index))]
                        for action in actions:
                            nextState = self.getSuccessor(state, action)
                            if not self.isInYourHome(nextState):
                                continue
                            if self.paths[(self.nearestBorder, nextState.getAgentPosition(self.index))] < distance:
                                bestAction = action
                                if shortDistanceToBorder > self.paths[
                                    (self.nearestBorder, nextState.getAgentPosition(self.index))]:
                                    shortDistanceToBorder = self.paths[
                                        (self.nearestBorder, nextState.getAgentPosition(self.index))]

                        return bestAction

                    # distance=self.getMazeDistance(self.lastPacmanPosition,state.getAgentPosition(self.index))
                    # for action in actions:
                    #   nextState=self.getSuccessor(state,action)
                    #   if not self.isInYourHome(nextState):
                    #     continue
                    #   if self.getMazeDistance(self.lastPacmanPosition,nextState.getAgentPosition(self.index))<distance:
                    #     if shortDistanceToEnemy>self.getMazeDistance(self.lastPacmanPosition,nextState.getAgentPosition(self.index)):
                    #       shortDistanceToEnemy=self.getMazeDistance(self.lastPacmanPosition,nextState.getAgentPosition(self.index))
                    #       bestAction=action
                    lastPacmanPositionShortestDistanceToBorder = float('inf')
                    lastPacmanPositionShortestPosition = None

                    for border in self.border:
                        distance = self.paths[(self.lastPacmanPosition, border)]
                        if distance < lastPacmanPositionShortestDistanceToBorder:
                            lastPacmanPositionShortestDistanceToBorder = distance
                            lastPacmanPositionShortestPosition = border

                    ###
                    self.nearestBorder = lastPacmanPositionShortestPosition
                    ####

                    distanceToLastPacmanPositionShortestPosition = self.paths[
                        (state.getAgentPosition(self.index), lastPacmanPositionShortestPosition)]
                    for action in actions:
                        nextState = self.getSuccessor(state, action)
                        if not self.isInYourHome(nextState):
                            continue
                        if self.paths[(lastPacmanPositionShortestPosition, nextState.getAgentPosition(
                                self.index))] < distanceToLastPacmanPositionShortestPosition:
                            bestAction = action

                return bestAction
            # initional state
            else:
                bestAction = 'Stop'
                shortDistanceToBorder = float('inf')
                actions = self.getActions(state)
                distance = self.paths[(self.nearestBorder, state.getAgentPosition(self.index))]
                for action in actions:
                    nextState = self.getSuccessor(state, action)
                    if not self.isInYourHome(nextState):
                        continue
                    if self.paths[(self.nearestBorder, nextState.getAgentPosition(self.index))] < distance:
                        bestAction = action
                        if shortDistanceToBorder > self.paths[
                            (self.nearestBorder, nextState.getAgentPosition(self.index))]:
                            shortDistanceToBorder = self.paths[
                                (self.nearestBorder, nextState.getAgentPosition(self.index))]

                return bestAction

    def enemyPowerOnDecision(self, state,lastState):
        enemy = self.getOpponents(state)
        actions = self.getActions(state)
        oldCapsule, newCapsule = self.getCapsulesYouAreDefending(lastState), self.getCapsulesYouAreDefending(state)
        oldFood, newFood = self.getFoodYouAreDefending(lastState), self.getFoodYouAreDefending(state)
        invaders = []
        for oneEnemy in enemy:
            if state.getAgentState(oneEnemy).isPacman:
                invaders.append(oneEnemy)

        bestAction = 'Stop'

        shortDistanceToBorder = float('inf')
        ####################################################
        if len(invaders) == 0:
            # no invaders
            # go to best border
            self.setBetterBorder(state)
            distance = self.paths[(state.getAgentPosition(self.index), self.nearestBorder)]

            for action in actions:
                nextState = self.getSuccessor(state, action)
                if not self.isInYourHome(nextState):
                    continue
                if self.paths[(nextState.getAgentPosition(self.index), self.nearestBorder)] < distance:
                    bestAction = action
                    if shortDistanceToBorder > self.paths[(nextState.getAgentPosition(self.index), self.nearestBorder)]:
                        shortDistanceToBorder = self.paths[(nextState.getAgentPosition(self.index), self.nearestBorder)]
            return bestAction

        bestPosition = None
        shortDistanceToEnemy = float('inf')
        for invader in invaders:
            if state.getAgentPosition(invader) != None:
                # can observe this enemy
                distance = self.paths[(state.getAgentPosition(invader), state.getAgentPosition(self.index))]

                for action in actions:
                    nextState = self.getSuccessor(state, action)
                    if not self.isInYourHome(nextState):
                        continue
                    if self.paths[(state.getAgentPosition(invader), nextState.getAgentPosition(self.index))] < distance:
                        # near to enemy agent
                        if shortDistanceToEnemy > self.paths[
                            (state.getAgentPosition(invader), nextState.getAgentPosition(self.index))]:
                            shortDistanceToEnemy = self.paths[
                                (state.getAgentPosition(invader), nextState.getAgentPosition(self.index))]
                            bestAction = action
                            bestPosition = state.getAgentPosition(invader)
        if bestAction != 'Stop':
            nextState = self.getSuccessor(state, action)
            self.lastPacmanPosition = bestPosition

            return bestAction
        else:
            # enemy exist
            # cannot observe any enemy

            if self.red:
                for i in range(self.boundary + 1):
                    for j in range(state.data.layout.height):
                        if oldFood[i][j] != newFood[i][j]:
                            self.lastPacmanPosition = (i, j)
            else:
                for i in range(self.boundary, state.data.layout.width):
                    for j in range(state.data.layout.height):
                        if oldFood[i][j] != newFood[i][j]:
                            self.lastPacmanPosition = (i, j)

            if self.lastPacmanPosition == None:
                bestAction = 'Stop'
                shortDistanceToBorder = float('inf')
                actions = self.getActions(state)
                distance = self.paths[(state.getAgentPosition(self.index), self.nearestBorder)]
                for action in actions:
                    nextState = self.getSuccessor(state, action)
                    if not self.isInYourHome(nextState):
                        continue
                    if self.paths[(nextState.getAgentPosition(self.index), self.nearestBorder)] < distance:
                        bestAction = action
                        if shortDistanceToBorder > self.paths[
                            (nextState.getAgentPosition(self.index), self.nearestBorder)]:
                            shortDistanceToBorder = self.paths[
                                (nextState.getAgentPosition(self.index), self.nearestBorder)]
                return bestAction

            distance = self.paths[(self.lastPacmanPosition, state.getAgentPosition(self.index))]
            for action in actions:
                nextState = self.getSuccessor(state, action)
                if not self.isInYourHome(nextState):
                    continue
                if self.paths[(self.lastPacmanPosition, nextState.getAgentPosition(self.index))] < distance:
                    if shortDistanceToEnemy > self.paths[
                        (self.lastPacmanPosition, nextState.getAgentPosition(self.index))]:
                        shortDistanceToEnemy = self.paths[
                            (self.lastPacmanPosition, nextState.getAgentPosition(self.index))]
                        bestAction = action

        return bestAction

    def isInYourHome(self, state):
        position = state.getAgentPosition(self.index)
        if self.red:
            if position[0] > self.boundary:
                return False
            return True
        else:
            if position[0] < self.boundary:
                return False
            return True

    def changeTargetFood(self, state):
        foods = self.getFood(state).asList()
        totalDistance = 0
        for food in foods:
            if food == self.nearestFood:
                foods.remove(food)
                continue
            totalDistance += self.getMazeDistance(state.getAgentPosition(self.index), food)
        averageDistance = totalDistance / (len(foods) - 1)
        for food in foods:
            if self.getMazeDistance(state.getAgentPosition(self.index), food) < averageDistance:
                foods.remove(food)
        self.nearestFood = random.choice(foods)

    def simulation(self, startState, startContainFood, state, depth):
        if depth < self.simulationDepth:
            # actions = self.findAvailableActions(state)
            # action = random.choice(actions)
            action = self.simulationHelper(state)
            if action=='Stop':
                return (-1000,action,depth)
            nextState = state.generateSuccessor(self.index, action)
            if self.getScore(nextState) > self.getScore(state):
                return (100 * self.containFood, action, depth + 1)
            if nextState.getAgentPosition(self.enemyGhostIndex) != None and nextState.getAgentPosition(
                    self.enemyPacmanIndex) != None:
                # can observe all enemy
                if self.getMazeDistance(nextState.getAgentPosition(self.enemyPacmanIndex),
                                        nextState.getAgentPosition(self.index)) < 10:
                    enemyPacmanAgent = DefensiveReflexAgent(self.enemyPacmanIndex)
                    enemyPacmanAgent.registerInitialState(nextState)
                    enemyPacmanAction = enemyPacmanAgent.chooseAction(nextState)
                    nextState = nextState.generateSuccessor(self.enemyPacmanIndex, enemyPacmanAction)

                self.enemyGhost.registerInitialState(nextState)
                enemyGhostAction = self.enemyGhost.chooseAction(nextState)
                nextState = nextState.generateSuccessor(self.enemyGhostIndex, enemyGhostAction)

                if self.haveEatFood(state, nextState):
                    self.containFood += 1
                if self.haveReturnBack(state, nextState):
                    self.containFood = 0
                # if nextState.getAgentPosition(self.index) == self.start:
                #     return (-1000, action, depth)
                return (self.simulation(startState, startContainFood, nextState, depth + 1)[0], action, depth + 1)

            if nextState.getAgentPosition(self.enemyGhostIndex) != None and nextState.getAgentPosition(
                    self.enemyPacmanIndex) == None:
                # can observe enemy ghost
                self.enemyGhost.registerInitialState(nextState)
                enemyGhostAction = self.enemyGhost.chooseAction(nextState)
                nextState = nextState.generateSuccessor(self.enemyGhostIndex, enemyGhostAction)

                if self.haveEatFood(state, nextState):
                    self.containFood += 1
                if self.haveReturnBack(state, nextState):
                    self.containFood = 0
                # if nextState.getAgentPosition(self.index) == self.start:
                #     return (-1000, action, depth)
                return (self.simulation(startState, startContainFood, nextState, depth + 1)[0], action, depth + 1)

            if nextState.getAgentPosition(self.enemyGhostIndex) == None and nextState.getAgentPosition(
                    self.enemyPacmanIndex) != None:
                # can observe enemy pacman agent
                if self.getMazeDistance(nextState.getAgentPosition(self.enemyPacmanIndex),
                                        nextState.getAgentPosition(self.index)) < 10:
                    enemyPacmanAgent = DefensiveReflexAgent(self.enemyPacmanIndex)
                    enemyPacmanAgent.registerInitialState(nextState)
                    enemyPacmanAction = enemyPacmanAgent.chooseAction(nextState)
                    nextState = nextState.generateSuccessor(self.enemyPacmanIndex, enemyPacmanAction)

                if self.haveEatFood(state, nextState):
                    self.containFood += 1
                if self.haveReturnBack(state, nextState):
                    self.containFood = 0
                # if nextState.getAgentPosition(self.index) == self.start:
                #     return (-1000, action, depth)
                return (self.simulation(startState, startContainFood, nextState, depth + 1)[0], action, depth + 1)

            if nextState.getAgentPosition(self.enemyGhostIndex) == None and nextState.getAgentPosition(
                    self.enemyPacmanIndex) == None:
                # cannot observe any enemy
                if self.haveEatFood(state, nextState):
                    self.containFood += 1
                if self.haveReturnBack(state, nextState):
                    self.containFood = 0
                # if nextState.getAgentPosition(self.index) == self.start:
                #     return (-1000, action, depth)
                return (self.simulation(startState, startContainFood, nextState, depth + 1)[0], action, depth + 1)
        else:
            return (self.evaluation(startState, startContainFood, state), None, depth)

    def haveReturnBack(self, state, nextState):
        if self.getScore(state) < self.getScore(nextState):
            return True
        return False

    def haveEatFood(self, state, nextState):
        oldFoodList, newFoodList = self.getFood(state).asList(), self.getFood(nextState).asList()
        if len(oldFoodList) > len(newFoodList):
            return True
        return False

    def simulationHelper(self, state):
        # capsules = self.getCapsules(state)
        # food = self.getFood(state).asList()
        # position = state.getAgentPosition(self.index)
        actions = self.findAvailableActions(state)

        # enemyPacmanPosition,enemyGhostPosition=state.getAgentPosition(self.enemyPacmanIndex),state.getAgentPosition(self.enemyGhostIndex)
        # distanceToEnemyPacman,distanceToEnemyGhost=self.getMazeDistance(enemyPacmanPosition,position),self.getMazeDistance(enemyGhostPosition,position)
        # if self.containFood >= 3:
        #     if util.flipCoin(0.5):
        #         for action in actions:
        #             nextState = state.generateSuccessor(self.index, action)
        #             if self.getMazeDistance(nextState.getAgentPosition(self.index),
        #                                     self.nearestBorder) < self.getMazeDistance(
        #                 state.getAgentPosition(self.index), self.nearestBorder):
        #                 return action

        return random.choice(actions)

    def evaluation(self, state, startContainFood, nextState):
        reward, oldFoodList, newFoodList = 0, self.getFood(state).asList(), self.getFood(nextState).asList()
        oldCapsule, newCapsule = self.getCapsules(state), self.getCapsules(nextState)
        oldBorderDistance, newBorderDistance = self.getMazeDistance(self.nearestBorder, state.getAgentPosition(
            self.index)), self.getMazeDistance(self.nearestBorder, nextState.getAgentPosition(self.index))
        oldShortDistance, newShortDistance = self.getMazeDistance(self.nearestFood, state.getAgentPosition(
            self.index)), self.getMazeDistance(self.nearestFood, nextState.getAgentPosition(self.index))
        oldEnemyPacmanPosition, newEnemyPacmanPosition, oldEnemyGhostPosition, oldPacmanPosition, newPacmanPosition = state.getAgentPosition(
            self.enemyPacmanIndex), nextState.getAgentPosition(self.enemyPacmanIndex), state.getAgentPosition(
            self.enemyGhostIndex), state.getAgentPosition(self.index), nextState.getAgentPosition(self.index)
        if newPacmanPosition == self.start:
            # been eaten by enemy
            reward = -1000
        else:
            # survive
            if self.powerOn == False:
                if len(oldCapsule) > len(newCapsule):
                    # eat capsule
                    reward += 500
                else:
                    if len(oldCapsule) != 0:
                        capsule = oldCapsule[0]
                        distanceToCapsule = self.getMazeDistance(state.getAgentPosition(self.index), capsule)
                        for oneCapsule in oldCapsule:
                            if self.getMazeDistance(state.getAgentPosition(self.index), oneCapsule) < distanceToCapsule:
                                distanceToCapsule = self.getMazeDistance(state.getAgentPosition(self.index), oneCapsule)
                                capsule = oneCapsule

                        if self.getMazeDistance(capsule, state.getAgentPosition(self.index)) > self.getMazeDistance(
                                capsule, nextState.getAgentPosition(self.index)):
                            # move toward capsule
                            return 30 * (self.getMazeDistance(capsule, state.getAgentPosition(
                                self.index)) - self.getMazeDistance(capsule, nextState.getAgentPosition(self.index)))

            if self.getScore(state) < self.getScore(nextState):
                # get higher score <=> storing food in home
                reward += 100 * self.containFood
            else:
                # still wondering without storing food
                if startContainFood < 3:
                    # eat few food, might focus on eating more food
                    if len(oldFoodList) > len(newFoodList):
                        # eat a food
                        reward += 70 * (self.containFood - startContainFood)
                    else:
                        # not eat a food
                        if oldShortDistance < newShortDistance:
                            # move away from food
                            if oldEnemyGhostPosition != None and oldEnemyPacmanPosition != None:
                                if self.getMazeDistance(oldEnemyGhostPosition,
                                                        oldPacmanPosition) < 2 or self.getMazeDistance(
                                    oldEnemyPacmanPosition, oldPacmanPosition) < 2:
                                    # when enemy is coming, escape will not result in negative reward
                                    return 0
                                else:
                                    # enemy far away from pacman, but move away from food
                                    return -50 * (newShortDistance - oldShortDistance)

                            # move away from food
                            if oldEnemyGhostPosition == None and oldEnemyPacmanPosition != None:
                                if self.getMazeDistance(oldEnemyPacmanPosition, oldPacmanPosition) < 2:
                                    # when enemy is coming, escape will not result in negative reward
                                    return 0
                                else:
                                    # enemy far away from pacman, but move away from food
                                    return -50 * (newShortDistance - oldShortDistance)

                            # move away from food
                            if oldEnemyGhostPosition != None and oldEnemyPacmanPosition == None:
                                if self.getMazeDistance(oldEnemyGhostPosition, oldPacmanPosition) < 2:
                                    # when enemy is coming, escape will not result in negative reward
                                    return 0
                                else:
                                    # enemy far away from pacman, but move away from food
                                    return -50 * (newShortDistance - oldShortDistance)

                                # no enemy nearby, but move away from food
                            return -50 * (newShortDistance - oldShortDistance)

                        else:
                            # move toward food
                            return 15 * (oldShortDistance - newShortDistance)
                else:

                    if len(oldFoodList) > len(newFoodList):
                        # eat a food
                        reward += 30 * (self.containFood - startContainFood)

                    if newBorderDistance < oldBorderDistance:
                        # home direction
                        reward += 3.5 * self.containFood * (oldBorderDistance - newBorderDistance)

                    # # eat much food, should go back and storing them
                    # reward-=10*(newBorderDistance-oldBorderDistance)
        return reward







