# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

def manhattan(start,end):
    xy1 = start
    xy2 = end
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        x=-1
        y=-1
        shortest=1000
        total=1000
        num=0
        for i in newFood:
            y+=1
            for j in i:
                x+=1
                if j:
                    num+=1
                    temp=manhattan((y,x),newPos)
                    if temp<shortest:
                        shortest=temp
            
            x=-1
        total-=shortest
        # ghost nearby
        if manhattan(newPos,successorGameState.getGhostPositions()[0])<=1:
            return -100+manhattan(newPos,successorGameState.getGhostPositions()[0])
        # not moving
        if currentGameState.getPacmanPosition()==newPos:
            return 0
        # eating a food
        if currentGameState.getNumFood()>successorGameState.getNumFood():
            return 1001

        return total

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        bestScore,bestAction=self.recursiveValues(gameState,'max',0,-1)
        if bestAction!='':
            return bestAction
        return 'Stop'
        # util.raiseNotDefined()


    def recursiveValues(self,gameState,type,num,depth):
        if type=='max':
            depth+=1
            # check leaf
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return (self.evaluationFunction(gameState),'')

            # next recursion
            nextCost=-1000
            nextAction=''
            for action in gameState.getLegalActions(0):
                if action!='Stop':
                    cost,oneAction=self.recursiveValues(gameState.generateSuccessor(0,action),'min',1,depth)
                    if cost>nextCost:
                        nextCost=cost
                        nextAction=action
            return nextCost,nextAction
        elif type=='min':
            # check leaf
            if gameState.isWin() or gameState.isLose():
                return (self.evaluationFunction(gameState),'')

            # next recursion
            nextCost=1000
            nextAction=''
            for action in gameState.getLegalActions(num):
                if action!='Stop':
                    if num<gameState.getNumAgents()-1:
                        cost,oneAction=self.recursiveValues(gameState.generateSuccessor(num,action),'min',num+1,depth)
                    else:
                        cost,oneAction=self.recursiveValues(gameState.generateSuccessor(num,action),'max',0,depth)
                    if cost<nextCost:
                        nextCost=cost
                        nextAction=action
            return nextCost,nextAction



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        bestScore,bestAction=self.recursiveValues(gameState,'max',0,-1,-1000,1000)
        if bestAction!='':
            return bestAction
        return 'Stop'
        # util.raiseNotDefined()

    def recursiveValues(self,gameState,type,num,depth,alpha,beta):
        if type=='max':
            depth+=1
            # check leaf
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return (self.evaluationFunction(gameState),'')

            # next recursion
            nextCost=-1000
            nextAction=''
            for action in gameState.getLegalActions(0):
                if action!='Stop':
                    cost,oneAction=self.recursiveValues(gameState.generateSuccessor(0,action),'min',1,depth,alpha,beta)
                    if cost>nextCost:
                        nextCost=cost
                        nextAction=action

                    # alpha-beta pruning
                    if nextCost>beta:
                        return (nextCost,action)
                    alpha=max(alpha,nextCost)

            return nextCost,nextAction
        elif type=='min':
            # check leaf
            if gameState.isWin() or gameState.isLose():
                return (self.evaluationFunction(gameState),'')

            # next recursion
            nextCost=1000
            nextAction=''
            for action in gameState.getLegalActions(num):
                if action!='Stop':
                    if num<gameState.getNumAgents()-1:
                        cost,oneAction=self.recursiveValues(gameState.generateSuccessor(num,action),'min',num+1,depth,alpha,beta)
                    else:
                        cost,oneAction=self.recursiveValues(gameState.generateSuccessor(num,action),'max',0,depth,alpha,beta)
                    if cost<nextCost:
                        nextCost=cost
                        nextAction=action

                    # alpha-beta pruning
                    if nextCost<alpha:
                        return (nextCost,action)
                    beta=min(beta,nextCost)

            return nextCost,nextAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        bestScore,bestAction=self.recursiveValues(gameState,'max',0,-1)
        if bestAction!='':
            return bestAction
        return 'Stop'
        # util.raiseNotDefined()

    def recursiveValues(self,gameState,type,num,depth):
        if type=='max':
            depth+=1
            # check leaf
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return (self.evaluationFunction(gameState),'')

            # next recursion
            nextCost=-1000
            nextAction=''
            for action in gameState.getLegalActions(0):
                if action!='Stop':
                    cost,oneAction=self.recursiveValues(gameState.generateSuccessor(0,action),'expect',1,depth)
                    if cost>nextCost:
                        nextCost=cost
                        nextAction=action
            return nextCost,nextAction
        elif type=='expect':
            # check leaf
            if gameState.isWin() or gameState.isLose():
                return (self.evaluationFunction(gameState),'')

            # next recursion
            nextCost=0
            allActions=0
            for action in gameState.getLegalActions(num):
                if action!='Stop':
                    if num<gameState.getNumAgents()-1:
                        cost,oneAction=self.recursiveValues(gameState.generateSuccessor(num,action),'expect',num+1,depth)
                    else:
                        cost,oneAction=self.recursiveValues(gameState.generateSuccessor(num,action),'max',0,depth)
                    nextCost+=cost
                    allActions+=1
            average=(nextCost*1.0)/allActions
            return average,''

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    x=-1
    y=-1
    total=0
    shortest=1000
    path=1000
    num=0
    newPos=currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    for i in newFood:
        y+=1
        for j in i:
            x+=1
            if j:
                num+=1
                temp=manhattan((y,x),newPos)
                if temp<shortest:
                    shortest=temp

        x=-1
    if num<=1:
        shortest=0
    newCapsules=currentGameState.getCapsules()
    minDisToCap=1000
    for capsule in newCapsules:
        disToCap=manhattan(capsule,newPos)
        if disToCap<minDisToCap:
            minDisToCap=disToCap
    path-=shortest
    minDisToGhost=1000
    for i in range(1,currentGameState.getNumAgents()):
        disToGhost=manhattan(currentGameState.getPacmanPosition(),currentGameState.getGhostPositions()[i-1])
        if disToGhost<=1:
            return -1000
        if minDisToGhost>disToGhost:
            minDisToGhost=disToGhost
    powerOn=(2-len(currentGameState.getCapsules()))*200
    if len(currentGameState.getCapsules())==0:
        minDisToCap=0
    total=path+(100000-num*20)+(100000-minDisToCap*10)+powerOn+(100000-minDisToGhost*4)
    return total
    # util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

