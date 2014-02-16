# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
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
        newGhostPos = successorGameState.getGhostPositions()
        "***YOUR CODE HERE***"
        import math
        score = 0;
        distances = [manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]
        minFood = (1.0/min(distances)) if len(newFood.asList()) > 0 else 1 #closer the distance, bigger the number
        newGhostPos = successorGameState.getGhostPositions()
        minGhost = min([manhattanDistance(newPos, ghostpos) for ghostpos in newGhostPos]) #farther the distance, bigger the number
        if minGhost == minFood: #don't head towards food if a ghost is there
          score = -2
        return score + math.sqrt(minFood * minGhost) + successorGameState.getScore() 

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
    # Starts off the minimax process (essentially max_value)
    def start_minimax(self, state, action, agentIndex, currentDepth, depth, ghosts):
      if state.isWin() or state.isLose():
        return state.evaluationFunction(state)
      successor = state.generateSuccessor(agentIndex, action)
      v = self.min_value(ghosts, successor, 1, currentDepth, depth)
      return v 

    #Maximizer for our minimax algo

    def max_value(self, state, agentIndex, currentDepth, depth, ghosts):
      if (currentDepth > depth - 1 or state.isLose() or state.isWin()): # if we have already won
        return self.evaluationFunction(state)
      legal_actions = state.getLegalActions(agentIndex)
      v = float("-inf")
      successors = [state.generateSuccessor(agentIndex, action) for action in legal_actions]
      if len(successors) == 0 or state.isLose() or state.isWin():
        return self.evaluationFunction(state)
      for successor in successors:
        v = max(v, self.min_value(ghosts, successor, 1, currentDepth, depth)) # pass in the successor, the ghosts and the currentDepth 
      return v 

    # Minimizer for our minimax algo.

    def min_value(self, ghosts, state, agentIndex, currentDepth, depth):
      if (currentDepth > depth or state.isLose() or state.isWin()): # if we have already won
        return self.evaluationFunction(state)
      if agentIndex < ghosts - 1:
        legal_actions = state.getLegalActions(agentIndex)
        successors = [state.generateSuccessor(agentIndex, action) for action in legal_actions]
        v = float("inf")
        for successor in successors:
          v = min(v, self.min_value(ghosts, successor, agentIndex + 1, currentDepth, depth))
        return v
      else:
        legal_actions = state.getLegalActions(agentIndex)
        successors = [state.generateSuccessor(agentIndex, action) for action in legal_actions]
        v = float("inf")
        for successor in successors:
          v = min(v, self.max_value(successor, 0, currentDepth + 1, depth, ghosts))
        return v
    
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
        depth = self.depth 
        actions = gameState.getLegalActions(0)
        action_costs = {}
        if depth > 0:
         for action in actions:
            action_costs[self.start_minimax(gameState, action, 0, 0, depth, gameState.getNumAgents())] = action
        return action_costs[max(action_costs)]
                

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    # Starts off the minimax process (essentially max_value)

    def prune_start_minimax(self, state, action, agentIndex, currentDepth, depth, ghosts, a, b):
      if state.isWin() or state.isLose():
        return state.evaluationFunction(state)
      successor = state.generateSuccessor(agentIndex, action)
      v = self.prune_min_value(ghosts, successor, 1, currentDepth, depth, a, b)
      return v 

    # Maximizer for our minimax algo
    
    def prune_max_value(self, state, agentIndex, currentDepth, depth, ghosts, a, b):
      if (currentDepth > depth - 1 or state.isLose() or state.isWin()): 
        return self.evaluationFunction(state)
      legal_actions = state.getLegalActions(agentIndex)
      v = float("-inf")
      if len(legal_actions) == 0 or state.isLose() or state.isWin():
        return self.evaluationFunction(state)
      for action in legal_actions:
        v = max(v, self.prune_min_value(ghosts, state.generateSuccessor(agentIndex, action), 1, currentDepth, depth, a, b))
        if v > b:
          return v
        a = max(a, v) 
      return v 
   
    # Minimizer for our minimax algo.
    
    def prune_min_value(self, ghosts, state, agentIndex, currentDepth, depth, a, b):
      if (currentDepth > depth or state.isLose() or state.isWin()): 
        return self.evaluationFunction(state)
      legal_actions = state.getLegalActions(agentIndex)
      v = float("inf")
      for action in legal_actions:
        v = min(v, self.prune_min_value(ghosts, state.generateSuccessor(agentIndex, action), agentIndex + 1, currentDepth, depth, a, b)) if agentIndex < ghosts - 1 else min(v, self.prune_max_value(state.generateSuccessor(agentIndex, action), 0, currentDepth + 1, depth, ghosts, a, b))
        if v < a:
          return v
        b = min(b,v)
      return v
    def getAction(self, gameState):
        
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        depth = self.depth 
        actions = gameState.getLegalActions(0)
        action_costs = {}
        a = float("-inf")
        b = float("inf")
        v = float("-inf")
        if depth > 0:
          for action in actions:
            val = self.prune_start_minimax(gameState, action, 0, 0, depth, gameState.getNumAgents(), a, b)
            v = max(v, val)
            action_costs[val] = action
            if v > b:
              break;
            a = max(a, v)
        return action_costs[max(action_costs)]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    # Starts off the minimax process (essentially max_value)

    def start_minimax(self, state, action, agentIndex, currentDepth, depth, ghosts):
      if state.isWin() or state.isLose():
        return state.evaluationFunction(state)
      successor = state.generateSuccessor(agentIndex, action)
      v = self.exp_value(ghosts, successor, 1, currentDepth, depth)
      return v 
    
    # Maximizer for our minimax algo
   
    def max_value(self, state, agentIndex, currentDepth, depth, ghosts):
      if (currentDepth > depth - 1 or state.isLose() or state.isWin()): # if we have already won
        return self.evaluationFunction(state)
      legal_actions = state.getLegalActions(agentIndex)
      v = float("-inf")
      successors = [state.generateSuccessor(agentIndex, action) for action in legal_actions]
      if len(successors) == 0 or state.isLose() or state.isWin():
        return self.evaluationFunction(state)
      for successor in successors:
        v = max(v, self.exp_value(ghosts, successor, 1, currentDepth, depth)) # pass in the successor, the ghosts and the currentDepth 
      return v 
    
    # Minimizer for our minimax algo.
    
    def exp_value(self, ghosts, state, agentIndex, currentDepth, depth):
      if (currentDepth > depth or state.isLose() or state.isWin()): # if we have already won
        return self.evaluationFunction(state)
      legal_actions = state.getLegalActions(agentIndex)
      successors = [state.generateSuccessor(agentIndex, action) for action in legal_actions]
      v = 0
      prob = (1.0/len(successors))
      for successor in successors:
        v += prob * self.exp_value(ghosts, successor, agentIndex + 1, currentDepth, depth) if agentIndex < ghosts - 1 else prob * self.max_value(successor, 0, currentDepth + 1, depth, ghosts)
      return v

    
    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        depth = self.depth 
        actions = gameState.getLegalActions(0)
        action_costs = {}
        if depth > 0:
          for action in actions:
            action_costs[self.start_minimax(gameState, action, 0, 0, depth, gameState.getNumAgents())] = action
        return action_costs[max(action_costs)]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: We made the minimum distance to food a main priority, but made sure to place a low priority on 
      going to the food if a ghost was there. we also took into consideration how scared the ghosts would be so our
      pacman would eat the power pellets
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    newGhostPos = currentGameState.getGhostPositions()
    score = 0;
    distances = [manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]
    minFood = (1.0/min(distances)) if len(newFood.asList()) > 0 else 1 #closer the distance, bigger the number
    newGhostPos = currentGameState.getGhostPositions()
    minGhost = min([manhattanDistance(newPos, ghostpos) for ghostpos in newGhostPos]) #farther the distance, bigger the number
    if minGhost == minFood: #don't head towards food if a ghost is there
        score = -30
    return score + minFood*2 + min(newScaredTimes) + currentGameState.getScore() 

# Abbreviation
better = betterEvaluationFunction
