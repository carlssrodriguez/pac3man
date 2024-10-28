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

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
      """
      A more advanced evaluation function to guide Pacman based on several game state factors.
      
      The function takes in the current and proposed successor GameState and returns a score 
      representing how favorable the new state is for Pacman.
      """
      # Generate the successor state after applying the action
      successorGameState = currentGameState.generatePacmanSuccessor(action)
      newPos = successorGameState.getPacmanPosition()  # New position of Pacman
      newFood = successorGameState.getFood()  # Remaining food
      newGhostStates = successorGameState.getGhostStates()  # Current state of the ghosts
      newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]  # Scared timers for ghosts

      # Start the score with the base score from the successor state
      score = successorGameState.getScore()

      # Consider the distance to the nearest food
      foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
      if foodDistances:
          score += 10 / min(foodDistances)  # Favor closer food by increasing score

      # Factor in ghost positions and whether they are scared or not
      for i, ghostState in enumerate(newGhostStates):
          ghostPos = ghostState.getPosition()
          ghostDistance = manhattanDistance(newPos, ghostPos)

          if newScaredTimes[i] > 0:
              # If the ghost is scared, Pacman is encouraged to approach it
              if ghostDistance > 0:
                  score += 200 / ghostDistance
          else:
              # If the ghost is not scared, Pacman should avoid getting too close
              if ghostDistance > 0:
                  score -= 10 / ghostDistance

      # Discourage stopping unless necessary by applying a penalty
      if action == Directions.STOP:
          score -= 50

      return score

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
    Minimax agent for adversarial search in Pacman.
    """

    def getAction(self, gameState):
        """
        Returns the best action according to the minimax algorithm.
        Uses self.depth to determine the search depth, and self.evaluationFunction 
        to evaluate game states at the leaves.
        """
        # Start minimax from Pacman (agentIndex = 0)
        return self.minimax(gameState, 0, 0)[1]

    def minimax(self, gameState, depth, agentIndex):
        """
        Applies minimax recursively. Returns a tuple: (value, action).
        """
        # If we've reached the depth limit or a terminal state (win/loss)
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None

        # Figure out how many agents are in the game
        num_agents = gameState.getNumAgents()

        # If it's Pacman's turn (maximize score)
        if agentIndex == 0:
            return self.maxValue(gameState, depth)

        # If it's a ghost's turn (minimize score)
        else:
            return self.minValue(gameState, depth, agentIndex)

    def maxValue(self, gameState, depth):
        """
        Maximizes Pacman's score by choosing the best possible action.
        """
        max_score = float('-inf')
        best_action = None

        # Check all possible actions for Pacman
        legal_actions = gameState.getLegalActions(0)

        for action in legal_actions:
            successor = gameState.generateSuccessor(0, action)
            # Calculate minimax value for the ghosts' turns
            value, _ = self.minimax(successor, depth, 1)

            # Update if a higher score is found
            if value > max_score:
                max_score = value
                best_action = action

        return max_score, best_action

    def minValue(self, gameState, depth, agentIndex):
        """
        Minimizes the score from the perspective of the ghosts.
        """
        min_score = float('inf')
        best_action = None

        # Get the legal actions for the current ghost
        legal_actions = gameState.getLegalActions(agentIndex)

        for action in legal_actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            # If the last ghost moved, move to Pacman (increase depth)
            if agentIndex == gameState.getNumAgents() - 1:
                value, _ = self.minimax(successor, depth + 1, 0)
            else:
                value, _ = self.minimax(successor, depth, agentIndex + 1)

            # Update if a lower score is found
            if value < min_score:
                min_score = value
                best_action = action

        return min_score, best_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Agent that uses Expectimax search to make decisions, considering probabilistic ghost behavior.
    """

    def getAction(self, gameState):
        """
        Chooses the best action for Pacman based on the expectimax algorithm.
        """
        # Start the expectimax search with Pacman as the first agent
        return self.expectimax(gameState, 0, 0)[1]

    def expectimax(self, gameState, depth, agentIndex):
        """
        Recursively applies expectimax. Returns (value, action).
        """
        # Base case: stop if we've reached the depth limit or hit a terminal state
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None

        # Determine if it's Pacman's turn or a ghost's turn
        if agentIndex == 0:
            return self.maximize(gameState, depth)
        else:
            return self.calculateExpectation(gameState, depth, agentIndex)

    def maximize(self, gameState, depth):
        """
        Maximizes Pacman's score.
        """
        best_score = float('-inf')
        best_move = None

        # Explore all possible moves for Pacman
        for action in gameState.getLegalActions(0):
            next_state = gameState.generateSuccessor(0, action)
            score, _ = self.expectimax(next_state, depth, 1)

            # Update best_score and best_move if a higher score is found
            if score > best_score:
                best_score = score
                best_move = action

        return best_score, best_move

    def calculateExpectation(self, gameState, depth, agentIndex):
        """
        Computes the expected value for ghost actions, assuming random behavior.
        """
        actions = gameState.getLegalActions(agentIndex)
        total_score = 0

        # Sum up the values of all actions, treating each one as equally likely
        for action in actions:
            next_state = gameState.generateSuccessor(agentIndex, action)

            # Move to the next agent, increasing depth if it's the last ghost
            if agentIndex == gameState.getNumAgents() - 1:
                score, _ = self.expectimax(next_state, depth + 1, 0)
            else:
                score, _ = self.expectimax(next_state, depth, agentIndex + 1)

            total_score += score

        # Calculate average (expected) value
        expected_score = total_score / len(actions) if actions else 0
        return expected_score, None


def betterEvaluationFunction(currentGameState):
    """
    A more effective evaluation function for Pacman that considers:
    - Closest food distance
    - Ghost distances (avoiding or chasing based on their state)
    - Distance to capsules (especially useful when ghosts are active)
    - Remaining food dots
    """

    # Obtener la posición de Pacman y detalles del estado actual
    pacmanPos = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    # Comenzamos con el puntaje base del estado
    score = currentGameState.getScore()

    # Distancia a la comida más cercana
    foodList = foodGrid.asList()
    if foodList:
        nearestFoodDist = min([manhattanDistance(pacmanPos, food) for food in foodList])
        score += 10 / nearestFoodDist  # Inverso de la distancia para dar prioridad a la comida cercana

    # Ajustar el puntaje en función de la distancia a los fantasmas
    for ghost in ghostStates:
        ghostPos = ghost.getPosition()
        distToGhost = manhattanDistance(pacmanPos, ghostPos)

        if ghost.scaredTimer > 0:
            # Incentivar acercarse a los fantasmas asustados
            score += 200 / distToGhost if distToGhost > 0 else 200
        else:
            # Evitar fantasmas si no están asustados
            if distToGhost > 0:
                score -= 10 / distToGhost

    # Incentivar recoger cápsulas si hay fantasmas activos en el tablero
    if capsules and any(ghost.scaredTimer == 0 for ghost in ghostStates):
        nearestCapsuleDist = min([manhattanDistance(pacmanPos, cap) for cap in capsules])
        score += 5 / nearestCapsuleDist  # Motivar la recogida de cápsulas cuando sea necesario

    # Penalizar la cantidad de comida restante en el tablero
    foodRemaining = len(foodList)
    score -= 4 * foodRemaining  # Restar puntos si quedan muchos alimentos

    return score

# Abbreviation
better = betterEvaluationFunction

