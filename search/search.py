# search.py
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]



from util import Stack

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first (Depth-First Search).
    Returns a list of actions to reach the goal.
    """
    # Use a stack to manage the search frontier (LIFO)
    stack = Stack()
    # Push the start state, an empty action list, and an empty visited path
    stack.push((problem.getStartState(), [], []))  # (state, actions, visited path)

    # A set to track visited states
    visited = set()

    while not stack.isEmpty():
        # Pop the current state from the stack
        current_state, actions, visited_path = stack.pop()

        # Check if we've reached the goal
        if problem.isGoalState(current_state):
            return actions

        # If the state has not been visited, explore it
        if current_state not in visited:
            visited.add(current_state)

            # Expand and explore the successors
            for successor, action, step_cost in problem.getSuccessors(current_state):
                if successor not in visited:
                    stack.push((successor, actions + [action], visited_path + [current_state]))

    return []  # Return empty if no solution is found



from util import Queue

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first (Breadth-First Search).
    Returns a list of actions to reach the goal.
    """
    # Use a queue to manage the search frontier (FIFO)
    queue = Queue()
    queue.push((problem.getStartState(), []))  # (state, actions)

    # A set to track visited states
    visited = set()

    while not queue.isEmpty():
        # Pop the current state from the queue
        current_state, actions = queue.pop()

        # Check if we've reached the goal
        if problem.isGoalState(current_state):
            return actions

        # If the state has not been visited, explore it
        if current_state not in visited:
            visited.add(current_state)

            # Expand and explore the successors
            for successor, action, step_cost in problem.getSuccessors(current_state):
                if successor not in visited:
                    queue.push((successor, actions + [action]))

    return []  # Return empty if no solution is found



from util import PriorityQueue

def uniformCostSearch(problem):
    """
    Search the node of least total cost first (Uniform Cost Search).
    Guarantees the least cost solution.
    """
    # Priority queue for managing the search frontier
    priority_queue = PriorityQueue()
    # Push the start state with a cost of 0
    priority_queue.push((problem.getStartState(), [], 0), 0)  # (state, actions, cost)

    # Dictionary to track the best cost to reach each state
    visited = {}

    while not priority_queue.isEmpty():
        # Pop the node with the lowest cost
        current_state, actions, current_cost = priority_queue.pop()

        # Check if we've reached the goal
        if problem.isGoalState(current_state):
            return actions

        # Explore the state only if it's a new or cheaper path
        if current_state not in visited or current_cost < visited[current_state]:
            visited[current_state] = current_cost

            # Expand and explore the successors
            for successor, action, step_cost in problem.getSuccessors(current_state):
                new_cost = current_cost + step_cost
                priority_queue.push((successor, actions + [action], new_cost), new_cost)

    return []  # Return empty if no solution is found


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

from util import PriorityQueue

def aStarSearch(problem, heuristic=nullHeuristic):
    """
    Search the node that has the lowest combined cost and heuristic first (A* Search).
    """
    # Priority queue for managing the search frontier
    priority_queue = PriorityQueue()
    # Push the start state with cost + heuristic as the priority
    start_state = problem.getStartState()
    priority_queue.push((start_state, [], 0), heuristic(start_state, problem))

    # Dictionary to track the best cost to reach each state
    visited = {}

    while not priority_queue.isEmpty():
        # Pop the node with the lowest cost + heuristic
        current_state, actions, current_cost = priority_queue.pop()

        # Check if we've reached the goal
        if problem.isGoalState(current_state):
            return actions

        # Explore the state only if it's a new or cheaper path
        if current_state not in visited or current_cost < visited[current_state]:
            visited[current_state] = current_cost

            # Expand and explore the successors
            for successor, action, step_cost in problem.getSuccessors(current_state):
                new_cost = current_cost + step_cost
                # Compute the priority using cost + heuristic
                priority = new_cost + heuristic(successor, problem)
                priority_queue.push((successor, actions + [action], new_cost), priority)

    return []  # Return empty if no solution is found


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
