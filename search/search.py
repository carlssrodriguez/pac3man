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
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    Pacman does not go to all the explored squares on his way to the goal.
    DFS explores a lot of squares that are not part of the final solution.
    Pacman may pass through only a few of the squares he explored, leaving many explored squares outside of the path to the goal.
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    
    # Stack (LIFO)
    fringe = Stack()
    # Initialize the fringe with the starting state of the problem
    fringe.push((problem.getStartState(), [], []))  # (state, actions, visited path)

    # Create a set to hold the visited nodes
    visited = set()

    while not fringe.isEmpty():
        # Pop the current node from the fringe
        current_state, actions, visited_path = fringe.pop()

        # If the current state is the goal, return the actions to reach it
        if problem.isGoalState(current_state):
            return actions

        # If the current state hasn't been visited
        if current_state not in visited:
            # Mark the state as visited
            visited.add(current_state)

            # Expand the current state to get its successors
            for successor, action, step_cost in problem.getSuccessors(current_state):
                # If the successor has not been visited, push it onto the stack
                if successor not in visited:
                    fringe.push((successor, actions + [action], visited_path + [current_state]))

    # If no solution is found, return an empty list
    return []


from util import Queue  

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()

    #  Queue FIFO
    fringe = Queue()
    # Initialize the fringe with the starting state of the problem
    fringe.push((problem.getStartState(), []))  # (state, actions) We dont get visited path because BFS guarantees the shortest path

    # Create a set to hold the visited nodes
    visited = set()

    while not fringe.isEmpty():
        # Pop the current node from the fringe
        current_state, actions = fringe.pop()

        # If the current state is the goal, return the actions to reach it
        if problem.isGoalState(current_state):
            return actions

        # If the current state hasn't been visited
        if current_state not in visited:
            # Mark the state as visited
            visited.add(current_state)

            # Expand the current state to get its successors
            for successor, action, step_cost in problem.getSuccessors(current_state):
                # If the successor has not been visited, push it onto the queue
                if successor not in visited:
                    fringe.push((successor, actions + [action]))

    # If no solution is found, return an empty list
    return []


from util import PriorityQueue

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()

    # Create a PriorityQueue to hold the fringe (the set of states to be explored)
    fringe = PriorityQueue()
    # Initialize the fringe with the starting state of the problem
    # The priority is the total cost (0 for the starting state)
    fringe.push((problem.getStartState(), [], 0), 0)  # (state, actions, cost), priority = cost

    # Create a dictionary to hold the best cost to reach each state
    visited = {}

    while not fringe.isEmpty():
        # Pop the node with the lowest cost from the fringe
        current_state, actions, current_cost = fringe.pop()

        # If this is the goal, return the actions to reach it
        if problem.isGoalState(current_state):
            return actions

        # If the current state hasn't been visited or we found a cheaper way to get there
        if current_state not in visited or current_cost < visited[current_state]:
            # Mark the state as visited with the current cost
            visited[current_state] = current_cost

            # Expand the current state to get its successors
            for successor, action, step_cost in problem.getSuccessors(current_state):
                # Calculate the new cost for reaching this successor
                new_cost = current_cost + step_cost
                # Push the successor to the priority queue with its total cost as the priority
                fringe.push((successor, actions + [action], new_cost), new_cost)

    # If no solution is found, return an empty list
    return []



def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
