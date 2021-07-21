"""
Authors: Lucas Barusek
Implementation of the following uninformed Search Algorithms
    - Breadth First search
    - Depth First Search
    - Iterative Depth First search
Additionally, implements Three heuristics, and the following informed searches
    - Best First search
    - A* Search

USAGE:
    'python3 vacuum.py environment search_technique graph_structure'
        - environment: 0 - 11 
        - search_technique: 'BFS', 'DFS', 'IDS', 'GBFS', 'A*', 'all'
        - graph_structure: 'graph' or 'tree'
    Informed searches use heuristic two by default. Must be changed 
    manually
    Additionally, if you want to change total weight function or 
    randomized directions, must be done manually
"""
from sys import argv, maxsize
from copy import deepcopy
from customEnvironment import VacuumEnvironment375, make_vacuum_environment
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
import heapq
import time


class VacuumNode:
    def __init__(self, state, pathCost, path):
        """ Initialize Node Class """
        self._state = state
        self._pathCost = pathCost
        self.path = path
        self.currentCost = 0

    def __lt__(self, otherNode):
        """ Compare nodes based on their heuristic cost """
        return self.currentCost < otherNode.currentCost

    def isGoalState(self):
        """ We have reached the goal state if all the spaces are clean """
        return self._state.all_clean()

    def setCurrentCost(self, cost):
        """ Returns the hueristic cost of the node """
        self.currentCost = cost

    def getPathCost(self):
        """ Returns the path cost """
        return self._pathCost

    def getPath(self):
        """ Returns the directional path taken to the node """
        return self.path

    def getState(self):
        """ Returns the VacuumEnvironment which represents the state """
        return self._state


DIRECTIONS = {(-1, 0): "U",
              (1, 0): "D",
              (0, -1): "L",
              (0, 1): "R"}


def BFS(vacuumState, graph_structure, randomized=False, weight=1):
    """ Implementation of BFS with Graph and Tree Search """

    # initial node is the starting space of the vacuum, which we have
    # already visited
    vacuumNode = VacuumNode(vacuumState, 0, [])
    visited = {vacuumState} # Set to track visited States
    queue = [] #queue for filo data structure
    nodesExpanded = 0
    max_frontier_size = 0
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    while not vacuumNode.isGoalState():
        nodesExpanded += 1
        currentState = vacuumNode.getState()
        # Shuffle nodes if the directions are not control variables
        if randomized: shuffle(directions)
        # Expand the neighbors by creating a copy of the state, and going in 
        # one of the four directions
        for direc in directions:
            copyOfState = deepcopy(currentState)            
            direction = DIRECTIONS[direc]
            copyOfState.move_agent(direction)

            # If we are doing graph search, then only add the node to the frontier 
            # if we havent't visited that state before
            if graph_structure == "graph":
                if copyOfState not in visited:
                    queue.append(VacuumNode(copyOfState, vacuumNode.getPathCost() + 1, 
                                            vacuumNode.getPath() + [direction]))
                    visited.add(copyOfState)
            # If we are doing tree search, add the node regardless
            else: 
                queue.append(VacuumNode(copyOfState, vacuumNode.getPathCost() + 1, 
                                        vacuumNode.getPath() + [direction]))

        # Keep track how large the frontier becomes, and get the next node in the frontier
        # Get the node at the front of the list since BFS uses FILO data structure
        max_frontier_size = max(max_frontier_size, len(queue))
        vacuumNode = queue.pop(0)

    return {'Cost': vacuumNode.getPathCost(), 'Nodes Generated': nodesExpanded, 
            'Space Complexity': max_frontier_size, 'path': vacuumNode.getPath(),
            'Total Cost': nodesExpanded + (weight * vacuumNode.getPathCost())}


def DFS(vacuumState, graph_structure, randomized=False, weight=1):
    """ Implementation of DFS with Graph and Tree Search """

    # initial node is the starting space of the vacuum, which we have
    # already visited
    vacuumNode = VacuumNode(vacuumState, 0, [])
    visited = {vacuumState} # Set to track visited States
    stack = [] # Stack for filo data structure
    nodesExpanded = 0
    max_frontier_size = 0
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    while not vacuumNode.isGoalState():
        # get the neighbors and shuffle their order so we don't always put the nodes
        # in the same order
        nodesExpanded += 1
        currentState = vacuumNode.getState()
        # Shuffle nodes if the directions are not control variables
        if randomized: shuffle(directions)
        for direc in directions:
            # Expand the neighbors by creating a copy of the state, and going in 
            # one of the four directions
            copyOfState = deepcopy(currentState)
            direction = DIRECTIONS[direc]
            copyOfState.move_agent(direction)

            # If we are doing graph search, then only add the node to the frontier 
            # if we havent't visited that state before
            if graph_structure == "graph":
                if copyOfState not in visited:
                    stack.append(VacuumNode(copyOfState, vacuumNode.getPathCost() + 1, 
                                            vacuumNode.getPath() + [direction]))
                    visited.add(copyOfState)
            # If we are doing tree search, add the node regardless
            else: 
                stack.append(VacuumNode(copyOfState, vacuumNode.getPathCost() + 1, 
                                        vacuumNode.getPath() + [direction]))

        # Keep track how large the frontier becomes, and get the next node in the frontier
        # Get the node at the back of the list since DFS uses LIFO data structure
        max_frontier_size = max(max_frontier_size, len(stack))
        vacuumNode = stack.pop()

    return {'Cost': vacuumNode.getPathCost(), 'Nodes Generated': nodesExpanded, 
            'Space Complexity': max_frontier_size, 'path': vacuumNode.getPath(),
            'Total Cost': nodesExpanded + (weight * vacuumNode.getPathCost())}


def Iterative_Depth_First_Search(vacuumState, graph_structure, randomized=False, weight=1):
    """ Implementation of IDFS with Graph and Tree Search """

    nodes_expanded = 0
    max_frontier_size = 0
    for max_depth in range(maxsize):

        # initial node is the starting space of the vacuum, which we have
        # already visited
        vacuumNode = VacuumNode(vacuumState, 0, [])
        visited = {vacuumState} # Set to track visited States
        stack = [] # Stack for filo data structure
        depth = 0
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        while not vacuumNode.isGoalState():

            # get the neighbors and shuffle their order so we don't always put the nodes
            # in the same order
            nodes_expanded += 1
            currentState = vacuumNode.getState()
            copyOfState = deepcopy(currentState)
            if randomized: shuffle(directions)

            #move agent in the four different directions
            for direc in directions:
                direction = DIRECTIONS[direc]
                copyOfState = deepcopy(currentState)
                copyOfState.move_agent(DIRECTIONS[direc])
                if graph_structure == 'graph' and depth < max_depth:

                    # only expand neighbor's whose state we have no seen yet. If the state has
                    # not been seen, add the node to the frontier
                    if copyOfState not in visited:
                        depth += 1
                        stack.append(VacuumNode(copyOfState, vacuumNode.getPathCost() + 1, vacuumNode.getPath() + [direction]))
                        visited.add(copyOfState)

                if graph_structure == 'tree' and depth < max_depth:
                    depth += 1
                    stack.append(VacuumNode(copyOfState, vacuumNode.getPathCost() + 1, vacuumNode.getPath() + [direction]))
                
            # Keep track how large the frontier becomes, and get the next node in the frontier
            max_frontier_size = max(max_frontier_size, len(stack))

            #pops the next neighbor if any. If stack is empty increase the max depth
            if stack:
                vacuumNode = stack.pop()
            else:
                break
        
        #break out of max depth for loop if our current node is the goal state
        if vacuumNode.isGoalState():
            break

    return {'Cost': vacuumNode.getPathCost(), 'Nodes Generated': nodes_expanded, 
            'Space Complexity': max_frontier_size, 'path': vacuumNode.getPath(),
            'Total Cost': nodes_expanded + (weight * vacuumNode.getPathCost())}


def getDirtyCells(grid):
    """ Returns a list containing the locations of the dirty tiles """

    dirtyCells = []
    for row in range(len(grid)):
        for col in range(len(grid[row])):
            if grid[row][col] == "Dirt":
                dirtyCells.append((row, col))
    
    return dirtyCells
    

def heuristicOne(node):
    """ Calculate Manhattan Distance to closest dirt """

    state = node.getState()
    dirtyCells = np.asarray(getDirtyCells(state.grid))
    if len(dirtyCells) == 0: return 0
    currentLoc = state.agent_loc
    
    # CITE: https://codereview.stackexchange.com/questions/28207/finding-the-closest-point-to-a-list-of-points
    # DESC: Calculate closest point from a list of points using numpy
    distances = np.sum((dirtyCells - currentLoc)**2, axis=1)
    closest = np.argmin(distances)

    # returns Manhattan distance regardless of obstacles to closest dirt
    return abs(dirtyCells[closest][0] - currentLoc[0]) + abs(dirtyCells[closest][1] - currentLoc[1])


def heuristicTwo(node):
    """ Calculate sum of distances to all the dirts """

    state = node.getState()
    dirtyCells = getDirtyCells(state.grid)
    if len(dirtyCells) == 0: return 0
    currentLoc = state.agent_loc
    totalSum = 0

    # Calculates the sum of manhattan distances to all the dirts
    for i in range(len(dirtyCells)):
        totalSum += abs(dirtyCells[i][0] - currentLoc[0]) + abs(dirtyCells[i][1] - currentLoc[1])

    return totalSum


def heuristicThree(node):
    """ Calculate hypotenuse Distance to closest dirt """

    state = node.getState()
    dirtyCells = np.asarray(getDirtyCells(state.grid))
    if len(dirtyCells) == 0: return 0
    currentLoc = state.agent_loc
    
    # CITE: https://codereview.stackexchange.com/questions/28207/finding-the-closest-point-to-a-list-of-points
    # DESC: Calculate closest point from a list of points using numpy
    distances = np.sum((dirtyCells - currentLoc)**2, axis=1)
    closest = np.argmin(distances)

    # calculate hypotenuse distance
    hypotenuseSquared = abs(dirtyCells[closest][0] - currentLoc[0])**2 + abs(dirtyCells[closest][1] - currentLoc[1])**2
    return np.sqrt(hypotenuseSquared)


def bestFirstSearch(vacuumState, graph_structure, heuristic, randomized=False, weight=1):
    """ Implementation of Greedy Best First Search with Graph and Tree Search """

    vacuumNode = VacuumNode(vacuumState, 0, [])
    visited = {vacuumState} # Set to track visited States
    heap = [] # queue for filo data structure
    nodesExpanded = 0
    max_frontier_size = 0
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    while not vacuumNode.isGoalState():
        nodesExpanded += 1
        currentState = vacuumNode.getState()
        # Shuffle nodes if the directions are not control variables
        if randomized: shuffle(directions)
        for direc in directions:
            # Expand the neighbors by creating a copy of the state, and going in 
            # one of the four directions
            copyOfState = deepcopy(currentState)
            direction = DIRECTIONS[direc]
            copyOfState.move_agent(direction)

            # Create a node and calculate the heuristic value for that node
            # Best-First-Search only takes h(n) into account
            node = VacuumNode(copyOfState, vacuumNode.getPathCost() + 1, 
                                      vacuumNode.getPath() + [direction])
            cost = heuristic(node)
            node.setCurrentCost(cost)

            # If we are doing graph search, then only add the node to the frontier 
            # if we havent't visited that state before
            if graph_structure == 'graph':
                if copyOfState not in visited: 
                    heapq.heappush(heap, node)
                    visited.add(copyOfState)
            # If we are doing tree search, add the node regardless
            else: 
                heapq.heappush(heap, node)
        
        # Keep track how large the frontier becomes, and get the next node in the frontier
        # Use the heappop to get the minimum heuristic cost value
        max_frontier_size = max(max_frontier_size, len(heap))
        vacuumNode = heapq.heappop(heap)

    return {'Cost': vacuumNode.getPathCost(), 'Nodes Generated': nodesExpanded, 
            'Space Complexity': max_frontier_size, 'path': vacuumNode.getPath(),
            'Total Cost': nodesExpanded + (weight * vacuumNode.getPathCost())}


def AStarSearch(vacuumState, graph_structure, heuristic, randomized=False, weight=1):
    """ Implementation of A* Search with Graph and Tree Search """

    vacuumNode = VacuumNode(vacuumState, 0, [])
    visited = {vacuumState} # Set to track visited States
    heap = [] # queue for filo data structure
    nodesExpanded = 0
    max_frontier_size = 0
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    while not vacuumNode.isGoalState():
        nodesExpanded += 1
        currentState = vacuumNode.getState()
        # Shuffle nodes if the directions are not control variables
        if randomized: shuffle(directions)
        for direc in directions:
            # Expand the neighbors by creating a copy of the state, and going in 
            # one of the four directions
            copyOfState = deepcopy(currentState)
            direction = DIRECTIONS[direc]
            copyOfState.move_agent(direction)

            # Create a node and calculate the heuristic value for that node
            # A* only takes h(n) and g(n) into account
            node = VacuumNode(copyOfState, vacuumNode.getPathCost() + 1, 
                                      vacuumNode.getPath() + [direction])
            cost = node.getPathCost() + heuristic(node)
            node.setCurrentCost(cost)

            # If we are doing graph search, then only add the node to the frontier 
            # if we havent't visited that state before
            if graph_structure == 'graph':
                if copyOfState not in visited: 
                    heapq.heappush(heap, node)
                    visited.add(copyOfState)
            # If we are doing tree search, add the node regardless
            else: 
                heapq.heappush(heap, node)

        # Keep track how large the frontier becomes, and get the next node in the frontier
        # Use the heappop to get the minimum heuristic cost value
        max_frontier_size = max(max_frontier_size, len(heap))
        vacuumNode = heapq.heappop(heap)

    return {'Cost': vacuumNode.getPathCost(), 'Nodes Generated': nodesExpanded, 
            'Space Complexity': max_frontier_size, 'path': vacuumNode.getPath(),
            'Total Cost': nodesExpanded + (weight * vacuumNode.getPathCost())}


def main():
    """ Main Driver of the Program """

    if len(argv) != 4:
        print('Usage: python3 vacuum.py environment search_technique graph_structure')
        exit(1)

    if int(argv[1]) not in range(0, 12):
        print('Not a valid environment. Choose from 0 to 11')
        exit(1)
    
    if argv[2] not in ('BFS', 'DFS', 'IDS', 'GBFS', 'A*', 'all'):
        print('Not a valid search techninque. Please choose between \'all\', \'BFS\', \'DFS\', \'IDS\', \'GBFS\', and \'A\\*\'')
        exit(1)
    
    if argv[3] not in ('graph', 'tree'):
        print('Not a valid structure. Please choose betweem \'graph\' and \'tree\'.')
        exit(1)

    if argv[2] == 'BFS':
        print(BFS(make_vacuum_environment(int(argv[1])), argv[3]))
    elif argv[2] == 'DFS':
        print(DFS(make_vacuum_environment(int(argv[1])), argv[3]))
    elif argv[2] == 'IDS':
        print(Iterative_Depth_First_Search(make_vacuum_environment(int(argv[1])), argv[3]))
    elif argv[2] == 'GBFS':
        print(bestFirstSearch(make_vacuum_environment(int(argv[1])), argv[3], heuristicTwo))
    elif argv[2] == 'A*':
        print(AStarSearch(make_vacuum_environment(int(argv[1])), argv[3], heuristicTwo))
    elif argv[2] == 'all':
        print('BFS:', BFS(make_vacuum_environment(int(argv[1])), argv[3]))
        print('DFS:', DFS(make_vacuum_environment(int(argv[1])), argv[3]))
        print('IDFS:', Iterative_Depth_First_Search(make_vacuum_environment(int(argv[1])), argv[3]))
        print('GBFS:', bestFirstSearch(make_vacuum_environment(int(argv[1])), argv[3], heuristicTwo))
        print('A*:', AStarSearch(make_vacuum_environment(int(argv[1])), argv[3], heuristicTwo))
    else:
        print('Error')


if __name__ == "__main__":
    main()