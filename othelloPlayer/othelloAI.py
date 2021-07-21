"""
Lucas Barusek

This program implements varying board game AI algorithms to play a game of Othello (Reversi)
The algorithms that were implemented were the following:
    -Minimax
    -Minimax with Alpha Beta Pruning
    -Quiscient Minimax Search with Alpha Beta Pruning
    -Monte Carlo Tree Search
Additionally, we implemented Three heuristics for minimax algorithms
    -HeurisitcOne - Gives the utility as a function of how many pieces are on the board
    -HeuristicTwo - Gives the utility as a function of the stability of the board
    -HeuristicTHree - Gives the utility as a function of pecies on the baord, stability of the board, corners captured, and phase in the game
For the minimax algorithms, the default heursitic is heuristcthree. If you would like to change the default paramter, just enter in the value of '1','2', or '3'
The Game can played beween any AI or Human Player. Run th program and follow the prompts in order to setup the game
The tournament player that we will submit is Minimax with Alpha-Beta Pruning utilziing Heuristic Three
"""

from othelloSupport import *
import random, sys
import math
from time import time, sleep
from copy import deepcopy


MONTE_CARLO_TIMES_PER_MOVE = [2, 5, 6, 7]
# board weights in order to determine stability of a tile. 
# a higher value is a higher ranked overall stability
boardWeights = [[10,-2,6,4,4,6,-2,10],           
                [-2,-4,-2,-.5,-.5,-2,-4,-2],
                [6,-2,1,1,1,1,-2,6],
                [4,-.5,1,-1,-1,1,-.5,4],
                [4,-.5,1,-1,-1,1,-.5,4],
                [6,-2,1,1,1,1,-2,6],
                [-2,-4,-2,-.5,-.5,-2,4,-2],
                [10,-2,6,4,4,6,-2,10]]

#multiplicative factors taken into how optimal a move is
numPeicesWeightB = 100
numPeicesWeightE = 600
cornerWeight = 400
mobilityWeight = 150
stabiltiyWeight = 300


class MoveNotAvailableError(Exception):
    """Raised when a move isn't available."""
    pass


class OthelloTimeOut(Exception):
    """Raised when a player times out."""
    pass


class OthelloPlayer():
    """Parent class for Othello players."""

    def __init__(self, color):
        assert color in ["black", "white"]
        self.color = color

    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make. Each type of player
        should implement this method. remaining_time is how much time this player
        has to finish the game."""
        pass


class RandomPlayer(OthelloPlayer):
    """Plays a random move."""

    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make."""
        return random.choice(state.available_moves())


class HumanPlayer(OthelloPlayer):
    """Allows a human to play the game"""

    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make."""
        available = state.available_moves()
        print("----- {}'s turn -----".format(state.current))
        print("Remaining time: {:0.2f}".format(remaining_time))
        print("Available moves are: ", available)
        move_string = input("Enter your move as 'r c': ")

        # Takes care of errant inputs and bad moves
        try:
            moveR, moveC = move_string.split(" ")
            move = OthelloMove(int(moveR), int(moveC), state.current)
            if move in available:
                return move
            else:
                raise MoveNotAvailableError # Indicates move isn't available

        except (ValueError, MoveNotAvailableError):
            print("({}) is not a legal move for {}. Try again\n".format(move_string, state.current))
            return self.make_move(state, remaining_time)


def heuristicOne(node, maxColor):
    """determins how many of each of the opponents pawns on the board and returns the difference"""
    return node.count(maxColor) - node.count(opposite_color(maxColor))


def heuristicTwo(node, maxColor):
    """determins the total stability of the board"""
    weight = heuristicOne(node, maxColor)

    for row in range(8):
        for col in range(8):
            if node.board[row][col] == maxColor:
                weight += boardWeights[row][col]

    return weight


#Cite: Online Forum that proposed having weights as a ratio between players provides a more reasonable heursitic value than weights for the current player
def heuristicThree(node, maxColor):
    """"calculates the weight of the board. References mlitplative factors 
    so corners and stability are rated higher. The weights are expresssed as 
    a ration to ensure time is considered in weight.
    Stability: The chance that the given token will be able to be flipped
    Corners: the most stable peices and always favored.
    PeiceCount: The difference in number of the players peices and the opponents
    mobility:favored higher when the player has a wider number of moves to choose from"""
    weight = 0

    #calculate number of peices for each player
    myPeices = node.count(maxColor)
    opponentPeices = node.count(opposite_color(maxColor))

    #in the last few moves, only prioritize moves that give us the largest return in number of chips
    if(node.move_number > 57):
        return numPeicesWeightE * 100 * ((myPeices - opponentPeices)/(myPeices+opponentPeices))
    
    #number of peices
    turnWeight = (numPeicesWeightB + ((numPeicesWeightE - numPeicesWeightB) / 60) * node.move_number)
    weight += turnWeight * 100 * (myPeices - opponentPeices) / (myPeices + opponentPeices)

    #finding number of available moves
    origColor = node.current
    node.current = opposite_color(maxColor)
    oppMoves = len(node.available_moves())
    node.current = maxColor
    myMoves = len(node.available_moves())
    node.current = origColor
    
    #mobility is higher and favored. A higher mobility results in lesss possible moves the opponent can take
    if(oppMoves + myMoves) != 0:
        weight += mobilityWeight * 100 * (myMoves - oppMoves)/(myMoves + oppMoves)
    
    #determiens who currently has possesion of the corners.
    myCorners = 0
    oppCorners = 0
    for row in [0,7]:
        for col in [0,7]:
            if node.board[row][col] == opposite_color(maxColor):
                oppCorners += 1
            elif node.board[row][col] == maxColor:
                myCorners += 1
            else:
                continue
    if(myCorners + oppCorners) != 0:
        weight += cornerWeight * 100 * (myCorners - oppCorners)/(myCorners + oppCorners)

    #a higher stability results in tiles less likely to be flipped thus these spots are favored.
    myStability, oppStability = 0,0
    for row in range(8):
        for col in range(8):
            if node.board[row][col] == maxColor:
                myStability += boardWeights[row][col]
            elif node.board[row][col] == opposite_color(maxColor):
                oppStability += boardWeights[row][col]
            else:
                continue
    
    
    if myStability + oppStability == 0:
        return weight

    weight += stabiltiyWeight * 100 * (myStability - oppStability)/(myStability + oppStability)
    return weight


def terminalUtility(node, maxColor):
    """if the terminal node results in a win return this node immedientaly 
    with the highest possible rated weight"""
    winner = node.winner()
    if winner == "draw":
        return 0
    elif winner == maxColor:
        return sys.maxsize
    else: # draw
        return - sys.maxsize


def get_tiles_to_flip(state, moveLoc):
        """ Returns the list of tiles that will be flipped as a result of a move """
        
        # array to keep track of locations to convert to the players color
        locsToConvert = [moveLoc]

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                
                (r, c) = moveLoc
                # If we are flanking, then we need to convert 
                # the locations in between the tiles
                if state.flanking(r + dr, c + dc, dr, dc,
                                    opposite_color(state.current),
                                    state.current):

                    # Iterate through all locs between the flanking
                    # tile and the move location, and compile 
                    # these locations to the list
                    currRow = r + dr; currCol = c + dc
                    while state.board[currRow][currCol] != state.current:
                        locsToConvert.append((currRow, currCol))
                        currRow += dr
                        currCol += dc

        return locsToConvert


class MinimaxPlayer(OthelloPlayer):
    """Allows a minimax player to play the game"""
    def __init__(self, color, heuristicFunc=heuristicThree, pruning=False, depth=4):
        assert color in ["black", "white"]
        self.color = color
        self.heuristicFunc = heuristicFunc
        self.pruning = pruning
        self.maxDepth = depth
    
    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make."""
        if self.pruning:
            return self.alphaBetaPruning(state, -sys.maxsize, sys.maxsize, 0, True)[1]
        else:
            return self.minimax(state, 0, True)[1]

    def minimax(self, node, depth, MAX_TURN):
        """An AI player that utilizes minimax algorithm without pruning. Exercies the given heuristic that user chooses"""
        available = node.available_moves()
        if len(available) == 0:
            return (terminalUtility(node, self.color), None)
        if depth >= self.maxDepth:
            return (self.heuristicFunc(node, self.color), None)
            
        #choose the first move as the highest possible move and change it according to if a new move is found
        bestMove = available[0]
        if MAX_TURN:
            value = - sys.maxsize
            for move in available:
                child = node.apply_move(move)
                moveValue, _ = self.minimax(child, depth+1, not MAX_TURN)
                if moveValue > value:
                    value = moveValue
                    bestMove = move
            
        else: # MIN's turn
            value = sys.maxsize
            for move in available:
                child = node.apply_move(move)
                moveValue, _ = self.minimax(child, depth+1, not MAX_TURN)
                if moveValue < value:
                    value = moveValue
                    bestMove = move 

        return (value, bestMove)


    def alphaBetaPruning(self, node, alpha, beta, depth, MAX_TURN):
        """Minimax Algorithm that utilized alpha-beta pruning to reduce the number of nodes search."""
        available = node.available_moves()
        if len(available) == 0:
            return (terminalUtility(node, self.color), None)
        if depth >= self.maxDepth:
            return (self.heuristicFunc(node, self.color), None)
    
        bestMove = available[0]
        if MAX_TURN:
            value = -sys.maxsize
            for move in available:
                child = node.apply_move(move)
                recurValue, _ = self.alphaBetaPruning(child, alpha, beta, depth+1, not MAX_TURN)
                if recurValue >= value:
                    bestMove = move
                    value = recurValue
                    alpha = max(value, alpha)
                    if alpha >= beta:
                        return (value, bestMove)
        else:
            value = sys.maxsize
            for move in available:
                child = node.apply_move(move)
                recurValue, _ = self.alphaBetaPruning(child, alpha, beta, depth+1, not MAX_TURN)
                if recurValue <= value:
                    bestMove = move
                    value = recurValue
                    beta = min(value, beta)
                    if alpha >= beta:
                        return (value, bestMove)

        return (value, bestMove)


class QuiescentPlayer(OthelloPlayer):
    """Allows a minimax player to play the game"""
    def __init__(self, color, heuristicFunc=heuristicThree, quiescentValue=3, depth=4):
        assert color in ["black", "white"]
        self.color = color
        self.heuristicFunc = heuristicFunc
        self.quiescentValue = quiescentValue
        self.maxDepth = depth
    
    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make."""
        return self.quiescentPruning(state, -sys.maxsize, sys.maxsize, 0, True)[1]
    

    def quiescentPruning(self, node, alpha, beta, depth, MAX_TURN):
        """ Implements alpha beta pruning with a quiscent check as well """
        available = node.available_moves()
        if len(available) == 0:
            return (terminalUtility(node, self.color), None)

        # If we've reached max depth or we are in a quiscent position, return the
        # heuristic value
        if (depth >= self.maxDepth) or self.quiescentPosition(node, available):
            return (self.heuristicFunc(node, self.color), None)

        bestMove = available[0]
        if MAX_TURN:
            value = -sys.maxsize
            for move in available:
                child = node.apply_move(move)
                recurValue, _ = self.quiescentPruning(child, alpha, beta, depth+1, not MAX_TURN)
                if recurValue >= value:
                    bestMove = move
                    value = recurValue
                    alpha = max(value, alpha)
                    if alpha >= beta:
                        return (value, bestMove)
        else:
            value = sys.maxsize
            for move in available:
                child = node.apply_move(move)
                recurValue, _ = self.quiescentPruning(child, alpha, beta, depth+1, not MAX_TURN)
                if recurValue <= value:
                    bestMove = move
                    value = recurValue
                    beta = min(value, beta)
                    if alpha >= beta:
                        return (value, bestMove)

        return (value, bestMove)

    def quiescentPosition(self, node, available):
        """ return true iff we are in a quiescent position and can return a safe state"""
        numTilesFlipped = []
        # only check for quiescent if it's the opponent's turn
        if opposite_color(node.current) == self.color:
            for move in available:
                numTilesFlipped.append(len(get_tiles_to_flip(node, move.pair)))

            # we are in a quiscent position if the maximum number of tiles that
            # can be flipped is less than the quiescent value
            if max(numTilesFlipped) <= self.quiescentValue:
                return True

        return False


class ShortTermMinimizer(OthelloPlayer):
    """ AI that makes the move that results in the least tiles
    of its color being on the board after the move """

    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make."""
        available = state.available_moves()
        numTilesFlipped = []
        for move in available:
            numTilesFlipped.append(len(get_tiles_to_flip(state, move.pair)))
        move = available[numTilesFlipped.index(min(numTilesFlipped))]
        return OthelloMove(move.pair[0], move.pair[1], state.current)


class MonteCarloNode():
    def __init__(self, state, parent):
        self.state = state
        self.parent = parent

    def __hash__(self):
        return hash(tuple([self.state.current, self.state.move_number]+ \
                    [tile for row in self.state.board for tile in row]))

    def __eq__(self, other):
        return self.state.current == other.state.current and\
            self.state.move_number == other.state.move_number and\
            [tile for row in self.state.board for tile in row] == \
            [tile for row in other.state.board for tile in row]

    def __ne__(self, other):
        return not self.__eq__(other)


class MonteCarloPlayer(OthelloPlayer):
    """ AI that implements MCTS """
    def __init__(self, color):
        assert color in ["black", "white"]
        self.color = color
        # Hash maps to implement the 'tree'
        self.wins = {}
        self.gamesPlayed = {}
    
    def make_move(self, state, remaining_time):
        """ Returns the best move based on the MTCS simulation  """
        # calculate time that the simulations can run, giving more weight to 
        # the mid and end game
        timeEnd = time() + MONTE_CARLO_TIMES_PER_MOVE[state.move_number // 20]
        root = MonteCarloNode(state, None) 

        # reset the root every time since tree is a hash map and children are
        # not dependent on their parents
        self.gamesPlayed[root] = 1
        self.wins[root] = 0
        self.numSimulations = 1
        while time() < timeEnd:
            # descend down the current tree and find the new node to add
            newNode, isTerminal = self.descend(root)

            # if the node is not terminal, then simulate to the end of the game
            if not isTerminal:
                self.wins[newNode] = 0; self.gamesPlayed[newNode] = 0
                winner = self.simulate(newNode)
            else: # if it is terminal, find out who won
                winner = newNode.state.winner()

            # get the delta based on the outcome of the game, and update the 
            # tree with the outcome 
            delta = self.getDelta(root, winner)
            self.updateTree(newNode, root, delta)

        # return the best node
        return self.bestMove(root)

    def descend(self, root):
        """ Starting at the root, descend down the tree until you get to a new node """
        knownNode = True
        currNode = deepcopy(root)
        while knownNode:
            currNode, knownNode = self.pickChild(currNode)
            if currNode.state.game_over():
                return (currNode, True)
        
        return (currNode, False)
    
    def pickChild(self, root):
        """ Expand a given node, and return either a child that has not been 
        visited, or the child with the highest UCB score"""
        available = root.state.available_moves()
        scores = []
        nodes = []
        for move in available:
            child = root.state.apply_move(move)
            mctsNode = MonteCarloNode(child, root)
            nodes.append(mctsNode)
            # If we haven't seen this node, return it
            if self.notVisited(mctsNode):
                return (mctsNode, False)
            else: # else find the UCB Score
                scores.append(self.ucb(mctsNode, root))

        return (nodes[scores.index(max(scores))], True)
    
    def ucb(self, node, parent):
        """ Calculates a UCB score of a given node, based on the formula
        we learned in class """
        gamesPlayed = self.gamesPlayed[node]
        if self.isMaxTurn(node):
            return (self.wins[node] / gamesPlayed) + \
                   math.sqrt((2 * math.log(self.gamesPlayed[parent]))) / gamesPlayed
        else:
            return ((gamesPlayed - self.wins[node]) / gamesPlayed) + \
                   math.sqrt((2 * math.log(self.gamesPlayed[parent]))) / gamesPlayed

    def isMaxTurn(self, node):
        """ returns true iff current player is MAX """
        return True if node.state.current == self.color else False

    def notVisited(self, node):
        """ return true iff we have not see a node """
        return node not in self.gamesPlayed

    def simulate(self, node):
        """ Simulate a game by picking random moves until it's over """
        currNode = deepcopy(node)
        while not currNode.state.game_over():
            move = random.choice(currNode.state.available_moves())
            child = currNode.state.apply_move(move)
            currNode = MonteCarloNode(child, node)

        # return the winner after the random game
        return currNode.state.winner()
        
    def updateTree(self, node, root, delta):
        """ Recursively update the tree and the score of each node """
        if not node.state.game_over():
            self.wins[node]+= delta
            self.gamesPlayed[node]+= 1
        if node != root: 
            self.updateTree(node.parent, root, delta)

    def bestMove(self, root):
        """ Returns move that results in the move with the best win percentage """
        available = root.state.available_moves()
        if len(available) == 1: return available[0] #edge case for one terminal node
        bestWeight = -1
        bestWeightGames = -1
        bestMove = available[0]
        for move in available:
            child = root.state.apply_move(move)
            mctsNode = MonteCarloNode(child, root)
            if mctsNode in self.gamesPlayed:
                weight = self.wins[mctsNode] / self.gamesPlayed[mctsNode]
                if (weight > bestWeight) or \
                    (weight == bestWeight and self.gamesPlayed[mctsNode] > bestWeightGames):
                    bestWeight = weight
                    bestWeightGames = self.gamesPlayed[mctsNode]
                    bestMove = move

        return bestMove

    def getDelta(self, root, winner):
        """ Returns the appropriate delta based on who won """
        if winner == "draw":
            return .5
        elif winner == self.color:
            return 1
        else:
            return 0


class TournamentPlayer(OthelloPlayer):
    """You should implement this class as your entry into the AI Othello tournament.
    You should implement other OthelloPlayers to try things out during your
    experimentation, but this is the only one that will be tested against your
    classmates' players.
    
    We decided to implement Alpha-Beta as our tournament player
    """
    def __init__(self, color, heuristicFunc=heuristicThree, depth=5):
        assert color in ["black", "white"]
        self.color = color
        self.heuristicFunc = heuristicFunc
        self.maxDepth = depth
    
    def make_move(self, state, remaining_time):
        """Given a game state, return a move to make."""
        return self.alphaBetaPruning(state, -sys.maxsize, sys.maxsize, 0, True)[1]

    def alphaBetaPruning(self, node, alpha, beta, depth, MAX_TURN):
        """Minimax Algorithm that utilized alpha-beta pruning to reduce the number of nodes search."""
        available = node.available_moves()
        if len(available) == 0:
            return (terminalUtility(node, self.color), None)
        if depth >= self.maxDepth:
            return (self.heuristicFunc(node, self.color), None)
    
        bestMove = available[0]
        if MAX_TURN:
            value = -sys.maxsize
            for move in available:
                child = node.apply_move(move)
                recurValue, _ = self.alphaBetaPruning(child, alpha, beta, depth+1, not MAX_TURN)
                if recurValue >= value:
                    bestMove = move
                    value = recurValue
                    alpha = max(value, alpha)
                    if alpha >= beta:
                        return (value, bestMove)
        else:
            value = sys.maxsize
            for move in available:
                child = node.apply_move(move)
                recurValue, _ = self.alphaBetaPruning(child, alpha, beta, depth+1, not MAX_TURN)
                if recurValue <= value:
                    bestMove = move
                    value = recurValue
                    beta = min(value, beta)
                    if alpha >= beta:
                        return (value, bestMove)

        return (value, bestMove)
    

################################################################################
    

def getTypeOfHeuristic(color):
    """Returns the type of heuristic the user wants"""
    func = input(f"\n1 - Number of Pawns focused Heuristic\
                    \n2 - Stability Focused Heuristic\
                    \n3 - Time Determining Heuristic\
                    \n\nWhat type of heuristic will {color.upper()} choose (1, 2, 3)? ").strip()
    if func == '1':
        return heuristicOne
    elif func == '2':
        return heuristicTwo
    elif func == '3':
        return heuristicThree
    else:
        print("ERROR: Heuristic must be 1, 2, 3")
        exit(-1)

def getTypeOfPlayer(color):
    """Returns the type of player color will be based on user input """
    player = input(f"\n0 - Human Player\
                    \n1 - Random Player\
                    \n2 - Minimax Player\
                    \n3 - Minimax w/ Alpha-Beta Pruning Player(Tournament Player)\
                    \n4 - Quiscient Search Player\
                    \n5 - MonteCarlo Tree Search Player\
                    \n\nWhat type of player will {color.upper()} be (0, 1, 2, 3, 4 or 5)? ").strip()
                
    if player == '0':
        return HumanPlayer(color)
    elif player == '1':
        return RandomPlayer(color)
    elif player == '2':
        heuristicFunc = getTypeOfHeuristic(color)
        return MinimaxPlayer(color, heuristicFunc)
    elif player == '3':
        heuristicFunc = getTypeOfHeuristic(color)
        return MinimaxPlayer(color, heuristicFunc, True)
    elif player == '4':
        heuristicFunc = getTypeOfHeuristic(color)
        return QuiescentPlayer(color, heuristicFunc)
    elif player == '5':
        return MonteCarloPlayer(color)
    else:
        print("ERROR: Player must be 0, 1, 2, 3, 4, or 5")
        exit(-1)


def main():
    """ Plays the game."""

    black_player = getTypeOfPlayer("black")
    white_player = getTypeOfPlayer("white")

    verbose = input("Do you want a verbose game (Y/n)? ").strip().lower()
    if verbose not in ['', 'y', 'n']:
        print("ERROR: Verbose must be 'y' or 'n'")
        exit(-1)
    if verbose == 'y' or verbose == '':
        verbose = True
    else:
        verbose = False

    timed = None
    if black_player.__class__.__name__ == "HumanPlayer" or \
       white_player.__class__.__name__ == "HumanPlayer":
        timed = False
    else:
        timed = input("Do you want a timed game (Y/n)? ").strip().lower()
        if timed not in ['', 'y', 'n']:
            print("ERROR: Timed must be 'y' or 'n'")
            exit(-1)
        if timed == 'y' or timed == '':
            timed = True
        else:
            timed = False

    print(f"Black: {black_player.__class__.__name__} vs. White: {white_player.__class__.__name__}!!!")
    sleep(1)
    game = OthelloGame(black_player, white_player, verbose=verbose)
    
    if timed:
        ###### Use this method if you want to play a timed game. Doesn't work with HumanPlayer
        winner = game.play_game_timed()
    else:
        ##### Use this method if you want to use a HumanPlayer
        winner = game.play_game()

    if not verbose:
        print("Winner is", winner)


if __name__ == "__main__":
    main()