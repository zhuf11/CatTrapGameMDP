import hexutil
import random
import copy
import time
import numpy as np
from ps_reward_4 import CatEvalFn

def hex_to_ij(hex):
    return hex.y, hex.x//2

def ij_to_hex(i,j):
    return hexutil.Hex(2*j+(i%2),i)

class InvalidMove(ValueError):
    pass

class Game(object):
    """Represents a game state.
    """
    def __init__(self, size):
        self.cat_i = size//2
        self.cat_j = size//2
        self.size  = size
        self.tiles = np.array([[0 for col in range(size)] for row in range(size)])
        self.deadline = 0
        self.terminated = False
        self.start_time = time.time()
        self.eval_fn = CatEvalFn()
        self.reached_maxdepth = False 

    def init_random_blocks(self,cat):
        n = random.randint(round(0.067*(self.size**2)),round(0.13*(self.size**2)))
        count = 0
        the_blocks=[]
        self.cat_i,self.cat_j = hex_to_ij(cat)
        self.tiles[self.cat_i][self.cat_j] = 6
        
        while count < n:
            i = random.randint(0,self.size-1)
            j = random.randint(0,self.size-1)
            if self.tiles[i][j]==0:
                self.tiles[i][j] = 1
                count = count + 1    

    def init_blocks(self,the_blocks,cat):
        i,j=hex_to_ij(cat)
        self.tiles[i][j] = 6
        
        for block in the_blocks:
            if block != [i,j]:
                self.game.tiles[block[0]][block[1]] = 1


    def CustomCat(self, randcat, MDP, max_depth, num_states):
        self.start_time = time.time()
        if randcat:
            result = self.RandomCat()
        elif MDP:
            result = self.MDPCat(max_depth, num_states)
        else:
            raise ValueError('No type selected.')
            
        elapsed_time = (time.time() - self.start_time) * 1000
        print ("Elapsed time: %.3fms " % elapsed_time)
        return result


    # Random Cat
    # Just a plain old Random Cat

    def RandomCat(self):
        moves=self.valid_moves() #["W","E","SE","SW","NE","NW"]
        print(moves)
        dir="NONE"

        if len(moves)>0:    
            dir = random.choice(moves)
        else: 
            return [self.cat_i,self.cat_j]

        return self.target(self.cat_i,self.cat_j,dir)


    # MDP Cat

    def MDPCat(self, max_depth, num_states):
        move = self.mdp(max_depth, num_states)
        return move 

#====================================================================================================        

    def valid_moves(self):
        tiles,cat_i,cat_j=self.tiles,self.cat_i,self.cat_j
        size = self.size
        moves=[]
        if (cat_j<size-1 and tiles[cat_i][cat_j+1]==0):
            moves.append("E")
        if (cat_j>0           and tiles[cat_i][cat_j-1]==0):
            moves.append("W")

        if (cat_i%2)==0:
            if (cat_i>0  and cat_j<size   and tiles[cat_i-1][cat_j]==0):
                moves.append("NE")
        else:
            if (cat_i>0  and cat_j<size-1 and tiles[cat_i-1][cat_j+1]==0):
                moves.append("NE")

        if (cat_i%2)==0:
            if (cat_i>0  and cat_j>0  and tiles[cat_i-1][cat_j-1]==0):
                moves.append("NW")
        else:
            if (cat_i>0  and cat_j>=0 and tiles[cat_i-1][cat_j]==0):
                moves.append("NW")

        if (cat_i%2)==0:
            if (cat_i<size-1  and cat_j<size   and tiles[cat_i+1][cat_j]==0):
                moves.append("SE")
        else:
            if (cat_i<size-1  and cat_j<size-1 and tiles[cat_i+1][cat_j+1]==0):
                moves.append("SE")

        if (cat_i%2)==0:
            if (cat_i<size-1  and cat_j>0  and tiles[cat_i+1][cat_j-1]==0):
                moves.append("SW")
        else:
            if (cat_i<size-1  and cat_j>0   and tiles[cat_i+1][cat_j]==0):
                moves.append("SW")
        return moves


    def target(self,i,j,dir):
        out=[i,j] 
        if dir == "E":
            out=[i,j+1]
        elif dir =="W":
            out=[i,j-1]
        elif dir == "NE":
            out = [i-1,j] if (i%2)==0 else [i-1,j+1]
        elif dir == "NW":
            out=[i-1,j-1] if (i%2)==0 else [i-1,j]
        elif dir == "SE":
            out=[i+1,j]   if (i%2)==0 else [i+1,j+1]            
        elif dir == "SW":    
            out=[i+1,j-1] if (i%2)==0 else [i+1,j]
        return out

    def utility(self):
        """
        Takes game state and computes pseudo-reward of that state.
        """
        # Absorbing states
        moves = self.valid_moves()
        if self.cat_i == 0 or self.cat_i == self.size or self.cat_j == 0 or self.cat_j == self.size:
            return float(500000)

        if len(moves) == 0:
            return float(-2500)

        return self.eval_fn.score_challenge(self)

    def apply_move(self,move,maximizing_player):
        if self.tiles[move[0]][move[1]] != 0:
            print(self.tiles)
            print(move)
            raise InvalidMove("Invalid Move!")

        if maximizing_player:
            self.tiles[move[0]][move[1]] = 1
        else:
            self.tiles[move[0]][move[1]] = 6        # place cat
            self.tiles[self.cat_i][self.cat_j] = 0  # remove old cat
            self.cat_i = move[0]
            self.cat_j = move[1]


    def time_left(self):
        return  (self.deadline - time.time()) * 1000

    def print_tiles(self):
        i=0
        while i < self.size: 
            print(self.tiles[i])
            if i+1 < self.size:
                print("",self.tiles[i+1])
            i=i+2
        return


    def tile_squasher(self):
        """
        Takes tiles arrays from Game object and flattens the tiles out and converts it into string.
        """
        squashed = []
        for tile in self.tiles:
            squashed.extend(tile)
        return str(squashed)

    def get_states(self, states, depth=3, curr_depth=0,num_states=5):
        """
        Recursively adds to given list of Game objects of all the possible states for the given starting state of the
        game. More precisely, it computes all the moves the cat can make and also takes into account all moves the
        player can make (blocking tiles).
        """
        states.append(self)

        if depth == 0:
            for i in range(self.size):
                for j in range(self.size):
                    if self.tiles[i][j] == 0:
                        other_game = copy.deepcopy(self)
                        other_game.apply_move([i, j], maximizing_player=False)
                        states.append(other_game)
            return

        if curr_depth >= depth:
            return

        actions = self.valid_moves()
        for action in actions:
            move = self.target(self.cat_i, self.cat_j, action)
            other_game = copy.deepcopy(self)
            other_game.apply_move(move, maximizing_player=False)
            all_states = []
            rewards = np.array([])
            for i in range(self.size):
                for j in range(self.size):
                    if other_game.tiles[i][j] == 0:
                        move_game = copy.deepcopy(other_game)
                        move_game.apply_move([i, j], maximizing_player=True)
                        rewards = np.append(rewards, move_game.utility())
                        all_states.append(move_game)
            if len(all_states) > num_states and num_states != -1:
                for idx in rewards.argsort()[:num_states]:
                    all_states[idx].get_states(states, depth, curr_depth+1, num_states)
            else:
                for state in all_states:
                    state.get_states(states, depth, curr_depth+1, num_states)

        return

    def get_action_states(self, move, num_states):
        """
        Returns a list of Game objects of all possible states from a given move. More precisely it computes the cat move
        and all the possible moves the player can do (blocking tiles).
        """
        states = []
        all_states = []
        rewards = np.array([])
        other_game = copy.deepcopy(self)
        other_game.apply_move(move, maximizing_player=False)
        for i in range(self.size):
            for j in range(self.size):
                if other_game.tiles[i][j] == 0:
                    move_game = copy.deepcopy(other_game)
                    move_game.apply_move([i, j], maximizing_player=True)
                    rewards = np.append(rewards, move_game.utility())
                    all_states.append(move_game)

        if len(all_states) > num_states and num_states != -1:
            for idx in rewards.argsort()[:num_states]:
                states.append(all_states[idx])
        else:
            for state in all_states:
                states.append(state)
        return states

    def mdp(self, max_depth, num_states):
        """
        Carries out value iteration algorithm.
        """
        max_change = float("inf")
        count = 0
        if max_depth == 0:
            epsilon = 0.1  # Epsilon for convergence
        else:
            epsilon = 0.1  # Epsilon for convergence
        gamma = 0.85  # Discount factor
        if gamma != 0:
            checker = epsilon * (1 - gamma) / gamma
        else:
            checker = epsilon

        U_ = {}  # Dict for storing all our U values
        pi = {}  # Dict for storing best actions
        states = []
        self.get_states(states, depth=max_depth, num_states=num_states)  # Getting all possible states (list of Game objects)
        legal_moves = self.valid_moves()
        if len(legal_moves) == 0:
            return self.cat_i, self.cat_j
        for s in states:
            U_[s.tile_squasher()] = 0  # Initializing all U values as zero

        # Value iteration algorithm
        if max_depth == 0:
            while max_change > checker and count < 1000:
                U = U_.copy()
                max_change = 0
                for s in states:
                    action = {}
                    actions = s.valid_moves()  # ["W","E","SE","SW","NE","NW"]
                    for a in actions:
                        move = s.target(s.cat_i, s.cat_j, a)
                        other_game = copy.deepcopy(s)
                        other_game.apply_move(move, maximizing_player=False)
                        x = U_[other_game.tile_squasher()]
                        action[a] = other_game.utility() + (gamma * x)

                    if len(actions) != 0:
                        pi[s.tile_squasher()] = max(action, key=action.get)
                        U_[s.tile_squasher()] = action[pi[s.tile_squasher()]]
                    else:
                        pi[s.tile_squasher()] = None
                        U_[s.tile_squasher()] = -100
                    if abs(U[s.tile_squasher()] - U_[s.tile_squasher()]) > max_change:
                        max_change = abs(U[s.tile_squasher()] - U_[s.tile_squasher()])

                count += 1
        else:
            while max_change > checker and count < 1000:
                U = U_.copy()
                max_change = 0
                for s in states:
                    action = {}
                    actions = s.valid_moves()  # ["W","E","SE","SW","NE","NW"]
                    for a in actions:
                        move = s.target(s.cat_i, s.cat_j, a)
                        next_game = s.get_action_states(move, num_states)
                        all_rewards = []
                        for game in next_game:
                            if game.tile_squasher() in U_:
                                all_rewards.append(game.utility() + gamma * U_[game.tile_squasher()])
                            else:
                                all_rewards.append(game.utility())
                        all_rewards = np.array(all_rewards)

                        # We are assuming that there is equal probability that player blocks tile for all tiles
                        probability_function = 1/len(next_game)
                        action[a] = (probability_function * all_rewards).sum()

                    if len(actions) != 0:
                        pi[s.tile_squasher()] = max(action, key=action.get)
                        U_[s.tile_squasher()] = action[pi[s.tile_squasher()]]
                    else:
                        pi[s.tile_squasher()] = None
                        U_[s.tile_squasher()] = -1000
                    if abs(U[s.tile_squasher()] - U_[s.tile_squasher()]) > max_change:
                        max_change = abs(U[s.tile_squasher()] - U_[s.tile_squasher()])

                count += 1
                if count == 1000:
                    print('Max iter reached.')

        action = pi[self.tile_squasher()]
        best_move = self.target(self.cat_i,self.cat_j, action)
        best_next = copy.deepcopy(self)
        best_next.apply_move(best_move, maximizing_player=False)
        return best_move



