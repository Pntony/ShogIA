# -*- coding: utf-8 -*-

import time
import random
import math
import queue


class MCTS_AI:
    # Warning:
    # When running the MCTS, 2 different trees are used: the game tree with all possible
    # moves only implicitly used, and the MCTS tree of the algorithm expanded through time.
    # When referring to the "tree", it is about the MCTS tree.
    #
    # The rewards at the nodes are adapted to the playing side at the root node v0, that is
    # a big reward at any node is good for black if it's black who plays at v0,
    # and good for white otherwise.
    
    def __init__(self, time_limit, C):
        # Time limit for one run of uct_search method.
        self.time_limit = time_limit
        # The C constant in the UCT formula.
        self.C = C
        
        ###
        self.time_block = 0
        ###
    
    # Run the UCT algorithm starting from s0 state until the time_limit is reached.
    # The best move from s0 is then returned in USI format.
    # time_limit must be in second.
    def best_move(self, s0):
        tic = time.perf_counter()
        
        if s0.is_terminal():
            return "resign"
        
        # Root node.
        v0 = Node(s0)
        
        ###
        nb_iter = 0
        self.time_block = 0
        ###
        
        time_spent = time.perf_counter() - tic
        while time_spent < self.time_limit:
            tic = time.perf_counter()
            
            v = self.tree_policy(v0)
            reward = self.default_policy(v.s)
            self.backup(v, reward)
            
            time_spent += time.perf_counter() - tic
            
            ###
            nb_iter += 1
            if nb_iter == 100:
                break
            print("-" * 60)
            print("Iteration", nb_iter)
            remaining_time = round(self.time_limit - time_spent)
            print("Remaining time : {} s".format(remaining_time))
            print("Time spent : {} s".format(round(time_spent)))
            print("Time of block : {} s".format(round(self.time_block)))
            self.print_tree(v0)
            ###
        
        ###
        print("-" * 60)
        print("nb iterations :", nb_iter)
        bv = self.best_child(v0, 0)
        print("Q(best_child) = {} ; N(best_child) = {}".format(bv.Q, bv.N))
        print("Win rate : {} %".format(round(100 * bv.Q / bv.N, 2)))
        print("-" * 60)
        print()
        ###
        
        return self.best_child(v0, 0).prev_a
    
    # Print all the nodes of the tree exploring it in breadth-first.
    def print_tree(self, v):
        # We will put in the queue the couples (node, depth level).
        q = queue.Queue()
        q.put((v, 0))
        # The current depth we are exploring.
        cur_dp = -1
        while not q.empty():
            v2, dp2 = q.get(0)
            if dp2 > cur_dp:
                print()
                print("Depth", dp2)
                cur_dp = dp2
            print(v2)
            for v3 in v2.children:
                q.put((v3, dp2+1))
    
    # Starting from the v0 root node, we descend the tree by selecting nodes
    # with their uct value until we reach a node not fully expanded yet.
    # This node is then expanded by one child and the added child is returned.
    # If the descent terminates at a terminal node, the latter is returned.
    def tree_policy(self, v0):
        v = v0
        while not v.s.is_terminal():
            if not v.is_fully_expanded():
                return self.expand(v)
            else:
                v = self.best_child(v, self.C)
        return v
    
    # Randomly select one untried move from v and then add the corresponding child node in the tree.
    # The added child node is returned.
    def expand(self, v):
        # Random choice of an action in untried actions from v. (legal_moves was shuffled)
        a = v.s.legal_moves[v.cursor_a]
        v.cursor_a += 1
        v2 = Node(v.s.next_state(a), v, a)
        v.children.append(v2)
        return v2
    
    # Return the best child of v according to the uct value with constant C.
    def best_child(self, v, C):
        return max(v.children, key=lambda v2: self.uct(v2, C))
    
    # Upper confidence bound for trees.
    def uct(self, v, C):
        if v.N == 0:
            return math.inf
        return v.Q / v.N + C * math.sqrt(2 * math.log(v.parent.N) / v.N)
    
    # Simulate a game with uniformly random moves starting from s.
    # Then it returns 1 if the player at s wins and 0 otherwise.
    def default_policy(self, s):
        # Just make a copy of s.
        s2 = s.next_state('')
        player = s2.playing_side
        while not s2.is_terminal():
            a = random.choice(s2.legal_moves)
            s2.update(a, self)
        return self.reward(s2, player)
    
    # Return 1 if the player wins at s, 0 otherwise.
    def reward(self, s, player):
        bw = int(s.black_wins())
        if player == 'b':
            return bw
        else:
            return 1 - bw
    
    # Update nodes' rewards starting from v which receive "reward" and then
    # backpropagating.
    def backup(self, v, reward):
        v2 = v
        while v2 != None:
            v2.N += 1
            v2.Q += reward
            v2 = v2.parent


class Node:
    
    def __init__(self, s, parent=None, prev_a=None):
        
        # Game state associated to this node.
        self.s = s
        # Total reward accumulated with the visits of this node.
        self.Q = 0
        # Total number of visits of this node.
        self.N = 0
        self.parent = parent
        self.children = []
        # The action (move) from the parent of s to s.
        self.prev_a = prev_a
        # We shuffle legal_moves and we will add each move in the tree in order.
        random.shuffle(self.s.legal_moves)
        # Index of the next action from s in legal_moves which will be added to the tree.
        self.cursor_a = 0
    
    # Return True if the node is fully expanded in the tree, ie. if all of its children
    # have been inserted.
    def is_fully_expanded(self):
        return self.cursor_a == len(self.s.legal_moves)
    
    def __str__(self):
        return "Move : {} ; Q : {} ; N : {} ; Children : {}"\
            .format(self.prev_a, self.Q, self.N, self.str_children())
    
    def str_children(self):
        children_move = [v.prev_a for v in self.children]
        return "[" + ", ".join(children_move) + "]"