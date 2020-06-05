# -*- coding: utf-8 -*-

import torch
from torch.distributions.dirichlet import Dirichlet
import math
import copy

import shogi_engine as egn
from encoding import *


class AI_vs_player:

    def __init__(self, nnet, params):
        self.ai = MCTS_NN(nnet, params)
    
    # Return the best move to do in the current state stored by the AI.
    # The state is also updated according to the chosen move.
    def next_move(self):
        if self.ai.s.finished(): return "resign"
        self.ai.run_MCTS()
        best_n = self.ai.max_pi_node()
        best_a = best_n.prev_a
        self.ai.update_tree(best_n)
        return best_a
    
    # Update the current state stored by the AI after the move 'a' has been made
    # by an opponent.
    def update(self, a):
        self.ai.update_tree(self.ai.get_child(a))


class Self_play:

    def __init__(self, nnet, params):
        self.ai = MCTS_NN(nnet, params)

        self.train_set_size = 0
        self.aimed_train_set_size = 0
        self.nb_iter = params["nb_iterations"]

    # Build a train set for the neural network playing against itself until
    # reaching a train set of size at least "train_set_size".
    ###
    def build_train_set(self, nb_games_per_iter, iter=0):
    ###
        train_set = []
        nb_games = 0
        while nb_games < nb_games_per_iter:
            best_n = None
            while not (best_n and best_n.is_leaf()):
                ###
                self.ai.run_MCTS(iter, self.nb_iter, len(train_set), nb_games)
                ###
                s = encode_game_state(self.ai.s)
                train_set.append([s, self.ai.pi, self.ai.root.Q / 2])
                best_n = self.ai.max_pi_node()
                ###
                self.train_set_size += 1
                print(f"move: {best_n.prev_a} ; " \
                    f"p: {round(best_n.p, 3)} ; " \
                    f"N0: {self.ai.root0.N}")
                ###
                # If a repetition is found, the game finishes with a draw.
                # if self.ai.seen_in_game(best_n):
                #     break
                self.ai.update_tree_self_play(best_n)
            # If the game was played until the end, we assign rewards, but if not,
            # that is if there was a repetition, the game's outcome is a draw and
            # we leave rewards to Q/2.
            if self.ai.s.finished():
                self.ai.assign_rewards(train_set)
                ###
                self.ai.params["ended_games"] += 1
                ###
            nb_games += 1
        return train_set
    
    # Update the neural net and reset the current root node to the original
    # root node of the tree.
    def update(self, nnet):
        self.ai.nnet = nnet
        self.ai.root = self.ai.root0
        self.s = egn.Game_state()


class MCTS_NN:
    # The reward at a given node v is from the perspective of the playing side at v's parent,
    # that is a big reward will always be good for both sides unlike trees in minimax.
    
    ###
    def __init__(self, nnet, params):
    ###
        self.root0 = None
        # The search tree. During the whole game, the same tree is used: the Q and N values
        # at each node are kept after each move. No reset. However, after a move,
        # the root is updated to the current state and its parent is removed only to keep
        # the useful part of the tree in order to save memory.
        self.root = None
        # The game state associated to the root.
        self.s = None
        # The set of all the previous game states reached during the game.
        # One state is represented using the triplet of the numbers encoding
        # the last move from the state plus the prior probability "p".
        # So it's a quadruplet.
        self.hist = None
        # The neural network used to evaluate states and get a policy.
        self.nnet = nnet
        # The improved policy "pi" of the root node computed after a run of the MCTS.
        self.pi = torch.zeros(139, 9, 9).numpy()

        # Total number of simulations for one run of MCTS.
        self.nb_simul = params["nb_simul"]
        # The alpha parameter for the dirichlet noise added to prior probabilities.
        self.alpha_dir = params["alpha_dir"]
        # The temperature parameter in the improved policy formula.
        self.tau = params["tau"]
        # The constant in the PUCT formula.
        self.c_puct = params["c_puct"]
        # The constant in PUCT formula determining the amount of noise added.
        self.eps_n = params["eps_n"]
        ###
        self.params = params
        ###

        self.init_tree()
    
    # Initialize the search tree for the MCTS runs.
    def init_tree(self):
        self.root0 = Node()
        self.root = self.root0
        self.root.N = 1
        self.s = egn.Game_state()
        self.hist = set()
        self.expand(self.root, self.s)
        self.get_nnet_output(self.s, self.root)
    
    # Update the search tree following the move to the node "n" which has just been made.
    # The root is set to the new current node and its parent is removed.
    def update_tree(self, n):
        self.s.update(n.prev_a)
        self.hist.add(self.code(n))
        self.root = n
        self.root.parent = None
        self.root.prev_a = None
        if self.root.N == 0:
            self.root.N = 1
            self.expand(self.root, self.s)
            self.get_nnet_output(self.s, self.root)
        self.pi = torch.zeros(139, 9, 9).numpy()
    
    def update_tree_self_play(self, n):
        self.s.update(n.prev_a)
        self.hist.add(self.code(n))
        self.root = n
        if self.root.N == 0:
            self.root.N = 1
            self.expand(self.root, self.s)
            self.get_nnet_output(self.s, self.root)
        self.pi = torch.zeros(139, 9, 9).numpy()
    
    # Return the root's child with the move 'a'.
    def get_child(self, a):
        return next(n for n in self.root.children if n.prev_a == a)
    
    # Return the best node to move to according to the improved policy "pi".
    def max_pi_node(self):
        return max(self.root.children, key=lambda n: self.pi[n.a_id])
    
    # Return True if the state associated to the node "n" has already
    # been previously encountered during the game.
    def seen_in_game(self, n):
        return self.code(n) in self.hist
    
    # Return the signature code of the game state associated to the node "n".
    # It's the triplet of the numbers encoding the last move from the state
    # plus the prior probability "p". So it's a quadruplet.
    def code(self, n):
        return (*n.a_id, round(n.p, 3))
    
    # Assign the rewards z to each sample of the train set.
    # The final reward is the mean of the Q value evaluated
    # by the neural net and the z value.
    # The rewards are from the perspective of the current player.
    def assign_rewards(self, train_set):
        # The player who plays at the last state of the train set
        # is the winner (the last state of the game isn't in the set).
        # So this last state is bad for the previous player.
        z = -1
        i = len(train_set) - 1
        while i >= 0 and not train_set[i][2]:
            train_set[i][2] = train_set[i][2] + z / 2
            z = -z
            i -= 1
    
    # Run the MCTS with the current search tree with "nb_simul" simulations.
    # The function just develops the search tree and computes the improved policy "pi"
    # without returning anything.
    ###
    def run_MCTS(self, iter=0, nb_iter=0, train_set_size=0, nb_games=0):
    ###
        for i in range(self.nb_simul):
            n, s = self.select_leaf()
            is_draw = False
            if s.finished():
                # If the game is finished, it's good for the previous player
                # so he gets a reward of 1.
                v = 1
            elif self.seen_in_game(n) or self.visited(n):
                v = -1
                is_draw = True
            else:
                self.expand(n, s)
                v = self.get_nnet_output(s, n)
            self.backup(n, v, is_draw)
        ###
        print(f"\niter: {iter}/{nb_iter} ; " \
            f"train set size: {train_set_size} ; "\
            f"depth: {self.root.depth} ; "\
            f"nb games: {nb_games}/{self.params['nb_games_per_iter']} ; " \
            f"nb ended games: {self.params['ended_games']}")
        ###
        self.compute_improved_policy()
    
    # Starting from the root node, we descend the tree by selecting nodes
    # maximizing the UCB until we reach a leaf node or a terminal node and we return it.
    # A dirichlet noise is added to the prior probabilities for the root node.
    def select_leaf(self):
        n = self.root
        s = copy.deepcopy(self.s)
        while not n.is_leaf():
            # We compute the UCBs only the moment we need them to avoid
            # wasting time and memory by computing them systematically during the expansion.
            # The root is excluded.
            if n.N == 1 and n.parent: self.update_children_ucb(n)
            if n.parent: n = n.max_ucb_node()
            # We add a noise to the policy tensor for the root.
            else: n = self.max_ucb_noise_node(n)
            s.update(n.prev_a)
        return n, s
    
    # Add the children of "n" with the state "s" in the search tree.
    def expand(self, n, s):
        for a in s.legal_moves:
            n.children.append(Node(parent=n, prev_a=a))
    
    # Compute and store the neural network's policy given the input "s", the state of "n",
    # and return the evaluation of the node.
    def get_nnet_output(self, s, n):
        # We add the batch axis to the tensor.
        s = encode_game_state(s).unsqueeze(0).cuda()
        with torch.no_grad():
            log_P, v = self.nnet(s)
        P = torch.exp(log_P).squeeze().cpu().numpy()
        sum = 0
        for n1 in n.children:
            n1.p = P[n1.a_id]
            sum += n1.p
        # We renormalize these prior probabilities because illegal moves in "P" are implicitly
        # masked to 0.
        for n1 in n.children:
            n1.p /= sum
        return v
    
    # Update Q and N values of nodes starting from n which is evaluated by the neural network
    # and then backpropagate, inverting the evaluation v at each layer since the side is inverted.
    # We also update at the same time the UCBs and the children's lists to keep them sorted
    # in the UCB descending order.
    def backup(self, n, v, is_draw):
        n1 = n
        v1 = v
        d = 0
        while n1:
            n1.N += 1
            n1.W += v1
            n1.Q = n1.W / n1.N
            # Update of the n1's UCB and its parent's best node regarding the UCB.
            # The root's children are treated apart during the selection process
            # because of the additional noise.
            if n1.parent and n1.parent.parent:
                n1.ucb = self.ucb(n1)
                self.reorder_children(n1.parent)
            if d > n1.depth: n1.depth = d

            n1 = n1.parent
            if not is_draw: v1 = -v1
            d += 1
    
    # Compute the improved policy "pi" after that the tree has been built.
    def compute_improved_policy(self):
        tau_inv = 1 / self.tau
        cst = sum(n.N ** tau_inv for n in self.root.children)
        for n in self.root.children:
            self.pi[n.a_id] = n.N ** tau_inv / cst
    
    # Compute the n's children's UCBs and sort the list of children
    # in the UCB's descending order.
    def update_children_ucb(self, n):
        for n1 in n.children:
            n1.ucb = self.ucb(n1)
        n.children.sort(key=lambda n1: n1.ucb, reverse=True)
    
    # Eventually reorder the n's children list by descending the previous
    # node maximizing the UCB if its UCB decreased too much.
    def reorder_children(self, n):
        c = n.children
        for i in range(1, len(c)):
            if c[i-1].ucb >= c[i].ucb: break
            c[i-1], c[i] = c[i], c[i-1]

    # Return the n's child node maximizing the UCB
    # with an additional noise to the policy tensor.
    def max_ucb_noise_node(self, n):
        Nc = len(n.children)
        dir_dist = Dirichlet(torch.zeros(Nc) + self.alpha_dir)
        noises = dir_dist.sample().numpy()
        i_max = max(range(Nc), key=lambda i: self.ucb_noise(n.children[i], noises[i]))
        return n.children[i_max]
    
    # Upper confidence bound based on the PUCT algorithm
    # (predictor + upper confidence bound for trees).
    def ucb(self, n):
        return n.Q + self.c_puct * n.p * math.sqrt(n.parent.N) / (1 + n.N)
    
    # UCB with an additional noise added to the prior probability.
    def ucb_noise(self, n, noise):
        return n.Q + self.c_puct * ((1 - self.eps_n) * n.p + self.eps_n * noise) \
            * math.sqrt(n.parent.N) / (1 + n.N)
    
    # Return True if the game state associated to the node "n" has already
    # been visited in one of the n's ancestors in the search tree.
    def visited(self, n):
        n1 = n.parent
        while n1 and n1.prev_a:
            if self.code(n) == self.code(n1):
                return True
            if self.root.prev_a and self.code(n1) == self.code(self.root):
                break
            n1 = n1.parent
        return False


class Node:
    
    def __init__(self, parent=None, prev_a=None):
        
        # Visit count of this node.
        self.N = 0
        # The total cumulated evaluation.
        self.W = 0
        # Action value of this node (average cumulated evaluation ; W / N).
        self.Q = 0
        # The prior probability given by the neural net.
        self.p = 0
        # The UCB of this node.
        self.ucb = 0

        self.parent = parent
        # The move from the parent of s to s.
        self.prev_a = prev_a
        # List of this node's children sorted in their UCBs descending order.
        # So the first child is the one maximizing the UCB.
        self.children = []

        # Indices of the move in the policy tensor. Not defined for the root.
        if self.prev_a:
            self.a_id = a_id(self.prev_a)
        # Depth of the tree starting from this node.
        # It's updated during the backup.
        self.depth = 0
    
    # Return True if the node is a leaf, that is a node without any child.
    def is_leaf(self):
        return self.N == 0
    
    # Return the node's child maximizing the UCB.
    def max_ucb_node(self):
        return self.children[0]