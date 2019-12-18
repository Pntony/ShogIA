# -*- coding: utf-8 -*-

import random


class Random_AI:
    
    def __init__(self):
        pass
    
    # Generate a random move from s0 state in USI format.
    def best_move(self, s0):
        if s0.is_terminal():
            return "resign"
        return random.choice(s0.legal_moves)