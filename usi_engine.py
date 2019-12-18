# -*- coding: utf-8 -*-

import shogi_engine as egn

import random_AI
import mcts_AI

import re
import random


class USI_engine:
    # The AI class used must have a "best_move" method. This method must take as an argument
    # a shogi_engine.Game_state object s which describes the current state of the game.
    # It must return a legal move from s in the USI protocol format ("resign" if it resigns).
    
    def __init__(self, AI):
        self.state = egn.Game_state()
        self.AI = AI

    def usi_engine(self, usi_engine_name):
        while True:
            # Catch the command sent by the GUI.
            usi_command = input()
            
            # The engine is being registered.
            if usi_command == "usi":
                print("id name", usi_engine_name)
                print("usiok")
            
            elif usi_command == "isready":
                print("readyok")
            
            # Update the engine variables according to the opponent's last move.
            elif re.search("position startpos moves", usi_command):
                last_move = usi_command.split()[-1]
                self.state.update(last_move)
            
            # Make a move and then update the variables.
            elif re.search("go", usi_command):
                best_move = self.AI.best_move(self.state)
                if best_move != "resign":
                    self.state.update(best_move)
                print("bestmove", best_move)
            
            elif usi_command == "quit":
                break
    
    def plays_itself(self, nb_moves, print_moves=False):
        for i in range(1, nb_moves+1):
            best_move = self.AI.best_move(self.state)
            if best_move == "resign":
                print()
                self.state.print_board()
                print()
                print("Total moves :", i-1)
                print()
                print("{} resigns".format(self.state.playing_side))
                break
            if print_moves:
                print()
                print("Move number {}".format(i))
                print("{} : {}".format(self.state.playing_side, best_move))
                print("Board before move :")
                self.state.print_board()
                print('\n')
            self.state.update(best_move)
    
    def plays_against_you(self):
        print()
        player_side = input("Select your side (b or w) : ")
        print()
        print("Initial board position :")
        self.state.print_board()
        print('\n')
        i = 1
        while True:
            if self.state.playing_side == player_side:
                move = input("Move number {} by {} : ".format(i, self.state.playing_side))
            else:
                move = self.AI.best_move(self.state)
                print("Move number {} by {} : {}".format(i, self.state.playing_side, move))
            if move == "quit":
                break
            elif move == "resign":
                print("{} resigns".format(self.state.playing_side))
                break
            self.state.update(move)
            print("Board after move :")
            self.state.print_board()
            print()
            i += 1


if __name__ == "__main__":
    
#    AI = random_AI.Random_AI()
    AI = mcts_AI.MCTS_AI(60, 2)
    
    random.seed(0)
    
    usi_egn = USI_engine(AI)
    
#    usi_egn.usi_engine("Random AI")
#    usi_egn.plays_itself(1000, print_moves=False)
    usi_egn.plays_against_you()