# -*- coding: utf-8 -*-

import re
import random
import shogi_engine as egn


class Random_AI:
    
    def __init__(self):
        self.game = egn.Game_state()

    def usi_engine(self):
        while True:
            # Catch the command sent by the GUI.
            usi_command = input()
            
            # The engine is being registered.
            if usi_command == "usi":
                print("id name Random AI")
                print("usiok")
            
            elif usi_command == "isready":
                print("readyok")
            
            # Update the engine variables according to the opponent's last move.
            elif re.search("position startpos moves", usi_command):
                last_move = usi_command.split()[-1]
                self.game.update_game_state(last_move)
            
            # Make a move and then update the variables.
            elif re.search("go", usi_command):
                best_move = self.random_move()
                if best_move != "resign":
                    self.game.update_game_state(best_move)
                print("bestmove", best_move)
            
            elif usi_command == "quit":
                break
    
    def plays_itself(self, nb_moves, print_moves=False):
        for i in range(1, nb_moves+1):
            best_move = self.random_move()
            if best_move == "resign":
                self.game.print_board()
                print()
                print("Total moves :", i-1)
                print()
                print("{} resigns".format(self.game.playing_side))
                break
            if print_moves:
                print("Move number {}".format(i))
                print("{} : {}".format(self.game.playing_side, best_move))
                print("Board before move :")
                self.game.print_board()
                print('\n')
            self.game.update_game_state(best_move)
    
    def plays_against_you(self):
        print()
        player_side = input("Select your side (b or w) : ")
        print()
        print("Initial board position :")
        self.game.print_board()
        print('\n')
        i = 1
        while True:
            if self.game.playing_side == player_side:
                move = input("Move number {} by {} : ".format(i, self.game.playing_side))
            else:
                move = self.random_move()
                print("Move number {} by {} : {}".format(i, self.game.playing_side, move))
            if move == "quit":
                break
            elif move == "resign":
                print("{} resigns".format(self.game.playing_side))
                break
            self.game.update_game_state(move)
            print("Board after move :")
            self.game.print_board()
            print('\n')
            i += 1
    
    # Generate a random move in the USI protocol format according to the game's state.
    def random_move(self):
        # First, randomly pick a piece to move.
        pieces = self.game.es_pieces[self.game.playing_side]
        pieces_names = list(pieces.keys())
        random.shuffle(pieces_names)
        piece2move = None
        for piece_name in pieces_names:
            piece = pieces[piece_name]
            # If the piece can be moved.
            if piece.reachable_squares != []:
                piece2move = piece
                break
        
        # Case of no legal move found.
        if piece2move == None:
            return "resign"
        
        # Randomly pick a square to move.
        square2move = random.choice(piece2move.reachable_squares)
        
        # Case of a drop.
        if piece.captured:
            best_move = piece.usi_name + '*' + egn.id2str(square2move)
        
        # Case of a regular move.
        else:
            best_move = egn.id2str(piece2move.pos) + egn.id2str(square2move)
            # Handle promotion.
            if piece2move.promotion == '':
                if piece2move.must_promote(square2move):
                    best_move += '+'
                elif piece2move.can_promote(square2move):
                    if random.choice([0, 1]):
                        best_move += '+'
        
        return best_move


if __name__ == "__main__":
    random.seed(0)
    rand_AI = Random_AI()
    
#    rand_AI.usi_engine()
    rand_AI.plays_itself(1000, print_moves=False)
#    rand_AI.plays_against_you()