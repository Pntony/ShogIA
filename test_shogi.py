# -*- coding: utf-8 -*-

import shogi_engine as egn

game = egn.Game_state()

print()
print("Initial board position :")
game.print_board()
i = 1
while True:
    move = input("Move number {} by {} : ".format(i, game.playing_side))
    if move == "quit":
        break
    print("Board after move :")
    game.update_game_state(move)
    game.print_board()
    i += 1