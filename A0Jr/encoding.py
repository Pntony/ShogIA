import torch
import numpy as np
from shogi_engine import str2id


pieces_codes = {
    'P': 0, 'L': 1, 'N': 2, 'S': 3, 'G': 4, 'B': 5, 'R': 6, 'K': 7,
    '+P': 8, '+L': 9, '+N': 10, '+S': 11, '+B': 12, '+R': 13
}


# Return the shogi game state "s" encoded into a pytorch tensor which will be a stack
# of 9x9 planes of binary numbers (bitboards). It will be the input of the neural net.
# The encoding is the one described in the paper on AlphaZero but without the repetitions
# planes and the T time-steps history.
def encode_game_state(s):
    t = torch.zeros(44, 9, 9)
    for piece in s.pieces.values():
        side_code = 0 if piece.side == s.playing_side else 1
        piece_code = pieces_codes[piece.usi_name.upper()]
        if not piece.captured:
            code = side_code * 14 + piece_code
            t[code, piece.x, piece.y] = 1
        else:
            code = 28 + side_code * 7 + piece_code
            t[code] += 1
    if s.playing_side == 'b': t[42] = 1
    t[43] = s.nb_moves
    return t


# Return the indices of the move 'a' in the USI protocol format for the policy tensor.
# Moves encoding is the one described in the paper on AlphaZero.
def a_id(a):

    # Case of a drop.

    if a[1] == '*':
        code = pieces_codes[a[0]]
        x, y = str2id(a[2:])
        return (132 + code, x, y)
    
    # Case of a regular move.

    x1, y1 = str2id(a[:2])
    x2, y2 = str2id(a[2:4])
    # Planes for promoting moves start at index 66.
    prom_id = 66 if a[4:] else 0
    dx = x2 - x1
    dy = y2 - y1

    # Case of a knight move.
    if abs(dx) == 2 and abs(dy) == 1:
        knight_id = int(dy > 0)
        return (knight_id + 64 + prom_id, x1, y1)
    
    # Case of a queen move.
    nb_sq = max(abs(dx), abs(dy))
    direction = (np.sign(dx) + 1) + 3 * (np.sign(dy) + 1)
    # dx = 0 and dy = 0 doesn't happen.
    if direction > 4: direction -= 1
    queen_id = direction + 8 * (nb_sq - 1)
    return (queen_id + prom_id, x1, y1)