import torch
import numpy as np
from shogi_engine import id2str, str2id


pieces_codes = {
    'P': 1, 'L': 2, 'N': 3, 'S': 4, 'B': 5, 'R': 6, 'G': 7, 'K': 8,
    '+P': 9, '+L': 10, '+N': 11, '+S': 12, '+B': 13, '+R': 14,
}


# Return the shogi game state "s" encoded into a pytorch tensor which will be a stack
# of 9x9 planes of binary numbers (bitboards). It will be the input of the neural net.
# The encoding is the one described in the paper on AlphaZero but without the repetitions
# planes and the T time-steps history.
def encode_game_state(s):
    t = torch.zeros(43, 9, 9)
    for piece in s.pieces.values():
        side_code = 0 if piece.side == s.playing_side else 1
        piece_code = pieces_codes[piece.usi_name.upper()] - 1
        if not piece.captured:
            code = side_code * 14 + piece_code
            t[code, piece.x, piece.y] = 1
        else:
            code = 28 + side_code * 7 + piece_code
            t[code] += 1
    if s.playing_side == 'b': t[42] = 1
    return t


# Same as encode_game_state but for LightGameState arrays.
def encode_light_game_state(ls):

    t = torch.zeros(43, 9, 9)
    ls_side = ls[-1,-2]

    # For pieces on the board.
    for x in range(9):
        for y in range(9):
            piece = ls[x,y]
            if not piece: continue
            side_code = 0 if np.sign(piece) == (1-2*ls_side) else 1
            piece_code = abs(piece) - 1
            code = side_code * 14 + piece_code
            t[code, x, y] = 1
    
    # For pieces in hand.
    for side in range(2):
        for piece in range(7):
            side_code = 0 if side == ls_side else 1
            code = 28 + side_code * 7 + piece
            t[code] = int(ls[9+side, piece])
    
    if ls_side == 0: t[42] = 1
    return t


def get_usi_name(code, side):
    name = list(pieces_codes.keys())[list(pieces_codes.values()).index(code)]
    return name if side == 'b' else name.lower()


def decode_game_state(t):

    ally = 'b' if torch.any(t[42] == 1) else 'w'
    enemy = 'w' if ally == 'b' else 'b'

    board = np.empty((9,9), dtype=object)
    for x in range(9):
        for y in range(9):
            board[x, y] = ''
            for k in range(14):
                if t[k, x, y]:
                    board[x, y] = get_usi_name(k+1, ally)
                if t[k+14, x, y]:
                    board[x, y] = get_usi_name(k+1, enemy)
    
    hands = {'b': {}, 'w': {}}
    for k in range(28, 35):
        piece = get_usi_name(k-27, 'b')
        hands[ally][piece] = int(t[k,0,0].item())
        hands[enemy][piece] = int(t[k+7,0,0].item())
    
    return (board, hands, ally)


def string_board(board):
    res = ''
    blank = ' '
    res += ' ' + blank
    for i in range(9, 0, -1):
        res += ' ' + str(i) + blank
    res += '\n'
    for i in range(9):
        res += chr(ord('a') + i) + blank
        for j in range(9):
            piece = board[i, j]
            if not piece:
                res += ' .' + blank
            else:
                blank2 = ' ' if len(piece) == 1 else ''
                res += blank2 + piece + blank
        if i < 9:
            res += '\n'
    return res


def string_decoded_state(dec):
    board, hands, ally, nb_moves = dec
    return (
        f"side: {ally} | moves: {nb_moves}" +
        f"\n\n{string_board(board)}" +
        f"\nb: {hands['b']}" +
        f"\nw: {hands['w']}"
    )


# Return the indices of the move 'a' in the USI protocol format in the policy tensor.
# Moves encoding is the one described in the paper on AlphaZero.
def a_id(a):

    # Case of a drop.

    if a[1] == '*':
        code = pieces_codes[a[0]] - 1
        x, y = str2id(a[2:])
        return (132 + code, x, y)
    
    # Case of a regular move.

    x1, y1 = str2id(a[:2])
    x2, y2 = str2id(a[2:4])
    # Planes for promoting moves start at index 66.
    prom_id = 66 if a[4:5] == '+' else 0
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


def decode_a_id(a_id, side):

    plane, x1, y1 = a_id

    # Case of a drop.

    if 132 <= plane <= 138:
        piece = get_usi_name(plane-131, 'b')
        return f"{piece}*{id2str((x1, y1))}"
    
    # Case of a regular move.

    # Case of a knight move.
    if 64 <= (plane % 66) <= 65:
        dx = -2 if side == 'b' else 2
        dy = -1 if (plane % 66) == 64 else 1
        
        x2 = x1 + dx
        y2 = y1 + dy

    # Case of a queen move.
    else:
        nb_sq = (plane % 66) // 8 + 1

        direction = (plane % 66) % 8
        if direction >= 4: direction += 1

        dx = direction % 3 - 1
        dy = direction // 3 - 1

        x2 = x1 + nb_sq * dx
        y2 = y1 + nb_sq * dy

    prom = '+' if plane // 66 == 1 else ''

    return id2str((x1, y1)) + id2str((x2, y2)) + prom


# Return the list of the n couples composed of the
# n biggest probabilities in p and their corresponding moves.
def max_moves(p, n):

    p = p.detach().cpu().numpy()
    side = 'b' if p[42, 0, 0] == 1 else 'w'

    idx = p.flatten().argsort()[-n:][::-1]
    idx = np.unravel_index(idx, p.shape)
    idx = [
        tuple([idx[x][i] for x in range(3)]) \
        for i in range(n)
    ]

    moves_max = []
    for a_id in idx:
        moves_max.append((
            round(p[a_id], 3), decode_a_id(a_id, side)
        ))

    return moves_max


# Print the max_moves of pi and exp(log_p).
def print_max_moves(pi, log_p, n):
    P = torch.exp(log_p)
    loss = -(pi * log_p).sum()
    for i in range(len(P)):
        print(
            "_" * 100 + "\n\n" +
            f"batch num: {i+1}"
            f"\n\npi max: {max_moves(pi[i], n)}"
            f"\n\np max: {max_moves(P[i], n)}"
            f"\n\nloss: {round(loss.item(), 2)}"
        )