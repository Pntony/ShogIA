# -*- coding: utf-8 -*-

import numpy as np
import copy
import tkinter as tk
from PIL import Image,ImageTk
###
import time
###

# Useful classes : Game_state, Piece

# Each piece of the same type is numbered from the left to the right 
# in the initial position.
# Black pieces are in upper case while white pieces are in lower case.
# Squares on the board are indexed from the top left to the bottom right.

#______________________________________________________________________________

# indices_to_string
# Convert indices format position for matrices to string format position
# for the USI protocol.
def id2str(indpos):
    row, col = indpos
    strcol = str(9 - col)
    strrow = chr(ord('a') + row)
    return strcol + strrow

# string_to_indices
# Convert string format position for the USI protocol to indices format
# position for matrices.
def str2id(strpos):
    strcol, strrow = strpos[0], strpos[1]
    row = ord(strrow) - ord('a')
    col = 9 - int(strcol)
    return row, col

def other_side(side):
        if side == 'b':
            return 'w'
        else:
            return 'b'


#______________________________________________________________________________

class Piece:
    """
    Useful attributes :
        - pos
        - x and y
        - side
        - captured
        - promotion
        - reachable_squares
    
    Useful methods :
        - __str__
        - usi_name
        - threats
        - can_promote
        - must_promote
    """
    
    def __init__(self, name, pos, side, game_state, captured=False):
        
        # Piece's name used as key in the dictionaries.
        self.name = name
        #  The piece's name in the USI protocol's format.
        self.usi_name = name[0]
        
        # Position on the board
        self._pos = pos
        self._x, self._y = pos
        
        # The side is :
        # -> 'b' if it is controlled by the black side.
        # -> 'w' if it is controlled by the white side.
        self.side = side
        
        # True if captured.
        self.captured = captured
        
        # Takes the value '+' if the piece is promoted, '' otherwise.
        self.promotion = ''
        
        # List of the squares the piece can reach.
        self.reachable_squares = []
        
        # An instance of Game_state class in which the piece is used.
        self.st = game_state
    
    @property
    def pos(self):
        return self._pos
    
    @pos.setter
    def pos(self, new_pos):
        self._pos = new_pos
        if new_pos == None:
            self._x, self._y = None, None
        else:
            self._x, self._y = new_pos
    
    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, new_x):
        self._x = new_x
        self._pos = (new_x, self._y)
    
    @property
    def y(self):
        return self._y
    
    @y.setter
    def y(self, new_y):
        self._y = new_y
        self._pos = (self._x, new_y)
    
    def draw(self,can,x,y): #To draw the piece on the board
        self.nomImage = "pieces\\shogi_" + self.name[0].lower() + self.side[0] + self.promotion + ".png" 
        self.image = Image.open(self.nomImage)
        if self.captured:
            self.image = self.image.resize((25, 25), Image.ANTIALIAS)
        self.photo = ImageTk.PhotoImage(self.image, master = can)
        can.create_image(x + 2,y + 2,anchor = tk.NW, image= self.photo)
    
    def __str__(self):
        if self.pos == None:
            pos = None
        else:
            pos = id2str(self.pos)
        reachable_squares = []
        for square in self.reachable_squares:
            reachable_squares.append(id2str(square))
        if self.captured:
            captured = 'yes'
        else:
            captured = 'no'
        if self.promotion == '+':
            promotion = 'yes'
        else:
            promotion = 'no'
        return "Name: {} ; Pos: {} ; Side: {} ; Captured: {} ; Promotion: {} ; Reachable squares: {}"\
            .format(self.name, pos, self.side, captured, promotion, reachable_squares)
    
    # Method used in reachable_squares_noN to adjust xdir and ydir.
    def adjust_dir(direction):
        if direction > 0:
            return 1
        elif direction < 0:
            return -1
        else:
            return 0
    
    # Update reachable_squares according to board_position for any type of piece, captured or not.
    def update_reachable_squares(self, dpos):
        if self.captured:
            self.update_reachable_squares_drop()
        else:
            self.update_reachable_squares_on_board(dpos)
    
    # Update reachable_squares according to board_position for any type 
    # of piece on the board. Do NOT use this for a captured piece.
    #
    # dpos is the list of the directions (xdir, ydir) allowed by the piece.
    # xdir and ydir can take the values +/-1 if the piece can move only by
    # 1 square along the given direction, +/-2 for an unlimited number of 
    # squares except if xdir = +/-2 and ydir = +/-1 in which case it's a
    # knight's move, and 0 for 0 square.
    #
    # Moves which would threaten the king are put in reachable_squares as well.
    def update_reachable_squares_on_board(self, dpos):
        self.reachable_squares = []
        for xdir, ydir in dpos:
            # Case of a knight's move.
            if abs(xdir) == 2 and abs(ydir) == 1:
                dx, dy = xdir, ydir
            else:
                dx, dy = Piece.adjust_dir(xdir), Piece.adjust_dir(ydir)
            x, y = self.x + dx, self.y + dy
            while True:
                if not (0 <= x <= 8 and 0 <= y <= 8):
                    break
                piece_xy = self.st.board_position[x, y]
                # Update pieces_reaching_it. For this matrix, (x, y) is considered reachable
                # even if there is an ally (but then no further squares beyond this ally).
                self.st.es_pieces_reaching_it[self.side][x, y].append(self)
                # If there isn't any piece at (x, y) or if there is an opponent's piece there,
                # the piece can move towards this square.
                if piece_xy == None or piece_xy.side != self.side:
                    self.reachable_squares.append((x, y))
                    # If it's an one square move or a knight's move or a capturing move,
                    # the piece can't move farther so we stop.
                    if (abs(xdir) <= 1 and abs(ydir) <= 1) \
                    or (abs(xdir) == 2 and abs(ydir) == 1) \
                    or (piece_xy != None and piece_xy.side != self.side):
                        break
                else:
                    break
                x, y = x + dx, y + dy
    
    # Update reachable_squares of a captured piece which could be dropped.
    # Moves threatening my king are put as well.
    def update_reachable_squares_drop(self):
        self.reachable_squares = []
        for x in range(9):
            for y in range(9):
                if self.st.board_position[x, y] == None:
                    # We skip all illegal drops.
                    if type(self) in [Pawn, Lance]:
                        if (self.side == 'b' and x == 0) or (self.side == 'w' and x == 8):
                            continue
                    elif type(self) == Knight:
                        if (self.side == 'b' and x <= 1) or (self.side == 'w' and x >= 7):
                            continue
                    if type(self) == Pawn and (self.pawn_on_column(y) or self.pawn_drop_checkmate(x, y)):
                        continue
                    self.reachable_squares.append((x, y))
                    self.st.es_pieces_reaching_it[self.side][x][y].append(self)
    
    # Remove moves in reachable_squares which threaten my king and then are illegal.
    def remove_moves_threatening_my_king(self):
        # Case of a drop.
        if self.captured:
            head = self.usi_name + '*'
        # Case of a move on the board.
        else:
            head = id2str(self.pos)
        self.reachable_squares = [square for square in self.reachable_squares \
                                  if not self.move_threatening_my_king(head+id2str(square))]
    
    # Return True if a non-promoted pawn of the same side is on column y.
    def pawn_on_column(self, y):
        for x in range(9):
            piece = self.st.board_position[x, y]
            if type(piece) == Pawn and piece.side == self.side and piece.promotion == '':
                return True
        return False
    
    # Return True if the move would threaten my own king, that is a move which will expose my king
    # to the opponent who can then take my king at the next turn, which is illegal.
    def move_threatening_my_king(self, move):
        simul_board_position = self.st.board_position.copy()
        # We only update board_position and then we must be careful about the way
        # we use other variables describing game's state.
        self.update_board_position(move, simul_board_position)
        king = self.st.get_king(self.side)
        king_pos = king.pos
        if type(self) == King:
            king_pos = str2id(move[2:])
        king_x, king_y = king_pos
        
        # We look at one square around the king for eventual enemies, plus squares for eventual
        # threatening knights.
        dpos = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1),
                (-2, -1), (-2, 1), (2, -1), (2, 1)]
        for dx, dy in dpos:
            x = king_x + dx
            y = king_y + dy
            if not (0 <= x <= 8 and 0 <= y <= 8):
                continue
            piece_xy = simul_board_position[x, y]
            # If the piece in (x, y) exists, is an ennemy and can reach the king.
            if piece_xy in self.st.es_pieces_reaching_it[other_side(king.side)][king_x, king_y]:
                return True
        
        # We look forward the king for an eventual enemy rook or lance.
        dx_fw = -1 if king.side == 'b' else 1
        if self.rbl_threaten_king(simul_board_position, king_pos, [(dx_fw, 0)], \
        lambda piece: piece.usi_name.upper() in ['R', '+R', 'L']):
            return True
        
        # We look along lines from the king's position (except forward, already done)
        # for an eventual enemy rook.
        dpos = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dpos.remove((dx_fw, 0))
        if self.rbl_threaten_king(simul_board_position, king_pos, dpos, lambda piece: type(piece) == Rook):
            return True
        
        # We lookg in diagonal from the king for an eventual enemy bishop.
        dpos = [(-1, 1), (1, 1), (1, -1), (-1, -1)]
        if self.rbl_threaten_king(simul_board_position, king_pos, dpos, lambda piece: type(piece) == Bishop):
            return True
        
        return False
    
    # Method used in move_threatening_my_king.
    # "rbl" stands for rook/bishop/lance.
    def rbl_threaten_king(self, board_position, king_pos, dpos, condition_on_piece):
        for dx, dy in dpos:
            x, y = king_pos
            while True:
                x += dx
                y += dy
                if not (0 <= x <= 8 and 0 <= y <= 8):
                    break
                piece_xy = board_position[x, y]
                if piece_xy != None:
                    if piece_xy.side != self.side and condition_on_piece(piece_xy):
                        return True
                    break
        return False
    
    def update_board_position(self, move, board_position):
        # Case of a drop.
        if move[1] == '*':
            dropped_piece = self.st.search_dropped_piece(move, self.side)
            x, y = str2id(move[2:])
            board_position[x, y] = dropped_piece
        # Case of a move on the board.
        else:
            x1, y1 = str2id(move[:2])
            x2, y2 = str2id(move[2:])
            board_position[x2, y2] = board_position[x1, y1]
            board_position[x1, y1] = None
    
    # Return True if dropping a pawn at (x, y) checkmate.
    def pawn_drop_checkmate(self, x, y):
        # Opponent's king.
        op_king = self.st.get_king(other_side(self.side))
        
        # We see if the pawn would be dropped in front of the opponent's king.
        dx_fw = -1 if op_king.side == 'b' else 1
        x_fw = op_king.x + dx_fw
        if (x, y) != (x_fw, op_king.y):
            return False
        
        # If the opponent's king can move.
        if op_king.reachable_squares != []:
            return False
        
        # If the pawn could be captured.
        if self.st.pieces_reaching_it((x, y), other_side(self.side)) != []:
            return False
        
        return True
    
    # Return the list of the opponent's pieces which threaten the piece.
    def threats(self):
        return self.st.pieces_reaching_it(self.pos, other_side(self.side))
    
    # Return True if the piece can be promoted if it moves to square2move.
    def can_promote(self, square2move):
        x, y = square2move
        if self.promotion == "" and type(self) != Gold and type(self) != King and ((self.side == 'b' and x <= 2) or (self.side == 'w' and x >= 6)):
            return True
        return False
    
    # Return True if the piece must promote if it moves to square2move.
    def must_promote(self, square2move):
        x, y = square2move
        name = self.name[0].upper()
        if name in ['P', 'L']:
            if (self.side == 'b' and x == 0) or (self.side == 'w' and x == 8):
                return True
        elif name == 'N':
            if (self.side == 'b' and x <= 1) or (self.side == 'w' and x >= 7):
                return True
        return False


class King(Piece):
    
    def __init__(self, name, pos, side, game_state, captured=False):
        Piece.__init__(self, name, pos, side, game_state, captured)
    
    def update_reachable_squares(self):
        dpos = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
        Piece.update_reachable_squares(self, dpos)


class Rook(Piece):
    
    def __init__(self, name, pos, side, game_state, captured=False):
        Piece.__init__(self, name, pos, side, game_state, captured)
    
    def update_reachable_squares(self):
        dpos = [(-2, 0), (0, 2), (2, 0), (0, -2)]
        if self.promotion == '+':
            dpos = dpos + [(-1, 1), (1, 1), (1, -1), (-1, -1)]
        Piece.update_reachable_squares(self, dpos)


class Bishop(Piece):
    
    def __init__(self, name, pos, side, game_state, captured=False):
        Piece.__init__(self, name, pos, side, game_state, captured)
    
    def update_reachable_squares(self):
        dpos = [(-2, 2), (2, 2), (2, -2), (-2, -2)]
        if self.promotion == '+':
            dpos = dpos + [(-1, 0), (0, 1), (1, 0), (0, -1)]
        Piece.update_reachable_squares(self, dpos)


class Gold(Piece):
    
    def __init__(self, name, pos, side, game_state, captured=False):
        Piece.__init__(self, name, pos, side, game_state, captured)
    
    def update_reachable_squares(self):
        if self.side == 'b':
            dpos = [(-1, 0), (-1, 1), (0, 1), (1, 0), (0, -1), (-1, -1)]
        else:
            dpos = [(-1, 0), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        Piece.update_reachable_squares(self, dpos)


class Silver(Piece):
    
    def __init__(self, name, pos, side, game_state, captured=False):
        Piece.__init__(self, name, pos, side, game_state, captured)
    
    def update_reachable_squares(self):
        if self.promotion == '':
            if self.side == 'b':
                dpos = [(-1, 0), (-1, 1), (1, 1), (1, -1), (-1, -1)]
            else:
                dpos = [(-1, 1), (1, 1), (1, 0), (1, -1), (-1, -1)]
            Piece.update_reachable_squares(self, dpos)
        else:
            Gold.update_reachable_squares(self)


class Knight(Piece):
    
    def __init__(self, name, pos, side, game_state, captured=False):
        Piece.__init__(self, name, pos, side, game_state, captured)
    
    def update_reachable_squares(self):
        if self.promotion == '':
            if self.side == 'b':
                dpos = [(-2, -1), (-2, 1)]
            else:
                dpos = [(2, -1), (2, 1)]
            Piece.update_reachable_squares(self, dpos)
        else:
            Gold.update_reachable_squares(self)


class Lance(Piece):
    
    def __init__(self, name, pos, side, game_state, captured=False):
        Piece.__init__(self, name, pos, side, game_state, captured)
    
    def update_reachable_squares(self):
        if self.promotion == '':
            if self.side == 'b':
                dpos = [(-2, 0)]
            else:
                dpos = [(2, 0)]
            Piece.update_reachable_squares(self, dpos)
        else:
            Gold.update_reachable_squares(self)


class Pawn(Piece):
    
    def __init__(self, name, pos, side, game_state, captured=False):
        Piece.__init__(self, name, pos, side, game_state, captured)
    
    def update_reachable_squares(self):
        if self.promotion == '':
            if self.side == 'b':
                dpos = [(-1, 0)]
            else:
                dpos = [(1, 0)]
            Piece.update_reachable_squares(self, dpos)
        else:
            Gold.update_reachable_squares(self)

#______________________________________________________________________________

class Game_state:
    """
    Useful attributes :
        - es_pieces
        - pieces
        - board_position
        - playing_side
        - legal_moves
    
    Useful methods :
        - print_board
        - print_pieces
        - update
        - check_threats
        - in_check
        - in_checkmate
    """
    
    def __init__(self):
        
        # "es" stands for "each side".
        self.es_pieces = {}
        
        self.es_pieces['w'] = {'l1': Lance('l1', (0, 0), 'w', self), 
                             'n1': Knight('n1', (0, 1), 'w', self), 
                             's1': Silver('s1', (0, 2), 'w', self),
                             'g1': Gold('g1', (0, 3), 'w', self),
                             'k': King('k', (0, 4), 'w', self),
                             'g2': Gold('g2', (0, 5), 'w', self), 
                             's2': Silver('s2', (0, 6), 'w', self), 
                             'n2': Knight('n2', (0, 7), 'w', self),
                             'l2': Lance('l2', (0, 8), 'w', self), 
                             'r': Rook('r', (1, 1), 'w', self),
                             'b': Bishop('b', (1, 7), 'w', self),
                             'p1': Pawn('p1', (2, 0), 'w', self),
                             'p2': Pawn('p2', (2, 1), 'w', self), 
                             'p3': Pawn('p3', (2, 2), 'w', self), 
                             'p4': Pawn('p4', (2, 3), 'w', self),
                             'p5': Pawn('p5', (2, 4), 'w', self),
                             'p6': Pawn('p6', (2, 5), 'w', self),
                             'p7': Pawn('p7', (2, 6), 'w', self),
                             'p8': Pawn('p8', (2, 7), 'w', self),
                             'p9': Pawn('p9', (2, 8), 'w', self)}
        
        self.es_pieces['b'] = {'P1': Pawn('P1', (6, 0), 'b', self),
                             'P2': Pawn('P2', (6, 1), 'b', self),
                             'P3': Pawn('P3', (6, 2), 'b', self),
                             'P4': Pawn('P4', (6, 3), 'b', self),
                             'P5': Pawn('P5', (6, 4), 'b', self),
                             'P6': Pawn('P6', (6, 5), 'b', self),
                             'P7': Pawn('P7', (6, 6), 'b', self),
                             'P8': Pawn('P8', (6, 7), 'b', self),
                             'P9': Pawn('P9', (6, 8), 'b', self),
                             'B': Bishop('B', (7, 1), 'b', self),
                             'R': Rook('R', (7, 7), 'b', self),
                             'L1': Lance('L1', (8, 0),'b', self),
                             'N1': Knight('N1', (8, 1), 'b', self),
                             'S1': Silver('S1', (8, 2), 'b', self),
                             'G1': Gold('G1', (8, 3), 'b', self),
                             'K': King('K', (8, 4), 'b', self),
                             'G2': Gold('G2', (8, 5), 'b', self),
                             'S2': Silver('S2', (8, 6), 'b', self),
                             'N2': Knight('N2', (8, 7), 'b', self),
                             'L2': Lance('L2', (8, 8), 'b', self)}
        
        self.pieces = {**self.es_pieces['w'], **self.es_pieces['b']}
        
        self.board_position = self.init_board_position()
        
        # "es" stands for "each side".
        self.es_pieces_reaching_it = {}
        # Matrix containing at each square x of the board the black pieces which can reach x.
        # Even if there is a black piece at x, we consider that x could still be reached by black pieces
        # though actually not really true. We choose this convention only for more convenience for 
        # the algorithm. However, reachable_squares does contain squares which can truly be reached.
        self.es_pieces_reaching_it['b'] = self.init_pieces_reaching_it()
        self.es_pieces_reaching_it['w'] = self.init_pieces_reaching_it()
        
        self.update_all_reachable_squares()
        self.playing_side = 'b'
        
        # List of all legal moves in USI format that playing_side can do.
        self.legal_moves = []
        
        self.update_legal_moves()
    
    # Return True if the state node is a leaf of the game tree.
    def is_terminal(self):
        return self.legal_moves == []
    
    def black_wins(self):
        return self.is_terminal() and (self.playing_side == 'w')
    
    def init_board_position(self):
        result = np.empty((9, 9), dtype=object)
        for piece in self.pieces.values():
            result[piece.x, piece.y] = piece
        return result
    
    def init_pieces_reaching_it(self):
        result = np.empty((9, 9), dtype=object)
        for i in range(9):
            for j in range(9):
                result[i, j] = []
        return result
    
    # Update legal_moves. Must be called after that playing_side has been updated.
    def update_legal_moves(self):
        self.legal_moves = []
        pieces = self.es_pieces[self.playing_side]
        for piece in pieces.values():
            for square in piece.reachable_squares:
                # Case of a drop.
                if piece.captured:
                    move = piece.usi_name.upper() + '*' + id2str(square)
                    self.legal_moves.append(move)
                # Case of a regular move.
                else:
                    move = id2str(piece.pos) + id2str(square)
                    if piece.must_promote(square):
                        self.legal_moves.append(move + '+')
                    elif piece.can_promote(square):
                        self.legal_moves.append(move)
                        self.legal_moves.append(move + '+')
                    else:
                        self.legal_moves.append(move)
    
    def print_board(self):
        blank = ' '
        print(' ' + blank, end='')
        for i in range(9, 0, -1):
            print(' ' + str(i), end=blank)
        print()
        for i in range(9):
            print(chr(ord('a') + i), end=blank)
            for j in range(9):
                piece = self.board_position[i, j]
                if piece == None:
                    print(' .', end=blank)
                else:
                    blank2 = ' ' if len(piece.usi_name) == 1 else ''
                    print(blank2 + piece.usi_name, end=blank)
            if i < 9:
                print()
    
    def print_pieces(self):
        for piece in self.pieces.values():
            print(piece)
    
    # Return the state after playing the move from this state.
    # If move is the empty string, a copy of the state is returned.
    def next_state(self, move):
        next_st = copy.deepcopy(self)
        if move != '':
            next_st.update(move)
        return next_st
    
    # Update the game's variables (board_position, pieces' attributes, pieces dictionaries, playing_side)
    # according to last_move (in USI protocol format).
    def update(self, last_move, AI=None):
        # Reset pieces_reaching_it.
        self.es_pieces_reaching_it['b'] = self.init_pieces_reaching_it()
        self.es_pieces_reaching_it['w'] = self.init_pieces_reaching_it()
        
        # Case of a drop.
        if last_move[1] == '*':
            dropped_piece = self.search_dropped_piece(last_move, self.playing_side)
            
            dropped_piece.captured = False
            
            # Update piece's position and board_position.
            x, y = str2id(last_move[2:])
            dropped_piece.pos = (x, y)
            self.board_position[x, y] = dropped_piece
        
        # Case of a regular move.
        else:
            x1, y1 = str2id(last_move[:2])
            x2, y2 = str2id(last_move[2:4])
            piece1 = self.board_position[x1, y1]
            piece2 = self.board_position[x2, y2]
            
            # Update piece1's state.
            piece1.pos = (x2, y2)
            if piece1.promotion == '':
                # last_move[4:] is '+' if there is a promotion, '' otherwise.
                piece1.promotion = last_move[4:]
                piece1.usi_name = last_move[4:] + piece1.usi_name
            
            # Case where there was a piece in (x2, y2) (then captured).
            if piece2 != None:
                # Update piece2's state.
                self.change_usi_name(piece2)
                self.change_piece_side(piece2)
                piece2.side = self.playing_side
                piece2.captured = True
                piece2.pos = None
                piece2.promotion = ''
            
            # Update board_position.
            self.board_position[x2, y2] = self.board_position[x1, y1]
            self.board_position[x1, y1] = None
        
        ###
        tic = time.perf_counter()
        ###
        self.update_all_reachable_squares()
        ###
        tic = time.perf_counter()
        ###
        self.remove_all_illegal_moves()
        ###
        if AI:
            AI.time_block += time.perf_counter() - tic
        ###
        
        self.playing_side = other_side(self.playing_side)
        
        self.update_legal_moves()
    
    # Update reachable_squares for every pieces (could be optimized).
    def update_all_reachable_squares(self):
        for piece in self.pieces.values():
            piece.update_reachable_squares()
    
    # Remove all illegal moves from reachable_squares of each pieces.
    def remove_all_illegal_moves(self):
        for piece in self.pieces.values():
            piece.remove_moves_threatening_my_king()
    
    # Return the piece which was dropped.
    def search_dropped_piece(self, move, side):
        dropped_piece_name = move[0]
        for piece in self.es_pieces[side].values():
            if piece.usi_name.upper() == dropped_piece_name.upper() and piece.captured:
                return piece
        raise Exception("Dropped piece not found: " + move)
    
    # Change the piece's usi_name into the other side's format.
    def change_usi_name(self, piece):
        # '+' is eventually erased.
        piece.usi_name = piece.usi_name[-1]
        if piece.usi_name.islower():
            piece.usi_name = piece.usi_name.upper()
        else:
            piece.usi_name = piece.usi_name.lower()
    
    # Move "piece" from its side's list of pieces to the other side's list.
    def change_piece_side(self, piece):
        side1_pieces = self.es_pieces[piece.side]
        side2_pieces = self.es_pieces[other_side(piece.side)]
        side2_pieces[piece.name] = side1_pieces[piece.name]
        side1_pieces.pop(piece.name)
    
    # Return the list of the pieces of "side" which can reach "pos".
    def pieces_reaching_it(self, pos, side):
        pieces_list = []
        for piece in self.es_pieces[side].values():
            if pos in piece.reachable_squares:
                pieces_list.append(piece)
        return pieces_list
    
    # Return the list of the opponent's pieces which put "side" in check.
    def check_threats(self, side):
        if side == 'b':
            king_name = 'K'
        else:
            king_name = 'k'
        return self.pieces[king_name].threats()
    
    # Return the king of 'side'.
    def get_king(self, side):
        if side == 'b':
            king_name = 'K'
        else:
            king_name = 'k'
        return self.pieces[king_name]
    
    # Return True if 'side' is in check.
    # (slightly more optimized than "self.check_threats(side) != []" because the latter
    # looks for every threats while only one threat is needed)
    def in_check(self, side):
        king = self.get_king(side)
        opponent_pieces = self.es_pieces[other_side(side)]
        for piece in opponent_pieces.values():
            if king.pos in piece.reachable_squares:
                return True
        return False
    
    # Return True if 'side' is in checkmate.
    def in_checkmate(self, side):
        if self.in_check(side):
            for piece in self.es_pieces[side].values():
                if piece.reachable_squares != []:
                    return False
            return True
        return False
