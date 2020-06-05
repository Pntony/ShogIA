# -*- coding: utf-8 -*-

import numpy as np
import copy
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

# Return the usi format of the move from 'sq1' to 'sq2' (squares).
def usi_move(sq1, sq2):
    return id2str(sq1) + id2str(sq2)

# Return the usi format of the drop of 'piece' on 'sq'.
def usi_drop(piece, sq):
    return piece.usi_name.upper() + '*' + id2str(sq)

# Return True if there is any piece between 'piece1' and 'piece2' on 'board'.
def found_piece_btw(board, piece1, piece2):
    dx = np.sign(piece2.x - piece1.x)
    dy = np.sign(piece2.y - piece1.y)
    x, y = piece1.x + dx, piece1.y + dy
    while (x, y) != piece2.pos:
        if board[x, y] != None:
            return True
        x += dx
        y += dy
    return False


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
    
    def __init__(self, name, pos, side, captured=False):
        
        # Piece's name used as key in the dictionaries.
        self.name = name
        #  The piece's name in the USI protocol's format.
        self.usi_name = name[0]
        
        # Position on the board.
        self._pos = pos
        self._x, self._y = pos
        
        # The side is :
        # -> 'b' if it is controlled by the black side.
        # -> 'w' if it is controlled by the white side.
        self._side = side
        
        # True if captured.
        self.captured = captured
        
        # Takes the value '+' if the piece is promoted, '' otherwise.
        self._promotion = ''
    
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
    
    @property
    def side(self):
        return self._side
    
    @side.setter
    def side(self, new_side):
        self._side = new_side
    
    @property
    def promotion(self):
        return self._promotion
    
    @promotion.setter
    def promotion(self, new_prom):
        self._promotion = new_prom
    
    def __str__(self):
        if self.pos == None:
            pos = None
        else:
            pos = id2str(self.pos)
        if self.captured:
            captured = 'yes'
        else:
            captured = 'no'
        return "Name: {} ; Usi name: {} ; Pos: {} ; Captured: {}"\
            .format(self.name, self.usi_name, pos, captured)
    
    # Return True if the piece can be promoted if it moves to square2move.
    def can_promote(self, square2move):
        if self.promotion == '+' or type(self) in {King, Gold}:
            return False
        x, y = square2move
        if (self.side == 'b' and x <= 2) or (self.side == 'w' and x >= 6):
            return True
        return False
    
    # Return True if the piece must promote if it moves to square2move.
    def must_promote(self, square2move):
        if self.promotion == '+':
            return False
        x, y = square2move
        if type(self) in {Pawn, Lance}:
            if (self.side == 'b' and x == 0) or (self.side == 'w' and x == 8):
                return True
        elif type(self) == Knight:
            if (self.side == 'b' and x <= 1) or (self.side == 'w' and x >= 7):
                return True
        return False


class King(Piece):
    
    def __init__(self, name, pos, side, captured=False):
        Piece.__init__(self, name, pos, side, captured)
        self.sdpos = set()
        self.dpos = {(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)}


class Rook(Piece):
    
    def __init__(self, name, pos, side, captured=False):
        Piece.__init__(self, name, pos, side, captured)
        # Sliding dpos.
        self.sdpos = {(-1, 0), (0, 1), (1, 0), (0, -1)}
        # No slide.
        self.dpos = set()
    
    @property
    def promotion(self):
        return self._promotion
    
    @promotion.setter
    def promotion(self, new_prom):
        self._promotion = new_prom
        if new_prom == '':
            self.dpos = set()
        else:
            self.dpos = {(-1, 1), (1, 1), (1, -1), (-1, -1)}


class Bishop(Piece):
    
    def __init__(self, name, pos, side, captured=False):
        Piece.__init__(self, name, pos, side, captured)
        self.sdpos = {(-1, 1), (1, 1), (1, -1), (-1, -1)}
        self.dpos = set()
    
    @property
    def promotion(self):
        return self._promotion
    
    @promotion.setter
    def promotion(self, new_prom):
        self._promotion = new_prom
        if new_prom == '':
            self.dpos = set()
        else:
            self.dpos = {(-1, 0), (0, 1), (1, 0), (0, -1)}


class Gold(Piece):
    
    def __init__(self, name, pos, side, captured=False):
        Piece.__init__(self, name, pos, side, captured)
        self.sdpos = set()
        if self.side == 'b':
            self.dpos = {(-1, 0), (-1, 1), (0, 1), (1, 0), (0, -1), (-1, -1)}
        else:
            self.dpos = {(-1, 0), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)}
    
    @property
    def side(self):
        return self._side
    
    @side.setter
    def side(self, new_side):
        self._side = new_side
        if self.side == 'b':
            self.dpos = {(-1, 0), (-1, 1), (0, 1), (1, 0), (0, -1), (-1, -1)}
        else:
            self.dpos = {(-1, 0), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)}


class Silver(Piece):
    
    def __init__(self, name, pos, side, captured=False):
        Piece.__init__(self, name, pos, side, captured)
        self.sdpos = set()
        if self.side == 'b':
            self.dpos = {(-1, 0), (-1, 1), (1, 1), (1, -1), (-1, -1)}
        else:
            self.dpos = {(-1, 1), (1, 1), (1, 0), (1, -1), (-1, -1)}
    
    @property
    def side(self):
        return self._side
    
    @side.setter
    def side(self, new_side):
        self._side = new_side
        if self.side == 'b':
            self.dpos = {(-1, 0), (-1, 1), (1, 1), (1, -1), (-1, -1)}
        else:
            self.dpos = {(-1, 1), (1, 1), (1, 0), (1, -1), (-1, -1)}
    
    @property
    def promotion(self):
        return self._promotion
    
    @promotion.setter
    def promotion(self, new_prom):
        self._promotion = new_prom
        if new_prom == '+':
            if self.side == 'b':
                self.dpos = {(-1, 0), (-1, 1), (0, 1), (1, 0), (0, -1), (-1, -1)}
            else:
                self.dpos = {(-1, 0), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)}


class Knight(Piece):
    
    def __init__(self, name, pos, side, captured=False):
        Piece.__init__(self, name, pos, side, captured)
        self.sdpos = set()
        if self.side == 'b':
            self.dpos = {(-2, -1), (-2, 1)}
        else:
            self.dpos = {(2, -1), (2, 1)}
    
    @property
    def side(self):
        return self._side
    
    @side.setter
    def side(self, new_side):
        self._side = new_side
        if self.side == 'b':
            self.dpos = {(-2, -1), (-2, 1)}
        else:
            self.dpos = {(2, -1), (2, 1)}
    
    @property
    def promotion(self):
        return self._promotion
    
    @promotion.setter
    def promotion(self, new_prom):
        self._promotion = new_prom
        if new_prom == '+':
            if self.side == 'b':
                self.dpos = {(-1, 0), (-1, 1), (0, 1), (1, 0), (0, -1), (-1, -1)}
            else:
                self.dpos = {(-1, 0), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)}


class Lance(Piece):
    
    def __init__(self, name, pos, side, captured=False):
        Piece.__init__(self, name, pos, side, captured)
        if self.side == 'b':
            self.sdpos = {(-1, 0)}
        else:
            self.sdpos = {(1, 0)}
        self.dpos = set()
    
    @property
    def side(self):
        return self._side
    
    @side.setter
    def side(self, new_side):
        self._side = new_side
        if self.side == 'b':
            self.sdpos = {(-1, 0)}
            self.dpos = set()
        else:
            self.sdpos = {(1, 0)}
            self.dpos = set()
    
    @property
    def promotion(self):
        return self._promotion
    
    @promotion.setter
    def promotion(self, new_prom):
        self._promotion = new_prom
        if new_prom == '+':
            if self.side == 'b':
                self.sdpos = set()
                self.dpos = {(-1, 0), (-1, 1), (0, 1), (1, 0), (0, -1), (-1, -1)}
            else:
                self.sdpos = set()
                self.dpos = {(-1, 0), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)}


class Pawn(Piece):
    
    def __init__(self, name, pos, side, captured=False):
        Piece.__init__(self, name, pos, side, captured)
        self.sdpos = set()
        if self.side == 'b':
            self.dpos = {(-1, 0)}
        else:
            self.dpos = {(1, 0)}
    
    @property
    def side(self):
        return self._side
    
    @side.setter
    def side(self, new_side):
        self._side = new_side
        if self.side == 'b':
            self.dpos = {(-1, 0)}
        else:
            self.dpos = {(1, 0)}
    
    @property
    def promotion(self):
        return self._promotion
    
    @promotion.setter
    def promotion(self, new_prom):
        self._promotion = new_prom
        if new_prom == '+':
            if self.side == 'b':
                self.dpos = {(-1, 0), (-1, 1), (0, 1), (1, 0), (0, -1), (-1, -1)}
            else:
                self.dpos = {(-1, 0), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)}
    

#______________________________________________________________________________

class Game_state:
    """
    Useful attributes :
        - es_pieces
        - pieces
        - board
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
        
        self.es_pieces['w'] = {'l1': Lance('l1', (0, 0), 'w'), 
                             'n1': Knight('n1', (0, 1), 'w'), 
                             's1': Silver('s1', (0, 2), 'w'),
                             'g1': Gold('g1', (0, 3), 'w'),
                             'k': King('k', (0, 4), 'w'),
                             'g2': Gold('g2', (0, 5), 'w'), 
                             's2': Silver('s2', (0, 6), 'w'), 
                             'n2': Knight('n2', (0, 7), 'w'),
                             'l2': Lance('l2', (0, 8), 'w'), 
                             'r': Rook('r', (1, 1), 'w'),
                             'b': Bishop('b', (1, 7), 'w'),
                             'p2': Pawn('p2', (2, 1), 'w'), 
                             'p1': Pawn('p1', (2, 0), 'w'),
                             'p3': Pawn('p3', (2, 2), 'w'), 
                             'p4': Pawn('p4', (2, 3), 'w'),
                             'p5': Pawn('p5', (2, 4), 'w'),
                             'p6': Pawn('p6', (2, 5), 'w'),
                             'p7': Pawn('p7', (2, 6), 'w'),
                             'p8': Pawn('p8', (2, 7), 'w'),
                             'p9': Pawn('p9', (2, 8), 'w')}
        
        self.es_pieces['b'] = {'P1': Pawn('P1', (6, 0), 'b'),
                             'P2': Pawn('P2', (6, 1), 'b'),
                             'P3': Pawn('P3', (6, 2), 'b'),
                             'P4': Pawn('P4', (6, 3), 'b'),
                             'P5': Pawn('P5', (6, 4), 'b'),
                             'P6': Pawn('P6', (6, 5), 'b'),
                             'P7': Pawn('P7', (6, 6), 'b'),
                             'P8': Pawn('P8', (6, 7), 'b'),
                             'P9': Pawn('P9', (6, 8), 'b'),
                             'B': Bishop('B', (7, 1), 'b'),
                             'R': Rook('R', (7, 7), 'b'),
                             'L1': Lance('L1', (8, 0),'b'),
                             'N1': Knight('N1', (8, 1), 'b'),
                             'S1': Silver('S1', (8, 2), 'b'),
                             'G1': Gold('G1', (8, 3), 'b'),
                             'K': King('K', (8, 4), 'b'),
                             'G2': Gold('G2', (8, 5), 'b'),
                             'S2': Silver('S2', (8, 6), 'b'),
                             'N2': Knight('N2', (8, 7), 'b'),
                             'L2': Lance('L2', (8, 8), 'b')}
        
        self.pieces = {**self.es_pieces['w'], **self.es_pieces['b']}
        
        # Set of the captured pieces for each side.
        self.piecesC = {'b': set(), 'w': set()}

        # Set of the USI names of the captured pieces of which
        # the legal drop moves have already been added.
        self.visited_piecesC = set()
        
        self.board = self.init_board()
        
        # Each side's 'pieces reaching it'.
        self.PRI = {}
        
        # Matrix containing at each square sq of the board the black pieces on the board
        # which can reach sq. Captured pieces are excluded.
        # Even if there is a black piece at sq, we consider that sq can still be reached by black pieces
        # though actually not really true. We choose this convention only for the algorithm to be more
        # efficient.
        # 
        # In PRI, all pseudo-legal moves are allowed because at each move,
        # we only look for enemy pieces which can move without protecting king
        # if they can check the king.
        self.PRI['b'] = self.init_PRI0()
        self.PRI['w'] = self.init_PRI0()
        self.init_PRI1()
        self.init_PRI2()
        
        self.playing_side = 'b'
        
        # List of all legal moves in USI format that playing_side can do.
        self.legal_moves = []
        
        self.update_legal_moves()
        
        # The number of moves the players did during the game.
        self.nb_moves = 0
        # The total number of legal moves generated in 'legal_moves' during the game.
        self.nb_nodes = len(self.legal_moves)
    
    # Return True if the state describes a finished game.
    def finished(self):
        return self.legal_moves == []
    
    def black_wins(self):
        return self.finished() and (self.playing_side == 'w')
    
    def init_board(self):
        result = np.empty((9, 9), dtype=object)
        for piece in self.pieces.values():
            result[piece.x, piece.y] = piece
        return result
    
    # Initialize 'PRI' with empty sets.
    def init_PRI0(self):
        result = np.empty((9, 9), dtype=object)
        for i in range(9):
            for j in range(9):
                result[i, j] = set()
        return result
    
    # Add all the pieces to PRI for one square moves.
    def init_PRI1(self):
        for piece in self.pieces.values():
            for dx, dy in piece.dpos:
                x, y = piece.x + dx, piece.y + dy
                if not (0 <= x <= 8 and 0 <= y <= 8):
                    continue
                self.PRI[piece.side][x, y].add(piece)
    
    # Add all the pieces to PRI for sliding moves.
    def init_PRI2(self):
        for piece in self.pieces.values():
            for dx, dy in piece.sdpos:
                x, y = piece.x + dx, piece.y + dy
                while 0 <= x <= 8 and 0 <= y <= 8:
                    self.PRI[piece.side][x, y].add(piece)
                    if self.board[x, y] != None:
                        break
                    x, y = x + dx, y + dy
    
    # Update 'legal_moves' by adding into it all legal moves of 'playing_side'.
    # It must be called after that playing_side has been updated because the
    # legal moves will be those of the next player who will choose among them.
    def update_legal_moves(self):
        self.legal_moves = []
        self.visited_piecesC = set()
        
        king = self.get_king(self.playing_side)
        side2 = other_side(self.playing_side)
        
        # If the king is threaten by at least 2 enemy pieces,
        # then we have no choice but to move the king to a safe square.
        if len(self.PRI[side2][king.x, king.y]) >= 2:
            self.add_legal_moves(king)
        
        # If the king is theaten by just one enemy piece,
        # then we move the king or we capture the threatening piece or we block its path.
        elif len(self.PRI[side2][king.x, king.y]) == 1:
            self.add_legal_moves(king)
            # Threatening piece.
            thr_piece = next(iter(self.PRI[side2][king.x, king.y]))
            self.add_legal_moves_saving_king(king, thr_piece)
        
        # If the king is not threaten, then we allow all pseudo-legal moves
        # except those which remove the cover from the king.
        else:
            for piece in self.es_pieces[self.playing_side].values():
                self.add_legal_moves(piece)
    
    # Add all the moves which save the king from the threatening_piece 'thr_piece' in 'legal_moves',
    # that is the moves which capture the knight if thr_piece is a knight, or the moves which
    # intercept thr_piece's path to the king to cover the latter (including capturing thr_piece).
    def add_legal_moves_saving_king(self, king, thr_piece):
        # If thr_piece is just next to the king or if thr_piece is a knight,
        # we just look for pieces not covering the king which can capture thr_piece.
        next2king = abs(thr_piece.x - king.x) <= 1 and abs(thr_piece.y - king.y) <= 1
        if next2king or type(thr_piece) == Knight:
            for piece in self.PRI[self.playing_side][thr_piece.x, thr_piece.y]:
                if type(piece) != King and not self.covering_king(piece, king):
                    self.add_one_legal_move(piece, thr_piece.pos)
            return None
        
        # Otherwise, we look for pieces not covering the king
        # which can intercept thr_piece's path to the king.
        self.add_legal_moves_btw(king.pos, thr_piece.pos, king)
    
    # Add legal moves between 'sq1' and 'sq2' in legal_moves.
    # The king must be given.
    def add_legal_moves_btw(self, sq1, sq2, king):
        x1, y1 = sq1
        x2, y2 = sq2
        dx = np.sign(x2 - x1)
        dy = np.sign(y2 - y1)
        
        # For pieces on the board.
        self.add_legal_board_moves_btw(sq1, sq2, dx, dy, king)
        
        # For captured pieces ready to be dropped.
        for piece in self.piecesC[self.playing_side]:
            self.add_legal_drop_moves_btw(piece, sq1, sq2, dx, dy)
    
    # Add legal moves on the board between 'sq1' and 'sq2' in legal_moves.
    # The steps 'dx' and 'dy' and the king must be given.
    def add_legal_board_moves_btw(self, sq1, sq2, dx, dy, king):
        x, y = sq1
        while (x, y) != sq2:
            x += dx
            y += dy
            for piece in self.PRI[self.playing_side][x, y]:
                if type(piece) != King and not self.covering_king(piece, king):
                    self.add_one_legal_move(piece, (x, y))
    
    # Add the move from piece.pos to 'sq2move' (square to move) handling promotion.
    def add_one_legal_move(self, piece, sq2move):
        move = usi_move(piece.pos, sq2move)
        if piece.must_promote(sq2move):
            self.legal_moves.append(move + '+')
        elif piece.can_promote(sq2move):
            self.legal_moves.append(move)
            self.legal_moves.append(move + '+')
        else:
            self.legal_moves.append(move)
    
    # Update PRI before moving the piece for the moved piece as well as
    # for the captured piece 'cap_piece' if there is (set to None otherwise).
    def update_PRI1(self, moved_piece, cap_piece):
        # Update for moved_piece.
        self.update_piece_from_PRI(moved_piece, moved_piece.pos, moved_piece.dpos, add=False)
        self.update_sliding_piece_from_PRI(moved_piece, moved_piece.pos, moved_piece.sdpos, add=False)
        # Update for cap_piece (captured piece).
        if cap_piece:
            self.update_piece_from_PRI(cap_piece, cap_piece.pos, cap_piece.dpos, add=False)
            self.update_sliding_piece_from_PRI(cap_piece, cap_piece.pos, cap_piece.sdpos, add=False)
    
    # Update PRI after moving 'piece' from 'sq1'.
    # If it's a drop, sq1 should be set to None.
    def update_PRI2(self, piece, sq1=None):
        # Update for the moved piece.
        self.update_piece_from_PRI(piece, piece.pos, piece.dpos, add=True)
        self.update_sliding_piece_from_PRI(piece, piece.pos, piece.sdpos, add=True)
        if sq1:
            # Update for the rooks/lances/bishops reaching sq1.
            self.update_rlb_from_PRI(sq1, piece.pos, add=True)
            # Update for the rooks/lances/bishops reaching 'piece.pos'.
            self.update_rlb_from_PRI(sq1, piece.pos, add=False)
        else:
            self.update_rlb_from_PRI_drop(piece.pos)
    
    # Update rooks, lances and bishops from PRI reaching 'sq1' or 'sq2' by adding or removing them
    # (according to the value of the boolean 'add'). (the piece moved from sq1 to sq2)
    def update_rlb_from_PRI(self, sq1, sq2, add):
        x1, y1 = sq1
        x2, y2 = sq2
        x, y = sq1 if add else sq2
        for side in ['b', 'w']:
            for piece in self.PRI[side][x, y]:
                if not piece.sdpos:
                    continue
                dx = np.sign(x - piece.x)
                dy = np.sign(y - piece.y)
                # It is possible that (dx, dy) is in 'piece.dpos' if 'piece' is promoted.
                if (dx, dy) not in piece.sdpos:
                    continue
                # If the piece is aligned with the moved piece before and after the move
                # and that they got closer, we don't do addings to PRI but just removals from PRI
                # and we stop at 'sq1'.
                if self.got_closer(x1, y1, x2, y2, piece.x, piece.y):
                    if not add:
                        self.update_sliding_piece_from_PRI(piece, sq2, {(dx, dy)}, add, sq_limit=sq1)
                # Same as the case before but when the two pieces got more distant.
                # Then we don't do removals from PRI but just addings to PRI.
                elif self.got_more_distant(x1, y1, x2, y2, piece.x, piece.y):
                    if add:
                        self.update_sliding_piece_from_PRI(piece, sq1, {(dx, dy)}, add)
                else:
                    self.update_sliding_piece_from_PRI(piece, (x, y), {(dx, dy)}, add)
    
    # Remove rooks, lances and bishops reaching 'sq2' from PRI for the case of a drop on 'sq2'.
    def update_rlb_from_PRI_drop(self, sq2):
        x, y = sq2
        for side in ['b', 'w']:
            for piece in self.PRI[side][x, y]:
                if type(piece) not in {Rook, Lance, Bishop}:
                    continue
                dx = np.sign(x - piece.x)
                dy = np.sign(y - piece.y)
                self.update_sliding_piece_from_PRI(piece, sq2, {(dx, dy)}, add=False)
    
    # Used in 'update_rlb_from_PRI' method.
    def got_closer(self, x1, y1, x2, y2, xp, yp):
        if x1 == x2 == xp:
            return abs(yp - y2) < abs(yp - y1)
        elif y1 == y2 == yp:
            return abs(xp - x2) < abs(xp - x1)
        elif abs(x2 - x1) == abs(y2 - y1) and abs(xp - x2) == abs(yp - y2):
            return np.sign(xp - x2) == np.sign(x2 - x1) and np.sign(yp - y2) == np.sign(y2 - y1)
        return False
    
    # Used in 'update_rlb_from_PRI' method.
    def got_more_distant(self, x1, y1, x2, y2, xp, yp):
        if x1 == x2 == xp:
            return abs(yp - y2) > abs(yp - y1)
        elif y1 == y2 == yp:
            return abs(xp - x2) > abs(xp - x1)
        elif abs(x2 - x1) == abs(y2 - y1) and abs(xp - x2) == abs(yp - y2):
            return np.sign(xp - x1) == np.sign(x1 - x2) and np.sign(yp - y1) == np.sign(y1 - y2)
        return False
    
    def print_board(self):
        blank = ' '
        print(' ' + blank, end='')
        for i in range(9, 0, -1):
            print(' ' + str(i), end=blank)
        print()
        for i in range(9):
            print(chr(ord('a') + i), end=blank)
            for j in range(9):
                piece = self.board[i, j]
                if piece == None:
                    print(' .', end=blank)
                else:
                    blank2 = ' ' if len(piece.usi_name) == 1 else ''
                    print(blank2 + piece.usi_name, end=blank)
            if i < 9:
                print()
    
    def print_PRI(self):
        for x in range(9):
            print('[', end='')
            for y in range(9):
                print('{', end='')
                first = True
                for piece in self.PRI['b'][x, y] | self.PRI['w'][x, y]:
                    if not first:
                        print(', ', end='')
                    print(piece.usi_name, end='')
                    first = False
                print('}', end='')
                if y <= 7:
                    print(', ', end='')
            print(']')
    
    def print_pieces(self):
        for piece in self.pieces.values():
            print(piece)
    
    # Return the state after playing the move from this state.
    # If move is the empty string, a copy of the state is returned.
    def next_state(self, move):
        next_st = copy.deepcopy(self)
        if move:
            next_st.update(move)
        return next_st
    
    # Update the game's variables (board, pieces' attributes, pieces dictionaries, playing_side)
    # according to last_move (in USI protocol format).
    def update(self, last_move, AI=None):
        
        # Case of a drop.
        if last_move[1] == '*':
            dropped_piece = self.search_dropped_piece(last_move)
            
            dropped_piece.captured = False
            self.piecesC[self.playing_side].remove(dropped_piece)
            
            # Update piece's position and board.
            x, y = str2id(last_move[2:])
            dropped_piece.pos = (x, y)
            self.board[x, y] = dropped_piece
            
            self.update_PRI2(dropped_piece)
        
        # Case of a regular move.
        else:
            x1, y1 = str2id(last_move[:2])
            x2, y2 = str2id(last_move[2:4])
            piece1 = self.board[x1, y1]
            piece2 = self.board[x2, y2]
            
            self.update_PRI1(piece1, piece2)
            
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
                self.piecesC[self.playing_side].add(piece2)
                piece2.pos = None
                piece2.promotion = ''
            
            # Update board.
            self.board[x2, y2] = self.board[x1, y1]
            self.board[x1, y1] = None
            
            self.update_PRI2(piece1, (x1, y1))
        
        self.playing_side = other_side(self.playing_side)
        
        ###
        tic = time.perf_counter()
        ###
        self.update_legal_moves()
        ###
        if AI:
            AI.time_block += time.perf_counter() - tic
        ###
        
        self.nb_moves += 1
        self.nb_nodes += len(self.legal_moves)
    
    # Return the piece which was dropped.
    def search_dropped_piece(self, move):
        dropped_piece_name = move[0]
        for piece in self.piecesC[self.playing_side]:
            if piece.usi_name.upper() == dropped_piece_name.upper():
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
    
    # Move "piece" from its side's dictionary of pieces to the other side's dictionary.
    def change_piece_side(self, piece):
        side1_pieces = self.es_pieces[piece.side]
        side2_pieces = self.es_pieces[other_side(piece.side)]
        side2_pieces[piece.name] = side1_pieces[piece.name]
        side1_pieces.pop(piece.name)
    
    # Return the king of 'side'.
    def get_king(self, side):
        king_name = 'K' if side == 'b' else 'k'
        return self.pieces[king_name]

    # Return the set of the reachable squares by 'piece' with squares in usi format.
    # It includes squares occupied by an ally.
    def reachable_squares(self, piece):
        res = set()
        for x in range(9):
            for y in range(9):
                if piece in self.PRI[piece.side][x, y]:
                    res.add(id2str((x, y)))
        return res
    
    # Return True if the king can move.
    def king_can_move(self, king):
        for dx, dy in king.dpos:
            x, y = king.x + dx, king.y + dy
            if not (0 <= x <= 8 and 0 <= y <= 8):
                continue
            piece_xy = self.board[x, y]
            if (piece_xy == None or piece_xy.side != king.side) \
            and not self.PRI[other_side(king.side)][king.x, king.y]:
                return True
        return False
    
    # Return True if a non-promoted pawn of the same side as 'pawn' is on column y.
    def pawn_on_column(self, pawn, y):
        for x in range(9):
            piece = self.board[x, y]
            if type(piece) == Pawn and piece.side == pawn.side and piece.promotion == '':
                return True
        return False
    
    # Return True if dropping a pawn at (x, y) checkmate, which is illegal.
    def pawn_drop_checkmate(self, pawn, x, y):
        # Opponent's king.
        op_king = self.get_king(other_side(pawn.side))
        
        # We see if the pawn would be dropped in front of the opponent's king.
        dx_fw = -1 if op_king.side == 'b' else 1
        x_fw = op_king.x + dx_fw
        if (x, y) != (x_fw, op_king.y):
            return False
        
        # If the pawn could be captured.
        if self.PRI[other_side(pawn.side)][x, y]:
            return False
        
        # If the opponent's king can move.
        if self.king_can_move(op_king):
            return False
        
        return True
    
    # Add all legal moves of 'piece' in 'legal_moves'.
    # Must be used only if the king is not in check.
    def add_legal_moves(self, piece):
        if not piece.captured:
            self.add_legal_board_moves(piece)
        else:
            self.add_legal_drop_moves(piece)
    
    def add_legal_board_moves(self, piece):
        getattr(self, "add_legal_board_moves_" + type(piece).__name__.lower())(piece)
    
    def add_legal_board_moves_king(self, king):
        self.add_legal_board_moves_aux(king, king.dpos)

    def add_legal_board_moves_rook(self, rook):
        # If the rook is covering its king, it must keep the cover by staying in the same direction.
        stay_dpos = self.keep_cover(rook)
        sdpos = rook.sdpos
        if stay_dpos:
            sdpos = stay_dpos & sdpos
        self.add_legal_board_slide_moves(rook, sdpos)
        if rook.promotion == '+':
            dpos = rook.dpos
            if stay_dpos:
                dpos = stay_dpos & dpos
            self.add_legal_board_moves_aux(rook, dpos)
    
    def add_legal_board_moves_bishop(self, bishop):
        stay_dpos = self.keep_cover(bishop)
        sdpos = bishop.sdpos
        if stay_dpos:
            sdpos = stay_dpos & sdpos
        self.add_legal_board_slide_moves(bishop, sdpos)
        if bishop.promotion == '+':
            dpos = bishop.dpos
            if stay_dpos:
                dpos = stay_dpos & dpos
            self.add_legal_board_moves_aux(bishop, dpos)
    
    def add_legal_board_moves_gold(self, gold):
        stay_dpos = self.keep_cover(gold)
        dpos = gold.dpos
        if stay_dpos:
            dpos = stay_dpos & dpos
        self.add_legal_board_moves_aux(gold, dpos)
    
    def add_legal_board_moves_silver(self, silver):
        stay_dpos = self.keep_cover(silver)
        dpos = silver.dpos
        if stay_dpos:
            dpos = stay_dpos & dpos
        self.add_legal_board_moves_aux(silver, dpos)
    
    def add_legal_board_moves_knight(self, knight):
        # If the knight is covering its king, it can't move.
        if self.keep_cover(knight):
            return None
        self.add_legal_board_moves_aux(knight, knight.dpos)
    
    def add_legal_board_moves_lance(self, lance):
        stay_dpos = self.keep_cover(lance)
        if lance.promotion == '':
            sdpos = lance.sdpos
            if stay_dpos:
                sdpos = stay_dpos & sdpos
            self.add_legal_board_slide_moves(lance, sdpos)
        else:
            dpos = lance.dpos
            if stay_dpos:
                dpos = stay_dpos & dpos
            self.add_legal_board_moves_aux(lance, dpos)
    
    def add_legal_board_moves_pawn(self, pawn):
        stay_dpos = self.keep_cover(pawn)
        dpos = pawn.dpos
        if stay_dpos:
            dpos = stay_dpos & dpos
        self.add_legal_board_moves_aux(pawn, dpos)
    
    # Add one square legal moves of 'piece' on the board in 'legal_moves'.
    # Must be used only if the king is not in check.
    # 
    # dpos is the set of the directions (xdir, ydir) allowed by the piece.
    # xdir and ydir can take the values +/-1 if the piece can move by
    # 1 square along the given direction, and 0 for 0 square.
    # dpos must take into account whether 'self' is covering the king.
    # If it is, then dpos must only contain directions that keep the king covered.
    # 
    # This also can be used for knights given the correct dpos (+/-2, +/-1).
    def add_legal_board_moves_aux(self, piece, dpos):
        for dx, dy in dpos:
            x, y = piece.x + dx, piece.y + dy
            if not (0 <= x <= 8 and 0 <= y <= 8):
                continue
            piece_xy = self.board[x, y]
            # If there isn't any piece at (x, y) or if there is an opponent's piece there,
            # the piece can move towards this square.
            if not piece_xy or piece_xy.side != piece.side:
                # If 'self' is a king and that moving to (x, y) would put it in check,
                # then we skip (x, y).
                if type(piece) == King and self.PRI[other_side(piece.side)][x, y]:
                    continue
                self.add_one_legal_move(piece, (x, y))
    
    # Same as 'add_legal_moves' method but for sliding pieces with multiple squares along
    # one direction.
    def add_legal_board_slide_moves(self, piece, dpos):
        for dx, dy in dpos:
            x, y = piece.x + dx, piece.y + dy
            while 0 <= x <= 8 and 0 <= y <= 8:
                piece_xy = self.board[x, y]
                # If there isn't any piece at (x, y) or if there is an opponent's piece there,
                # the piece can move towards this square.
                if not piece_xy or piece_xy.side != piece.side:
                    self.add_one_legal_move(piece, (x, y))
                    # If it's a capturing move, the piece can't move farther so we stop.
                    if piece_xy and piece_xy.side != piece.side:
                        break
                else:
                    break
                x, y = x + dx, y + dy
    
    def add_legal_drop_moves(self, piece):
        if piece.usi_name in self.visited_piecesC:
            return None
        self.visited_piecesC.add(piece.usi_name)
        name = type(piece).__name__.lower()
        if name in ['knight', 'lance', 'pawn']:
            getattr(self, "add_legal_drop_moves_" + name)(piece)
        else:
            self.add_legal_drop_moves_aux(piece)
    
    def add_legal_drop_moves_knight(self, knight):
        xa, xb = (2, 8) if knight.side == 'b' else (0, 6)
        self.add_legal_drop_moves_aux(knight, xa, xb)
    
    def add_legal_drop_moves_lance(self, lance):
        xa, xb = (1, 8) if lance.side == 'b' else (0, 7)
        self.add_legal_drop_moves_aux(lance, xa, xb)
    
    def add_legal_drop_moves_pawn(self, pawn):
        xa, xb = (1, 8) if pawn.side == 'b' else (0, 7)
        for y in range(9):
            if self.pawn_on_column(pawn, y):
                continue
            for x in range(xa, xb+1):
                if not self.board[x, y] and not self.pawn_drop_checkmate(pawn, x, y):
                    self.legal_moves.append(usi_drop(pawn, (x, y)))
    
    # Add legal drop moves of 'piece' in 'legal_moves'.
    # Must be used only if the king is not in check.
    # 'piece' must not be a pawn.
    # All drops from rows xa to xb included are considered.
    def add_legal_drop_moves_aux(self, piece, xa=0, xb=8):
        for x in range(xa, xb+1):
            for y in range(9):
                if not self.board[x, y]:
                    self.legal_moves.append(usi_drop(piece, (x, y)))

    def add_legal_drop_moves_btw(self, piece, sq1, sq2, dx, dy):
        if piece.usi_name in self.visited_piecesC:
            return None
        self.visited_piecesC.add(piece.usi_name)
        name = type(piece).__name__.lower()
        if name in ['knight', 'lance', 'pawn']:
            getattr(self, "add_legal_drop_moves_btw_" + name)(piece, sq1, sq2, dx, dy)
        else:
            self.add_legal_drop_moves_btw_aux(piece, sq1, sq2, dx, dy)
    
    def add_legal_drop_moves_btw_lance(self, lance, sq1, sq2, dx, dy):
        if lance.side == 'b':
            xa, xb = 1, 8
        else:
            xa, xb = 0, 7
        self.add_legal_drop_moves_btw_aux(lance, sq1, sq2, dx, dy, xa, xb)
    
    def add_legal_drop_moves_btw_knight(self, knight, sq1, sq2, dx, dy):
        if knight.side == 'b':
            xa, xb = 2, 8
        else:
            xa, xb = 0, 6
        self.add_legal_drop_moves_btw_aux(knight, sq1, sq2, dx, dy, xa, xb)
    
    def add_legal_drop_moves_btw_pawn(self, pawn, sq1, sq2, dx, dy):
        x1, y1 = sq1
        x2, y2 = sq2
        
        if dy == 0 and self.pawn_on_column(pawn, y1):
            return None
        
        if pawn.side == 'b':
            xa, xb = 1, 8
        else:
            xa, xb = 0, 7
        
        x, y = x1, y1
        while (x, y) != sq2:
            x += dx
            y += dy
            if not (xa <= x <= xb):
                continue
            if dy != 0 and self.pawn_on_column(pawn, y):
                continue
            if self.board[x, y] == None and not self.pawn_drop_checkmate(pawn, x, y):
                self.legal_moves.append(usi_drop(pawn, (x, y)))
    
    # Same as 'add_legal_drop_moves' method but here we consider only drops between
    # sq1 and sq2 (and between rows xa and xb too). The steps 'dx' and 'dy' must be given.
    def add_legal_drop_moves_btw_aux(self, piece, sq1, sq2, dx, dy, xa=0, xb=8):
        x, y = sq1
        while (x, y) != sq2:
            x += dx
            y += dy
            if not (xa <= x <= xb):
                continue
            if self.board[x, y] == None:
                self.legal_moves.append(usi_drop(piece, (x, y)))
    
    # Add or remove (depends on the value of the boolean 'add') 'piece' to/from PRI
    # in the directions indicated in 'dpos' starting from the square 'sq'.
    def update_piece_from_PRI(self, piece, sq, dpos, add):
        for dx, dy in dpos:
            x, y = sq[0] + dx, sq[1] + dy
            if not (0 <= x <= 8 and 0 <= y <= 8):
                continue
            if add:
                self.PRI[piece.side][x, y].add(piece)
            else:
                self.PRI[piece.side][x, y].remove(piece)
    
    # Same as 'update_piece_from_PRI' method but for sliding pieces with multiple squares along
    # one direction. It is possible here to set a limit square 'sq_limit' at which we stop
    # (sq_limit is also treated). Set it to None if there isn't.
    def update_sliding_piece_from_PRI(self, piece, sq, dpos, add, sq_limit=None):
        for dx, dy in dpos:
            x, y = sq[0] + dx, sq[1] + dy
            # If we want to remove a sliding piece from PRI[x, y] but the piece
            # is not here, it means that the piece was blocked by another piece
            # at 'sq' which has just been captured. Thus, we don't remove.
            if not add and 0 <= x <= 8 and 0 <= y <= 8 and piece not in self.PRI[piece.side][x, y]:
                return None
            while 0 <= x <= 8 and 0 <= y <= 8:
                if add:
                    self.PRI[piece.side][x, y].add(piece)
                else:
                    self.PRI[piece.side][x, y].remove(piece)
                piece_xy = self.board[x, y]
                # If 'piece_xy' is the opponent's king, then we go through it because
                # for example, a king put in check by a rook must not step back.
                if self.board[x, y] != None and not (type(piece_xy) == King and piece_xy.side != piece.side):
                    break
                
                if sq_limit and (x, y) == sq_limit:
                    break
                
                x, y = x + dx, y + dy
    
    # Return True if 'piece' is covering its king from a rook, bishop, or lance attack.
    def covering_king(self, p, king):
        # We exit immediately if p is not aligned with king.
        if not (p.x == king.x or p.y == king.y or abs(p.x - king.x) == abs(p.y - king.y)):
            return False
        
        pri_xy = self.PRI[other_side(p.side)][p.x, p.y]
        ps = self.pieces
        
        if p.x == king.x:
            # We look for rooks attacking 'self'.
            rooks = {ps['r'], ps['R']} & pri_xy
            while rooks:
                rook = rooks.pop()
                # If rook, self, and king are aligned and if self is between rook and king.
                if rook.x == p.x and np.sign(king.y - p.y) == np.sign(p.y - rook.y):
                    if not found_piece_btw(self.board, p, king):
                        return True
            return False
        
        elif p.y == king.y:
            # We look for rooks and lances attacking 'self'.
            rls = {ps['r'], ps['R'], ps['l1'], ps['l2'], ps['L1'], ps['L2']} & pri_xy
            while rls:
                # Threatening piece.
                thr = rls.pop()
                # If thr, self, and king are aligned and if self is between thr and king.
                if thr.y == p.y and np.sign(king.x - p.x) == np.sign(p.x - thr.x):
                    if not found_piece_btw(self.board, p, king):
                        return True
            return False
        
        # If the king and self are on a same diagonal.
        else:
            bishops = {ps['b'], ps['B']} & pri_xy
            while bishops:
                bishop = bishops.pop()
                # If king, self, and bishop are aligned and if self is
                # between king and bishop on the diagonal.
                if abs(king.x - p.x) == abs(king.y - p.y) \
                and np.sign(king.x - p.x) == np.sign(p.x - bishop.x) \
                and np.sign(king.y - p.y) == np.sign(p.y - bishop.y):
                    if not found_piece_btw(self.board, p, king):
                        return True
            return False
    
    # Return the set of the directions along which 'piece' must stay in order to keep
    # covering its king (empty if not covering).
    def keep_cover(self, piece):
        king = self.get_king(piece.side)
        stay_dpos = set()
        if self.covering_king(piece, king):
            dx = np.sign(piece.x - king.x)
            dy = np.sign(piece.y - king.y)
            stay_dpos = {(dx, dy), (-dx, -dy)}
        return stay_dpos