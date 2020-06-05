import numpy as np
import torch

import shogi_engine as egn
import encoding
import snnAI
import randomAI
import jsonManager


class Game:
# A shogi game data structure.
# Has the history of the game and can give a position of the game
# and its label. Used for the training of the neural net.

    def __init__(self, idx, length, is_draw, code=None):
        # Index of the game in the histories and
        # the ls_checks arrays of the buffer.
        self.idx = idx
        self.length = length
        self.is_draw = is_draw
        # The unique code of the game.
        if code: self.code = code
        else: self.code = 0
    
    # Return the encoded game state of the position reached before
    # the move with the index 'move_idx' (in the game history) was played
    # and its label.
    def get_labelled_pos(self, c, move_idx, histories, ls_checks):
        ls = self.get_ls(c, move_idx, histories, ls_checks)
        a_id = histories[self.idx, move_idx]
        z_move = self.label_pos(move_idx, histories)
        return (
            encoding.encode_light_game_state(ls),
            a_id, z_move
        )
    
    # Return the ls array corresponding to the light state before
    # the move of index 'move_idx' is played.
    def get_ls(self, c, move_idx, histories, ls_checks):
        
        check_idx = move_idx // c.state_checkpoint_interval - 1
        if check_idx == -1:
            ls = egn.ls0.copy()
        else:
            ls = ls_checks[self.idx, check_idx].copy()

        check_move_idx = (check_idx + 1) * c.state_checkpoint_interval
        for i in range(check_move_idx, move_idx):
            move = encoding.decode_a_id(
                np.unravel_index(histories[self.idx, i], (139,9,9)),
                'b' if ls[-1,-2] == 0 else 'w'
            )
            egn.update_ls(ls, move)
        
        return ls
    
    # Return the evaluation value of move_idx:
    # 1 if the player of move_idx wins.
    # -1 if he loses.
    # 0 if it's a draw.
    # The other moves' evaluations won't change. It will be the same values
    # returned by the nnet.
    def label_pos(self, move_idx, histories):
        
        # The evaluation value for move_idx.
        z_move = 0

        # Compute z_move.
        if not self.is_draw:
            # The player who played move_idx is black if and only if (move_idx % 2) = 0.
            player_is_black = not (move_idx % 2)
            # The winner is black if and only if (self.length % 2) = 1.
            black_wins = self.length % 2
            player_wins = (player_is_black == black_wins)
            z_move = 1 if player_wins else -1

        return z_move
    
    # Return the unique code of the game. It's computed by multiplying
    # the game's length and the codes of the last moves.
    def compute_code(self, c, histories):
        code = self.length
        for i in range(1, c.game_code_num_moves + 1):
            if i > self.length:
                break
            code *= int(histories[self.idx, self.length-i])
        return code


class Buffer:
# A buffer containing games and their labels.
# It can sample batches.

    def __init__(self, c):
        self.c = c
        self.buffer = []
        self.histories = np.zeros(
            (c.window_size, c.max_moves),
            dtype=np.int16
        )
        # For each game, at each checkpoint of the game,
        # a LightGameState array is saved in this big array
        # to accelerate the get_position method.
        # Indices of the array: (game, state, row, column).
        # The index of the last game added in the array is
        # (game_total_count - 1) % window_size.
        # The initial state of a game is never saved.
        num_checks = c.max_moves // c.state_checkpoint_interval
        self.ls_checks = np.zeros(
            (c.window_size, num_checks, 11, 9),
            dtype=np.int8
        )
    
    def save(self):
        jsonManager.save(self.buffer, self.c.buffer_file_name)
        with open(self.c.histories_file_name, 'wb') as f:
            np.save(f, self.histories)
        with open(self.c.state_checkpoints_file_name, 'wb') as f:
            np.save(f, self.ls_checks)
    
    def load(self):
        self.buffer = jsonManager.load(self.c.buffer_file_name)
        with open(self.c.histories_file_name, 'rb') as f:
            self.histories = np.load(f)
        with open(self.c.state_checkpoints_file_name, 'rb') as f:
            self.ls_checks = np.load(f)
    
    # Save the game in the buffer without exceeding the window size.
    def save_game(self, game, s):
        self.buffer.append(game)
        if len(self.buffer) > self.c.window_size:
            self.buffer.pop(0)
        for i, a in enumerate(s.history):
            self.histories[game.idx, i] = np.ravel_multi_index(
                encoding.a_id(a), (139, 9, 9)
            )
        self.ls_checks[game.idx] = s.game_ls_checks
    
    # Sample a mini-batch of positions from the games of the buffer.
    def sample_batch(self, P):
        total_nb_pos = sum(g.length for g in self.buffer)

        games_sample = np.random.choice(
            self.buffer,
            size=self.c.batch_size,
            p=[g.length / total_nb_pos for g in self.buffer]
        )

        pos_sample = [(g, np.random.randint(g.length)) for g in games_sample]
        
        return [
            g.get_labelled_pos(self.c, move_idx, self.histories, self.ls_checks) \
            for (g, move_idx) in pos_sample
        ]


class GenGameState(egn.GameState):

    def __init__(self, c):
        super().__init__()
        self.ls = egn.ls0.copy()
        self.game_ls_checks = np.zeros(
            (c.max_moves // c.state_checkpoint_interval, 11, 9),
            dtype=np.int8
        )
    
    def update(self, move):
        egn.GameState.update(self, move)
        egn.update_ls(self.ls, move)


class GameGenerator:
# A generator of games. The generated games are saved in a buffer.
    

    def __init__(self, c, rm, nnet):
        self.c = c
        self.rm = rm
        self.ai1 = snnAI.SnnAI(c, nnet, add_noise=True)
        self.ai2 = randomAI.RandomAI()
        # The length of this set is the number of distinct games
        # generated in the current iteration.
        self.game_codes = set()
        # The list of the GenGameState objets representing the games
        # played in parallel.
        self.states1 = []
        self.states2 = []
    

    def generate_games(self, buffer):

        self.game_codes = set()
        # If we were in the mid of a game generation, we load game_codes.
        if self.rm.game_count > 0:
            self.load_game_codes(buffer)
        
        num_gs = min(self.c.parallel_games, self.c.games_per_iter - self.rm.game_count)
        self.states1 = [GenGameState(self.c) for _ in range(num_gs // 2)]
        self.states2 = [GenGameState(self.c) for _ in range(num_gs - num_gs // 2)]

        # Whether it's the ai1's turn in states1.
        ai1_plays1 = True

        self.rm.begin_game_tick()

        while self.rm.game_count < self.c.games_per_iter:

            player1 = self.ai1 if ai1_plays1 else self.ai2
            player2 = self.ai2 if ai1_plays1 else self.ai1

            if self.states1:
                moves1 = player1.best_moves(self.states1, self.rm)
                self.make_moves(buffer, self.states1, moves1)

            if self.states2:
                moves2 = player2.best_moves(self.states2, self.rm)
                self.make_moves(buffer, self.states2, moves2)

            ai1_plays1 = not ai1_plays1


    def make_moves(self, buffer, states, moves):
        # We iterate in reverse because of the pop on the states.
        i = len(states) - 1
        while i >= 0:
            self.make_one_move(buffer, states, moves, i)
            i -= 1
    

    # Make all the updates for one state after one move.
    def make_one_move(self, buffer, states, moves, i):

        gs = states[i]
        gs.update(moves[i])

        # Make a checkpoint of ls.
        if gs.nb_moves % self.c.state_checkpoint_interval == 0 \
        and gs.nb_moves > 0:
            check_idx = gs.nb_moves // self.c.state_checkpoint_interval - 1
            gs.game_ls_checks[check_idx] = gs.ls

        # Draw if max_moves reached.
        is_draw = (not gs.finished() and gs.nb_moves == self.c.max_moves)
        if is_draw: self.rm.draw()

        if gs.finished() or is_draw:

            # Save the finished game.
            game = Game(
                self.rm.game_total_count % self.c.window_size,
                len(gs.history),
                is_draw
            )
            buffer.save_game(game, gs)
            game.code = game.compute_code(self.c, buffer.histories)
            self.game_codes.add(game.code)

            # Update runManager.
            self.rm.end_game(buffer, gs.nb_moves, len(self.game_codes))
            self.rm.begin_game_tick()
            
            # Update states.
            if self.c.games_per_iter - self.rm.game_count < self.c.parallel_games:
                states.pop(i)
            else:
                states[i] = GenGameState(self.c)
    

    # Load the set game_codes.
    def load_game_codes(self, buffer):
        last_game_idx = self.rm.game_count % self.c.window_size - 1
        for i in range(self.rm.game_count):
            self.game_codes.add(buffer.buffer[last_game_idx - i].code)