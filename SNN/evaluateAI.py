import datetime
import time

import shogi_engine as egn
import randomAI
import snnAI
import nnet



class Evaluator:


    def __init__(self, evaluator_ai, num_games, max_moves):

        if evaluator_ai == 'random':
            self.evaluator_ai = randomAI.RandomAI()
        else:
            self.evaluator_ai = evaluator_ai
        self.num_games = num_games
        self.max_moves = max_moves

        self.num_wins = 0
        self.num_draws = 0
        self.num_moves = 0
        self.duration = 0

        self.win_perc = -1
        self.win_black_perc = -1
        self.win_white_perc = -1
        self.draw_perc = -1
        self.win_no_draws_perc = -1
        self.avg_game_length = -1
        self.duration = -1


    def reset(self):
        self.num_wins = 0
        self.num_wins_black = 0
        self.num_wins_white = 0
        self.num_draws = 0
        self.num_moves = 0
        self.duration = 0


    # Evaluate the given AI with another AI by playing num_games games.
    def evaluate_ai(self, evaluated_ai):

        start_time = time.time()

        self.reset()

        # The games are separated in half.
        # The first part is the games for which the evaluated AI is black
        # and the second one is those for which the evaluator is black.
        states1 = [egn.GameState() for _ in range(self.num_games // 2)]
        states2 = [egn.GameState() for _ in range(self.num_games - len(states1))]
        
        # Whether it's the evaluator's turn in states1.
        evaluator_plays1 = False

        while states1 or states2:
            
            player1 = self.evaluator_ai if evaluator_plays1 else evaluated_ai
            player2 = evaluated_ai if evaluator_plays1 else self.evaluator_ai

            if states1:
                moves1 = player1.best_moves(states1)
                self.update_states(states1, moves1, evaluator_plays1)

            if states2:
                moves2 = player2.best_moves(states2)
                self.update_states(states2, moves2, not evaluator_plays1)

            evaluator_plays1 = not evaluator_plays1
        
        # Compute results
        self.compute_results()
        self.duration = time.time() - start_time
    

    # Update states list according to the moves and return the results
    # of the updates.
    # evaluator_plays indicates whether the player playing the moves
    # is the evaluator AI.
    def update_states(self, states, moves, evaluator_plays):

        i = len(states) - 1
        while i >= 0:

            s = states[i]
            s.update(moves[i])

            if s.finished():
                self.num_moves += s.nb_moves
                if not evaluator_plays:
                    self.num_wins += 1
                    if s.playing_side == 'w':
                        self.num_wins_black += 1
                    else:
                        self.num_wins_white += 1
                states.pop(i)

            elif s.nb_moves == self.max_moves:
                self.num_draws += 1
                states.pop(i)

            i -= 1
    

    def compute_results(self):
        self.win_perc = 100 * self.num_wins / self.num_games
        self.win_black_perc = 100 * self.num_wins_black / self.num_games
        self.win_white_perc = 100 * self.num_wins_white / self.num_games
        self.draw_perc = 100 * self.num_draws / self.num_games
        self.win_no_draws_perc = -1
        self.avg_game_length = -1
        if self.num_games != self.num_draws:
            self.win_no_draws_perc = 100 * self.num_wins / (self.num_games - self.num_draws)
            self.avg_game_length = self.num_moves / (self.num_games - self.num_draws)



if __name__ == '__main__':
    
    evaluated_ai = randomAI.RandomAI()

    num_games = 1000
    
    ev = Evaluator('random', num_games, 512)
    ev.evaluate_ai(evaluated_ai)

    str_eval_duration = str(datetime.timedelta(seconds=round(ev.duration)))

    print(
        f"wins: {round(ev.win_perc, 2)} % | "
        f"draws: {round(ev.draw_perc, 2)} % | "
        f"wins no draws: {round(ev.win_no_draws_perc, 2)} %"

        f"\nwins as black: {round(ev.win_black_perc, 2)} % | "
        f"wins as white: {round(ev.win_white_perc, 2)} %"

        f"\naverage length: {round(ev.avg_game_length)} | "
        f"eval duration: {str_eval_duration}"
    )