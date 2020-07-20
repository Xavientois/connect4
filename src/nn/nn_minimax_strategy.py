from board import Board
from strategy import Strategy
import math
import time


class NnMiniMaxStrategy(Strategy):

    def __init__(self, network, lookahead_limit, player_id, opponent_id, alpha_beta_prune=False):
        self.network = network
        self.lookahead_limit = lookahead_limit
        self.player_id = player_id
        self.opponent_id = opponent_id
        self.alpha_beta_prune = alpha_beta_prune
        self.total_time = 0
        self.total_moves = 0

    def move(self, game, player_id):
        def fn_eval(game, alpha, beta):
            return self._value_with_lookahead(game, player_id, self.lookahead_limit, alpha, beta)

        t0 = time.time()
        move = self._find_best_move_and_score_for_player(game, player_id, fn_eval, -math.inf, math.inf)[0]
        self.total_time += time.time() - t0
        self.total_moves += 1
        return move

    def average_move_time(self):
        return self.total_time / self.total_moves

    def _value_with_lookahead(self, game, curr_player, lookaheads_remaining, alpha, beta):
        if lookaheads_remaining == 0:
            board_state = self._build_board_state(game, curr_player)
            score = self.network.eval_position(board_state)
        else:
            other_player_id = self._get_other_player(curr_player)

            def fn_eval(game, alpha, beta):
                return self._value_with_lookahead(game, other_player_id, lookaheads_remaining-1, alpha, beta)
            score = -self._find_best_move_and_score_for_player(
                    game, other_player_id, fn_eval, alpha, beta)[1]

        return score

    def _find_best_move_and_score_for_player(self, game, curr_player, fn_eval, alpha, beta):
        valid_moves = game.board.get_valid_moves()
        move_scores = {}
        for move in valid_moves:
            game_copy = game.clone()
            game_copy.move(move, curr_player)
            if game_copy.finished:
                # This tricky line sets the score to be +inf if the curr_player (could be player_id or opponent) wins
                # Score is 0 on a tie and -inf if the curr_player loses
                move_scores[move] = math.inf if game_copy.winner == curr_player else 0 if not game_copy.winner else -math.inf
            else:
                # Because we are using the Nagamax variation of minimax, we must negate and flip our alpha & beta
                # values every time we recurse to look ahead one more.
                # see more here: https://en.wikipedia.org/wiki/Negamax
                move_scores[move] = fn_eval(game_copy, -beta, -alpha)

            if alpha < move_scores[move]:
                alpha = move_scores[move]
            
            if self.alpha_beta_prune and alpha >= beta:
                break

        best_move = max(list(move_scores.keys()), key=lambda k: move_scores.get(k))
        return best_move, move_scores[best_move]

    def _get_other_player(self, curr_player):
        return self.opponent_id if curr_player == self.player_id else self.player_id

    def game_over(self, reward, training=False):
        pass

    def save(self):
        pass

    def get_name(self):
        return self.network.get_name() + 'mm'

    def _build_board_state(self, game, player_id):
        bs = list([list([1 if v == player_id else 0 if v ==
                         Board.EMPTY_CELL else -1 for v in row]) for row in game.board.board])

        return bs
