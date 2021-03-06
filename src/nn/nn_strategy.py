from board import Board
from random import random, choice
from strategy import Strategy
import time

def no_exploration(g):
    return 0

class NnStrategy(Strategy):

    def __init__(self, network, mirroring=False, get_exploration_factor=no_exploration):
        self.discount_factor = 0.9
        self.get_exploration_factor = get_exploration_factor
        self.current_game_moves = []
        self.current_game_think_time = 0.0
        self.network = network
        self.current_batch_board_states = []
        self.current_batch_rewards = []
        self.batch_size = 100
        self.mirroring = mirroring

    def move(self, game, player_id):
        start_time = time.time()
        move_scores = {}

        valid_moves = game.board.get_valid_moves()

        if random() < self.get_exploration_factor(game.game_number):
            move_to_play = choice(valid_moves)

            game_copy = game.clone()
            game_copy.move(move_to_play, player_id)
            board_state = self._build_board_state(game_copy, player_id)

        else:
            for move in valid_moves:
                game_copy = game.clone()
                game_copy.move(move, player_id)
                board_state = self._build_board_state(game_copy, player_id)
                score = self.network.eval_position(board_state)
                move_scores[move] = (score, board_state)

            move_to_play = max(list(move_scores.keys()), key=lambda k: move_scores.get(k)[0])
            board_state = move_scores[move_to_play][1]

        self.current_game_moves.append((board_state, move_to_play))
        self.current_game_think_time += (time.time() - start_time)

        return move_to_play

    def game_over(self, reward, training=False):
        for board_state, move in reversed(self.current_game_moves):
            self.current_batch_board_states.append(board_state)
            self.current_batch_rewards.append(reward)
            if self.mirroring:
                self.current_batch_board_states.append(self._reverse_board_state(board_state))
                self.current_batch_rewards.append(reward)
            reward *= self.discount_factor

        if len(self.current_batch_rewards) >= self.batch_size:
            self.network.update(self.current_batch_board_states, self.current_batch_rewards)
            self.current_batch_board_states = []
            self.current_batch_rewards = []

        if training:
            self.network.training_game_over(len(self.current_game_moves), self.current_game_think_time)

        self.current_game_think_time = 0.0
        self.current_game_moves = []

    def save(self):
        self.network.save()

    def get_name(self):
        return self.network.get_name()

    def get_total_games_trained(self):
        return self.network.total_games_trained

    def get_average_game_think_time(self):
        return self.network.average_think_time

    def get_total_number_of_moves(self):
        return self.network.total_number_of_moves

    def _build_board_state(self, game, player_id):
        bs = list([list([1 if v == player_id else 0 if v == Board.EMPTY_CELL else -1 for v in row]) for row in game.board.board])

        return bs

    def _reverse_board_state(self, board_state):
        return [row[::-1] for row in board_state]
