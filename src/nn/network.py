import numpy as np
from keras.models import Sequential, load_model
from abc import ABC, abstractmethod
import os
import json

saved_model_dir = './saved_models'
saved_data_dir = './saved_data'

class Network(ABC):
    def __init__(self, model):
        self.model = model
        self.total_games_trained = 0
        self.total_number_of_moves = 0
        self.average_think_time = 0.0
        self.load()

    def eval_position(self, board_state):
        input = self._board_states_to_inputs([board_state])
        result = self.model.predict(input)
        return result

    def update(self, board_states, rewards):
        inputs = self._board_states_to_inputs(board_states)
        outputs = np.array(rewards)
        self.model.train_on_batch(inputs, outputs)

    def load(self):
        model_path = os.path.join(saved_model_dir, self.get_save_file())
        data_path = os.path.join(saved_data_dir, self.get_save_file(extension='json'))

        if os.path.isfile(model_path):
            self.model = load_model(model_path)
            print(('Loaded model from', model_path))

        if os.path.isfile(data_path):
            print(('Loaded data from', data_path))
            with open(data_path, 'r') as infile:
                data = json.loads(infile.read())
                self.total_games_trained = int(data['total_games_trained'])
                self.total_number_of_moves = int(data['total_number_of_moves'])
                self.average_think_time = float(data['average_think_time'])

    def save(self):
        if not os.path.isdir(saved_model_dir):
            os.makedirs(saved_model_dir)
        model_path = os.path.join(saved_model_dir, self.get_save_file())
        self.model.save(model_path)

        if not os.path.isdir(saved_data_dir):
            os.makedirs(saved_data_dir)
        data_path = os.path.join(saved_data_dir, self.get_save_file(extension='json'))
        with open(data_path, 'w') as outfile:
            json.dump(self._persisting_json_data(), outfile)

    def training_game_over(self, number_of_moves, think_time):
        total_think_time = think_time + self.average_think_time * self.total_number_of_moves
        self.total_number_of_moves += number_of_moves
        self.average_think_time = total_think_time / self.total_number_of_moves
        self.total_games_trained += 1

    def _board_states_to_inputs(self, board_states):
        inputs = np.array(board_states)
        inputs = np.expand_dims(inputs, axis=3)

        return inputs

    def _persisting_json_data(self):
        return {
            'average_think_time': self.average_think_time,
            'total_games_trained': self.total_games_trained,
            'total_number_of_moves': self.total_number_of_moves
        }

    @abstractmethod
    def get_save_file(self, extension):
        pass

    @abstractmethod
    def get_name(self):
        pass
