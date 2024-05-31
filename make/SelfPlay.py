import numpy as np
import pickle
from pathlib import Path
from ResidualNetwork import build_model
from MonteCarloTreeSearch import MCTS
from Game import State
import tensorflow as tf

def self_play(model, num_games):
    data = []

    for _ in range(num_games):
        state = State()
        mcts = MCTS(model)
        game_data = []

        while not state.is_game_over():
            action = mcts.choose_action(state)
            game_data.append((state.board, action))
            state = state.next_state(action)

        result = state.game_result()
        for board, action in game_data:
            data.append((board, action, result))

    with open('data/self_play_data.pkl', 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    model = build_model()
    try:
        model.load_weights('model/best.h5')
        print("Loaded model weights from 'model/best.h5'.")
    except (OSError, IOError) as e:
        print("Could not load model weights from 'model/best.h5', initializing a new model with random weights.")

    self_play(model, num_games=100)
