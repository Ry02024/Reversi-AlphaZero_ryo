import numpy as np
import pickle
from pathlib import Path
from ResidualNetwork import build_model
from MonteCarloTreeSearch import MCTS
from Game import State
import tensorflow as tf

def self_play(model, num_games, log_game=False):
    data = []
    results = {"wins": 0, "losses": 0, "draws": 0}

    for game_index in range(num_games):
        state = State()
        mcts = MCTS(model)
        game_data = []
        turn = 0

        if log_game and game_index == 0:  # 最初のゲームのみ詳細ログを表示
            print(f"Starting game {game_index + 1}/{num_games}...")

        while not state.is_game_over():
            action = mcts.choose_action(state)
            game_data.append((state.board, action))
            state = state.next_state(action)
            turn += 1
            if log_game and game_index == 0 and turn % 10 == 0:  # 10ターンごとにログを表示
                print(f"Game {game_index + 1}/{num_games} - Turn {turn}")

        result = state.game_result()
        if result == 1:
            results["wins"] += 1
        elif result == -1:
            results["losses"] += 1
        else:
            results["draws"] += 1

        for board, action in game_data:
            data.append((board, action, result))

        if log_game and game_index == 0:  # 最初のゲームの結果を表示
            print(f"Game {game_index + 1}/{num_games} finished. Result: {result}")

    with open('data/self_play_data.pkl', 'wb') as f:
        pickle.dump(data, f)

    print("All games completed.")
    print(f"Results: {results['wins']} Wins, {results['losses']} Losses, {results['draws']} Draws")

    return results

if __name__ == '__main__':
    model = build_model()
    try:
        model.load_weights('model/best.h5')
        print("Loaded model weights from 'model/best.h5'.")
    except (OSError, IOError) as e:
        print("Could not load model weights from 'model/best.h5', initializing a new model with random weights.")

    # 全試合結果を表示しないように設定し、最初の試合の詳細ログを表示
    self_play(model, num_games=100, log_game=True)
