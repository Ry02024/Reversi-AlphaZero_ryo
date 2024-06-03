from Game import State
from MonteCarloTreeSearch import pv_mcts_action
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from pathlib import Path
from shutil import copy
import numpy as np
import matplotlib.pyplot as plt

EN_GAME_COUNT = 1 # 1評価あたりのゲーム数
EN_TEMPERATURE = 1.0 # ボルツマン分布の温度

def first_player_point(ended_state):
    # 1:先手勝利, 0:先手敗北, 0.5:引き分け
    if ended_state.is_lose():
        return 0 if ended_state.is_first_player() else 1
    return 0.5


def play(next_actions):
    state = State()

    while True:
        if state.is_done():
            break;

        next_action = next_actions[0] if state.is_first_player() else next_actions[1]
        action = next_action(state)

        state = state.next(action)

    return first_player_point(state)


def update_best_player():
    copy('./model/latest.h5', './model/best.h5')
    print('Change BestPlayer')

def plot_evaluation_results(results):
    games = range(1, len(results) + 1)
    points = [result['points'] for result in results]

    plt.figure(figsize=(10, 6))
    plt.plot(games, points, 'bo-', label='Points per game')
    plt.title('Evaluation Results')
    plt.xlabel('Game')
    plt.ylabel('Points')
    plt.legend()
    plt.show()

def evaluate_network():
    model0 = load_model('./model/latest.h5')
    model1 = load_model('./model/best.h5')

    # モンテカルロ木探索で行動選択を行う関数の生成
    next_action0 = pv_mcts_action(model0, EN_TEMPERATURE)
    next_action1 = pv_mcts_action(model1, EN_TEMPERATURE)
    next_actions = (next_action0, next_action1)


    total_point = 0
    evaluation_results = []
    for i in range(EN_GAME_COUNT):
        if i % 2 == 0:
            points = play(next_actions)
            total_point += points
        else:
            points = 1 - play(list(reversed(next_actions)))
            total_point += points

        evaluation_results.append({'game': i + 1, 'points': points})

        print('\rEvaluate {}/{}'.format(i + 1, EN_GAME_COUNT), end='')
    print('')

    average_point = total_point / EN_GAME_COUNT
    print('AveragePoint', average_point)

    K.clear_session()
    del model0
    del model1

    plot_evaluation_results(evaluation_results)

    if average_point > 0.5:
        update_best_player()
        return True
    else:
        return False

if __name__ == '__main__':
    evaluate_network()
