from Game import State
from MonteCarloTreeSearch import pv_mcts_scores
from ResidualNetwork import DN_OUTPUT_SIZE
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from pathlib import Path
import numpy as np
import pickle
import os


SP_GAME_COUNT = 10 # セルフプレイを行うゲーム数（本家は25000）
SP_TEMPERATURE = 1.0 # ボルツマン分布の温度パラメータ

def first_player_value(ended_state):
    if ended_state.is_lose():
        return -1 if ended_state.is_first_player() else 1
    return 0

def write_data(history):
    now = datetime.now()
    os.makedirs('./data/', exist_ok=True)
    path = './data/{:04}{:02}{:02}{:02}{:02}{:02}.history'.format(
        now.year, now.month, now.day, now.hour, now.minute, now.second)
    with open(path, mode='wb') as f:
        pickle.dump(history, f)


def play(model, i):
    history = []
    state = State()

    while True:
        if state.is_done():
            break

        scores = pv_mcts_scores(model, state, SP_TEMPERATURE)

        # 学習データに状態と方策を追加
        policies = [0] * DN_OUTPUT_SIZE
        for action, policy in zip(state.legal_actions(), scores):
            policies[action] = policy
        history.append([[state.pieces, state.enemy_pieces], policies, None])

        action = np.random.choice(state.legal_actions(), p=scores)
        state = state.next(action)

    # 学習データに価値を追加
    value = first_player_value(state)
    for i in range(len(history)):
        history[i][2] = value
        value = -value
    return history


def self_play():
    history = []
    model = load_model('./model/best.h5')
    for i in range(SP_GAME_COUNT):
        print(f"第{i+1}ゲーム開始")
        h = play(model,i)
        history.extend(h)

        print('\rSelfPlay {}/{}'.format(i+1, SP_GAME_COUNT), end='')
    print('')

    write_data(history)

    K.clear_session()
    del model

if __name__ == '__main__':
    self_play()
