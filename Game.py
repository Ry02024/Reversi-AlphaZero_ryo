import random
import argparse
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime
import os
import json

class State:
    #############################
    ####### ゲームの初期設定 #######
    #############################

    def __init__(self, pieces=None, enemy_pieces=None, depth=0):
        self.dxy = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))

        self.pass_end = False

        self.pieces = pieces
        self.enemy_pieces = enemy_pieces
        self.depth = depth

        if pieces is None or enemy_pieces is None:
            self.pieces = [0] * 64
            self.pieces[27] = self.pieces[36] = 1
            self.enemy_pieces = [0] * 64
            self.enemy_pieces[28] = self.enemy_pieces[35] = 1

    def piece_count(self, pieces):
        return sum(1 for i in pieces if i == 1)

    ##################################
    ####### ゲームの勝敗と終了判定 ########
    ##################################

    def is_done(self):
        return self.piece_count(self.pieces) + self.piece_count(self.enemy_pieces) == 64 or self.pass_end

    def is_lose(self):
        return self.is_done() and self.piece_count(self.pieces) < self.piece_count(self.enemy_pieces)

    def is_draw(self):
        return self.is_done() and self.piece_count(self.pieces) == self.piece_count(self.enemy_pieces)

    ##########################
    ####### 合法手の判定 #######
    ##########################

    # 任意のマスが合法手か判定
    def legal_action_xy(self, x, y, flip=False):
        # 任意のマスの任意の方向が合法手か判定
        def legal_action_xy_dxy(x, y, dx, dy):
            # １つ目 相手の石
            x, y = x + dx, y + dy
            if y < 0 or 7 < y or x < 0 or 7 < x or self.enemy_pieces[x + y * 8] != 1:
                return False

            # 2つ目以降
            for j in range(8):
                # 空
                if y < 0 or 7 < y or x < 0 or 7 < x or (self.enemy_pieces[x + y * 8] == 0 and self.pieces[x + y * 8] == 0):
                    return False

                # 自分の石
                if self.pieces[x + y * 8] == 1:
                    # 反転
                    if flip:
                        for i in range(8):
                            x, y = x - dx, y - dy
                            if self.pieces[x + y * 8] == 1:
                                return True
                            self.pieces[x + y * 8] = 1
                            self.enemy_pieces[x + y * 8] = 0
                    return True
                # 相手の石
                x, y = x + dx, y + dy
            return False

        # 空きなし
        if self.enemy_pieces[x + y * 8] == 1 or self.pieces[x + y * 8] == 1:
            return False

        # 石を置く
        if flip:
            self.pieces[x + y * 8] = 1

        # 任意の位置が合法手かどうか
        flag = False
        for dx, dy in self.dxy:
            if legal_action_xy_dxy(x, y, dx, dy):
                flag = True
        return flag

    # 合法手のリストの取得
    def legal_actions(self):
        actions = []
        for j in range(8):
            for i in range(8):
                if self.legal_action_xy(i, j):
                    actions.append(i + j * 8)
        if len(actions) == 0:
            actions.append(64)
        return actions

    ##############################
    ######## 次の状態の取得 #########
    ##############################

    def next(self, action):
        state = State(self.pieces.copy(), self.enemy_pieces.copy(), self.depth + 1)
        # 挟んだ石を裏返す処理
        if action != 64:
            state.legal_action_xy(action % 8, action // 8, True)
        state.pieces, state.enemy_pieces = state.enemy_pieces, state.pieces

        # 2回連続パス判定
        if action == 64 and state.legal_actions() == [64]:
            state.pass_end = True
        return state

    ###########################
    ######## 先手かどうか ########
    ###########################

    def is_first_player(self):
        return self.depth % 2 == 0

    #######################
    ####### 出力設定 #######
    #######################

    def __str__(self):
        ox = ('o', 'x') if self.is_first_player() else ('x', 'o')
        board_str = ''
        for i in range(64):
            if self.pieces[i] == 1:
                board_str += ox[0]
            elif self.enemy_pieces[i] == 1:
                board_str += ox[1]
            else:
                board_str += '-'
            if i % 8 == 7:
                board_str += '\n'
        return board_str


#########################
######## 動作確認 ########
#########################

# ランダムで行動選択
def random_action(state):
    legal_actions = state.legal_actions()
    return legal_actions[random.randint(0, len(legal_actions) - 1)]

# モデルで行動選択
def model_action(state, model):
    board_np = np.array(state.pieces + state.enemy_pieces).reshape((1, 8, 8, 2))
    policies, _ = model.predict(board_np)
    legal_actions = state.legal_actions()
    legal_policies = [(policies[0][action], action) for action in legal_actions]
    _, action = max(legal_policies)
    return action

def log_game_state(state, action, file):
    # 機械学習用ログ
    log_entry = {
        "type": "machine",
        "state": {
            "pieces": state.pieces,
            "enemy_pieces": state.enemy_pieces
        },
        "action": action
    }
    with open(file, 'a') as f:
        f.write(json.dumps(log_entry) + "\n")
    
    # 人間向けログ
    human_log_entry = {
        "type": "human",
        "action": action,
        "board": str(state)
    }
    with open(file, 'a') as f:
        f.write(json.dumps(human_log_entry) + "\n")

def log_game_result(result, file):
    with open(file, 'a') as f:
        f.write(json.dumps({"type": "result", "result": result}) + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_model', action='store_true', help='Use model for action selection')
    parser.add_argument('--model_path', type=str, default='path_to_your_model.h5', help='Path to the model file')
    parser.add_argument('--log_file', type=str, default='game_log.json', help='Path to the log file')
    args = parser.parse_args()

    use_model = args.use_model
    model = None

    if use_model:
        model = load_model(args.model_path)

    log_file = args.log_file
    
    if log_file:
        with open(log_file, 'w') as f:
            f.write("")

    state = State()

    while True:
        if state.is_done():
            break

        if use_model and model:
            action = model_action(state, model)
        else:
            action = random_action(state)

        if log_file:
            log_game_state(state, action, log_file)
        
        state = state.next(action)

        print(state)
        print()

    if log_file:
        if state.is_draw():
            result = "Draw"
        elif state.is_lose():
            result = "Lose"
        else:
            result = "Win"
        log_game_result(result, log_file)
