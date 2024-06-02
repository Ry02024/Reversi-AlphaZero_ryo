import random
import argparse
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime
import os
import json

class State:
    def __init__(self, pieces=None, enemy_pieces=None, depth=0):
        self.dxy = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
        self.pass_end = False
        self.pieces = pieces if pieces is not None else [0] * 64
        self.enemy_pieces = enemy_pieces if enemy_pieces is not None else [0] * 64
        self.depth = depth
        if pieces is None or enemy_pieces is None:
            self.pieces[27] = self.pieces[36] = 1
            self.enemy_pieces[28] = self.enemy_pieces[35] = 1

    def piece_count(self, pieces):
        return sum(1 for i in pieces if i == 1)

    def is_done(self):
        return self.piece_count(self.pieces) + self.piece_count(self.enemy_pieces) == 64 or self.pass_end

    def is_lose(self):
        return self.is_done() and self.piece_count(self.pieces) < self.piece_count(self.enemy_pieces)

    def is_draw(self):
        return self.is_done() and self.piece_count(self.pieces) == self.piece_count(self.enemy_pieces)

    def legal_action_xy(self, x, y, flip=False):
        def legal_action_xy_dxy(x, y, dx, dy):
            x, y = x + dx, y + dy
            if y < 0 or 7 < y or x < 0 or 7 < x or self.enemy_pieces[x + y * 8] != 1:
                return False
            for _ in range(8):
                if y < 0 or 7 < y or x < 0 or 7 < x or (self.enemy_pieces[x + y * 8] == 0 and self.pieces[x + y * 8] == 0):
                    return False
                if self.pieces[x + y * 8] == 1:
                    if flip:
                        for _ in range(8):
                            x, y = x - dx, y - dy
                            if self.pieces[x + y * 8] == 1:
                                return True
                            self.pieces[x + y * 8] = 1
                            self.enemy_pieces[x + y * 8] = 0
                    return True
                x, y = x + dx, y + dy
            return False
        if self.enemy_pieces[x + y * 8] == 1 or self.pieces[x + y * 8] == 1:
            return False
        if flip:
            self.pieces[x + y * 8] = 1
        return any(legal_action_xy_dxy(x, y, dx, dy) for dx, dy in self.dxy)

    def legal_actions(self):
        actions = [i + j * 8 for j in range(8) for i in range(8) if self.legal_action_xy(i, j)]
        if not actions:
            actions.append(64)
        return actions

    def next(self, action):
        state = State(self.pieces.copy(), self.enemy_pieces.copy(), self.depth + 1)
        if action != 64:
            state.legal_action_xy(action % 8, action // 8, True)
        state.pieces, state.enemy_pieces = state.enemy_pieces, state.pieces
        if action == 64 and state.legal_actions() == [64]:
            state.pass_end = True
        return state

    def is_first_player(self):
        return self.depth % 2 == 0

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
    while not state.is_done():
        action = random_action(state)
        state = state.next(action)
    
    pieces_count = state.piece_count(state.pieces)
    enemy_pieces_count = state.piece_count(state.enemy_pieces)
    total_moves = state.depth
    
    if state.is_draw():
        result = "Draw"
    elif state.is_lose():
        result = "Lose"
    else:
        result = "Win"
    
    print(f"Result:{result}")
    print(f"Player 'o' pieces: {pieces_count}")
    print(f"Player 'x' pieces: {enemy_pieces_count}")
    print(f"Total moves: {total_moves}")
