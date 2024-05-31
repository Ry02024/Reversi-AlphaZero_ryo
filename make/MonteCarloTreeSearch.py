import numpy as np
import math

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.legal_actions())

    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.value / child.visits) + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def expand(self):
        action = [action for action in self.state.legal_actions() if action not in [child.state.action_taken for child in self.children]][0]
        next_state = self.state.next_state(action)
        child_node = Node(next_state, parent=self)
        self.children.append(child_node)
        return child_node

    def update(self, value):
        self.visits += 1
        self.value += value

class MCTS:
    def __init__(self, model):
        self.model = model
        self.tree = {}

    def search(self, state):
        root = Node(state)

        for _ in range(1000):  # Number of simulations
            node = root
            while node.is_fully_expanded():
                node = node.best_child()
            if not node.state.is_game_over():
                node = node.expand()
            value = self.simulate(node.state)
            while node is not None:
                node.update(value)
                node = node.parent

        return root.best_child(c_param=0).state.action_taken

    def simulate(self, state):
        while not state.is_game_over():
            board_np = np.array(state.board)
            board_np = np.stack((board_np == 1, board_np == -1), axis=-1).astype(np.int32)  # チャンネル次元を追加
            policy, value = self.model.predict(board_np.reshape(1, 8, 8, 2))
            policy = policy[0]  # バッチ次元を削除
            legal_actions = state.legal_actions()
            legal_policy = np.zeros(64)
            for action in legal_actions:
                legal_policy[action] = policy[action]
            action = np.argmax(legal_policy)
            state = state.next_state(action)
        return state.game_result()

    def choose_action(self, state):
        return self.search(state)
