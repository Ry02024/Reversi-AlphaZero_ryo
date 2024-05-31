class State:
    def __init__(self):
        self.board = self.initial_board()
        self.player = 1

    def initial_board(self):
        board = [[0] * 8 for _ in range(8)]
        board[3][3], board[3][4], board[4][3], board[4][4] = 1, -1, -1, 1
        return board

    def legal_actions(self):
        actions = []
        for x in range(8):
            for y in range(8):
                if self.board[x][y] == 0 and self.is_legal(x, y):
                    actions.append((x, y))
        return actions

    def is_legal(self, x, y):
        # Implement legality check based on Reversi rules
        pass

    def next_state(self, action):
        new_state = State()
        new_state.board = [row[:] for row in self.board]
        new_state.player = -self.player
        # Apply the action to the board and flip the opponent's pieces
        return new_state

    def game_result(self):
        player1_count = sum(row.count(1) for row in self.board)
        player2_count = sum(row.count(-1) for row in self.board)
        if player1_count > player2_count:
            return 1
        elif player2_count > player1_count:
            return -1
        else:
            return 0

    def is_game_over(self):
        return not self.legal_actions()
