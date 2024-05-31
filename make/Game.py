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
        if self.board[x][y] != 0:
            return False
        for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 8 and 0 <= ny < 8 and self.board[nx][ny] == -self.player:
                while 0 <= nx < 8 and 0 <= ny < 8:
                    nx += dx
                    ny += dy
                    if not (0 <= nx < 8 and 0 <= ny < 8):
                        break
                    if self.board[nx][ny] == self.player:
                        return True
                    if self.board[nx][ny] == 0:
                        break
        return False

    def next_state(self, action):
        new_state = State()
        new_state.board = [row[:] for row in self.board]
        new_state.player = -self.player
        x, y = action
        new_state.board[x][y] = self.player
        for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            nx, ny = x + dx, y + dy
            pieces_to_flip = []
            while 0 <= nx < 8 and 0 <= ny < 8:
                if new_state.board[nx][ny] == -self.player:
                    pieces_to_flip.append((nx, ny))
                elif new_state.board[nx][ny] == self.player:
                    for fx, fy in pieces_to_flip:
                        new_state.board[fx][fy] = self.player
                    break
                else:
                    break
                nx += dx
                ny += dy
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
