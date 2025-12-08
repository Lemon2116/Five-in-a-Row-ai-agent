import numpy as np
import math
from random import choice
import time
from five_in_a_row import GameState, EvaluationFunction

EMPTY = 0
BLACK = 1
WHITE = 2


class BaseAgent:
    def __init__(self, player_color="black"):
        self.player_color = player_color
        self.player = BLACK if player_color.lower() == "black" else WHITE

    def getAction(self, board):
        raise NotImplementedError



class MinimaxAgentNew(BaseAgent):

    def __init__(self, player_color, max_depth=4):
        super().__init__(player_color)
        self.opponent = WHITE if self.player == BLACK else BLACK
        self.max_depth = max_depth
        self.evaluation = EvaluationFunction(self.player)


    # ===========================================================
    #                TOP-LEVEL ACTION SELECTION
    # ===========================================================
    def getAction(self, board):
        startTime = time.time()
        state = GameState(board, current_player=self.player, last_move=None)
        legal_moves = state.get_legal_actions()
        size = board.shape[0]

        # First move = Random 9×9 region around center
        if np.all(board == EMPTY):

            mid = size // 2
            # 9×9 = radius 4 around center
            radius = 4

            # compute bounds
            xmin = max(0, mid - radius)
            xmax = min(size - 1, mid + radius)
            ymin = max(0, mid - radius)
            ymax = min(size - 1, mid + radius)

            # collect all empty squares inside the 9×9 region
            candidates = [
                (y, x)
                for y in range(ymin, ymax + 1)
                for x in range(xmin, xmax + 1)
                if board[y][x] == EMPTY
            ]

            # choose one randomly
            return choice(candidates)

        best_score = float("-inf")
        best_move = None
        alpha = float("-inf")
        beta = float("inf")

        for move in legal_moves:
            successor = state.generate_successor(move, state.current_player)
            score = self.minimax_value(successor, depth=1, alpha=alpha, beta=beta)

            if score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, best_score)

        elapsed = (time.time() - startTime) * 1000
        print(f"MinimaxAgentNew took {elapsed:.2f} ms to decide.")

        return best_move

    def minimax_value(self, state, depth, alpha, beta):
        if state.is_terminal() or depth == self.max_depth:
            return self.evaluation.evaluate(state.board, self.player)

        if state.current_player == self.player:
            return self._max_value(state, depth, alpha, beta)
        else:
            return self._min_value(state, depth, alpha, beta)


    # ===========================================================
    #                      MINIMAX CORE
    # ===========================================================
    def _max_value(self, state, depth, alpha, beta):
        value = float("-inf")

        legal_moves = state.get_legal_actions()

        for move in legal_moves:
            successor = state.generate_successor(move, state.current_player)

            value = max(value,
                        self.minimax_value(successor, depth + 1, alpha, beta))

            if value >= beta:
                return value  # β cutoff

            alpha = max(alpha, value)

        return value


    def _min_value(self, state, depth, alpha, beta):
        value = float("inf")

        legal_moves = state.get_legal_actions()

        for move in legal_moves:
            successor = state.generate_successor(move, state.current_player)

            value = min(value,
                        self.minimax_value(successor, depth + 1, alpha, beta))

            if value <= alpha:
                return value  # α cutoff

            beta = min(beta, value)

        return value

