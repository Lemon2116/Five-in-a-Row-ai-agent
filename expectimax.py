import numpy as np
from random import choice
import time
from five_in_a_row import GameState, EvaluationFunction

EMPTY = 0
BLACK = 1
WHITE = 2


class BaseAgent:
    """Simple base class so all agents share the same interface."""
    def __init__(self, player_color="black"):
        self.player_color = player_color
        self.player = BLACK if player_color.lower() == "black" else WHITE

    def getAction(self, board):
        raise NotImplementedError
        

class ExpectimaxAgent(BaseAgent):

    def __init__(self, player_color="black", max_depth=3):
        super().__init__(player_color)
        self.opponent = WHITE if self.player == BLACK else BLACK
        self.max_depth = max_depth
        self.evaluationFunction = EvaluationFunction(self.player)

    # -----------------------------------------------------------
    # Probabilities (uniform)
    # -----------------------------------------------------------
    def compute_action_probabilities(self, board, legal_actions):
        n = len(legal_actions)
        if n == 0:
            return {}

        p = 1.0 / n
        return {action: p for action in legal_actions}

    # -----------------------------------------------------------
    # Top-level action selection
    # -----------------------------------------------------------
    def getAction(self, board):
        startTime = time.time()
        state = GameState(board, current_player=self.player, last_move=None)
        legal = state.get_legal_actions()
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
        best_action = None

        for action in legal:
            successor = state.generate_successor(action, self.player)
            score = self.expectimax_value(successor, depth=1, agentIndex=1)

            if score > best_score:
                best_score = score
                best_action = action
        
        elapsed = (time.time() - startTime) * 1000
        print(f"ExpectimaxAgent took {elapsed:.2f} ms to decide.")

        return best_action

    # -----------------------------------------------------------
    # Expectimax Recursion
    # -----------------------------------------------------------
    def expectimax_value(self, state, depth, agentIndex):
        if state.is_terminal() or depth == self.max_depth:
            return self.evaluationFunction.evaluate(state.board, self.player)

        if agentIndex == 0:
            return self.max_value(state, depth)
        else:
            return self.exp_value(state, depth)

    def max_value(self, state, depth):
        legal = state.get_legal_actions()
        if len(legal) == 0:
            return self.evaluationFunction.evaluate(state.board, self.player)

        best = float("-inf")

        for action in legal:
            next_state = state.generate_successor(action, self.player)
            score = self.expectimax_value(next_state, depth + 1, agentIndex=1)
            best = max(best, score)

        return best

    def exp_value(self, state, depth):
        legal = state.get_legal_actions()
        if len(legal) == 0:
            return self.evaluationFunction.evaluate(state.board, self.player)

        probs = self.compute_action_probabilities(state.board, legal)

        total = 0
        for action in legal:
            next_state = state.generate_successor(action, self.opponent)
            score = self.expectimax_value(next_state, depth + 1, agentIndex=0)
            total += probs[action] * score

        return total
