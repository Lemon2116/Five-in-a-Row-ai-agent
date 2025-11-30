# ========================= NPCagent.py ============================
import numpy as np
from random import choice
import math
import copy

BOARD_SIZE = 15
EMPTY = 0
BLACK = 1
WHITE = 2

class BaseAgent:
    """
    Base class for all AI agents.
    All agents must implement getAction().
    """

    def __init__(self, player_color):
        self.color = player_color  # "black" or "white"

    def register_initial_state(self, board):
        """Place the first stone in the center."""
        center = BOARD_SIZE // 2
        board[center][center] = self.color
        return (center, center)

    def getAction(self, board):
        """
        Return (row, col) of the move.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    # ================= Utility methods ==================
    def get_legal_moves(self, board):
        """Return empty positions list."""
        moves = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r][c] is None:
                    moves.append((r, c))
        return moves

    def nearby_weighted_moves(self, board, max_distance=2):
        """
        Generate weighted probability distribution:
        - Moves near existing stones => higher probability
        - Moves far away => low probability
        """
        weighted = []

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r][c] is None:
                    score = self._proximity_score(board, (r, c), max_distance)
                    if score > 0:
                        weighted.append((score, (r, c)))

        # If no meaningful position found (rare), fallback to random corner area
        if not weighted:
            return [(1, m) for m in self.get_legal_moves(board)]

        # Normalize to probability distribution
        total = sum(w for w,_ in weighted)
        return [(w / total, move) for w, move in weighted]

    def _proximity_score(self, board, pos, dist):
        """Count friendly stones within distance → used for probability weighting"""
        r,c = pos
        score = 0
        for i in range(-dist, dist+1):
            for j in range(-dist, dist+1):
                nr, nc = r+i, c+j
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    if board[nr][nc] is not None:
                        score += 1
        return score


# =========================================================
# Expectimax Agent
# =========================================================

class ExpectimaxAgent(BaseAgent):
    def getAction(self, board):
        """
        Return (x,y) move.
        Probability is higher near stones, very low elsewhere.
        """

        legal_moves = list(zip(*np.where(board == EMPTY)))

        # -----------------
        # Compute proximity scores
        # -----------------
        scores = {}
        for (y,x) in legal_moves:
            # Look around neighbors within Manhattan range 1–3
            region = board[max(0,y-1):y, max(0,x-1):x]
            count_stones = np.count_nonzero(region != EMPTY)
            scores[(x,y)] = count_stones  # more neighbors = better probability

        # Normalize to probability
        total = sum(scores.values())
        if total == 0:  # opening or empty board
            # place on center
            s = board.shape[0]//2
            return (s,s)

        # Weighted sampling like probability selection
        moves, weights = zip(*scores.items())
        weights = np.array(weights, dtype=float) / sum(weights)

        # draw move according to distribution
        idx = np.random.choice(len(moves), p=weights)
        return moves[idx]