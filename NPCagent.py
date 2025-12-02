# ========================= NPCagent.py ============================
import numpy as np
from random import choice


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
        self.color = BLACK if player_color.lower() == "black" else WHITE
        self.opponent_color = WHITE if self.color == BLACK else BLACK
        
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