import numpy as np
from random import choice
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

class RandomAgent(BaseAgent):
    def __init__(self, player_color="black"):
        super().__init__(player_color)

    def getAction(self, board):
        state = GameState(board, current_player=self.player, last_move=None)
        legal_moves = state.get_legal_actions()
        return choice(legal_moves) if legal_moves else None