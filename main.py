from expectimax import ExpectimaxAgent
from minimax_modified import MinimaxAgentNew
from minimax import MinimaxAgent
from five_in_a_row import Game
from randomAgent import RandomAgent
from five_in_a_row import Game, BLACK, WHITE
import numpy as np
from RLearning import RLAgent


def main():
    mode = input(
        "\nSelect mode:\n"
        "1) Human vs Human\n"
        "2) Human vs Expectimax AI\n"
        "3) Minimax vs Expectimax (AI vs AI)\n"
        "4) Human vs RL (trained)\n"
        ">> "
    )

    if mode == "1":
        # Human vs Human
        game = Game(human=True, npc=None)

    elif mode == "2":
        # Human vs Expectimax AI (white)
        ai = ExpectimaxAgent(player_color="white")
        game = Game(human=True, npc=ai)

    elif mode == "3":
        # Minimax (black) vs Expectimax (white)
        ai1 = MinimaxAgent(player=BLACK, max_depth=2)
        ai2 = ExpectimaxAgent(player_color="white")
        game = Game(human=False, npc=[ai1, ai2])

    elif mode == "4":
        # Human vs RL agent (white)
        try:
            weights = np.load("rl_weights.npy")
            print("Loaded rl_weights.npy")
        except FileNotFoundError:
            print("WARNING: rl_weights.npy not found, using untrained weights.")
            weights = None

        ai = RLAgent(player=WHITE, weights=weights, train=False)
        game = Game(human=True, npc=ai)

    else:
        print("Invalid mode, defaulting to Human vs Human")
        game = Game(human=True, npc=None)

    game.run()