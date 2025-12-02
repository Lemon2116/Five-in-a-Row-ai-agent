from expectimax import ExpectimaxAgent
from minimax import MinimaxAgent
from five_in_a_row import Game

mode = input("\nSelect mode:\n1) Human vs Human\n2) Human vs AI\n3) AI vs AI\n>> ")

if mode == "1":
    game = Game(human=True, npc=None)

elif mode == "2":
    ai = ExpectimaxAgent(player_color="white")
    game = Game(human=True, npc=ai)

elif mode == "3":
    ai1 = MinimaxAgent("black")
    ai2 = ExpectimaxAgent("white")
    game = Game(human=False, npc=[ai1, ai2])

game.run()
