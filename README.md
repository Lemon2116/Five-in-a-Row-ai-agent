# Five-in-a-Row-ai-agent
This project investigates the design and evaluation of several agents for playing the Five-in-a-Row (Gomoku) game. We implemented three decision-making agents (Minimax, Expectimax, and a Reinforcement Learning (RL) agent) along with a Random agent used as a baseline for evaluation. Our heuristic evaluation function considers a variety of board configurations, penalizing states that lead toward loss while rewarding patterns that move the agent closer to victory. Although the time complexity of the search-based agents increases significantly with deeper search depths, all agents demonstrated competitive performance against human players. When constrained to shallower depths, Minimax exhibited weakened performance, whereas Expectimax consistently performed the best. The reinforcement learning agent successfully converged to a stable policy; however, it still requires more extensive training to reach optimal performance through more oriented self-play and real gameplay data.  
# AI Agents
Minimax Agent:
Expectimax Agent:
Reinforcement Agent:
# Sample Results
![Expectisqr](img/Expectisqr.png "Example: Expectimax vs. Expectimax")
# Winning Record
![Record](img/Record.png "Record of different types of agents playing against each other")

