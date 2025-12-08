# RLearning.py
# Simple Q-learning based reinforcement learning agent for Five-in-a-Row.
#
# This agent:
#   - Uses linear function approximation: Q(s,a) = w · phi(s,a)
#   - Learns offline via self-play vs. a random opponent (see train() at bottom)
#   - Can be plugged into the existing Game class by calling getAction(board)

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional

from five_in_a_row import SIZE, EMPTY, BLACK, WHITE

Coord = Tuple[int, int]


# ----------------------------------------------------------------------
# Utility: win check on a raw NumPy board
# ----------------------------------------------------------------------
def check_win_np(board: np.ndarray, x: int, y: int, player: Optional[int] = None) -> int:
    """
    Gomoku win check on a raw NumPy board.
    `board[y, x]` is assumed to be the last move for `player`.

    Returns:
        BLACK or WHITE if that player has 5 in a row including (x, y),
        otherwise 0.
    """
    size = board.shape[0]

    if player is None:
        player = int(board[y, x])

    if player == EMPTY:
        return 0

    directions = [
        (1, 0),   # horizontal
        (0, 1),   # vertical
        (1, 1),   # diag /
        (1, -1),  # diag \
    ]

    for dx, dy in directions:
        count = 1  # stone at (x, y)

        # forward
        nx, ny = x + dx, y + dy
        while 0 <= nx < size and 0 <= ny < size and board[ny, nx] == player:
            count += 1
            nx += dx
            ny += dy

        # backward
        nx, ny = x - dx, y - dy
        while 0 <= nx < size and 0 <= ny < size and board[ny, nx] == player:
            count += 1
            nx -= dx
            ny -= dy

        if count >= 5:
            return player

    return 0


# ----------------------------------------------------------------------
# RL Agent
# ----------------------------------------------------------------------
class RLAgent:
    """
    Q-learning agent with linear function approximation.

    Usage in the Pygame game:
        from RLearning import RLAgent
        ai = RLAgent(player=WHITE, train=False,
                     weights=np.load("rl_weights.npy"),
                     use_tactics=True)
        game = Game(human=True, npc=ai)

    For training, run this file directly:
        python RLearning.py
    """

    def __init__(
        self,
        player: int = BLACK,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        weights: Optional[np.ndarray] = None,
        train: bool = False,
        use_tactics: bool = True,
    ) -> None:
        self.player = player
        self.opponent = WHITE if player == BLACK else BLACK
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount factor
        self.epsilon = epsilon  # exploration rate
        self.train = train
        self.use_tactics = use_tactics

        # 10-dimensional feature vector:
        self.num_features = 10

        if weights is None:
            self.weights = np.zeros(self.num_features, dtype=float)
        else:
            w = np.asarray(weights, dtype=float)
            if w.shape != (self.num_features,):
                raise ValueError(
                    f"weights must have shape {(self.num_features,)}, got {w.shape}"
                )
            self.weights = w

        # Cache for one-step Q-learning
        self._last_state_board: Optional[np.ndarray] = None
        self._last_action: Optional[Coord] = None
        self._last_features: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API used by Game
    # ------------------------------------------------------------------
    def getAction(self, board: np.ndarray) -> Coord:
        """
        Choose an action given the current board.

        If train=True, this will also prepare the cache for a future
        Q-learning update via notify_transition().
        """
        legal_moves = self._legal_moves(board)
        if not legal_moves:
            return (SIZE // 2, SIZE // 2)

        # Special case: in PLAY mode, if board is empty, go to center
        if not self.train and np.count_nonzero(board != EMPTY) == 0:
            c = SIZE // 2
            return (c, c)

        # Optional tactical shortcuts (ONLY if use_tactics is True)
        if self.use_tactics:
            # --- 1. If we can win in one move, do it ---
            for move in legal_moves:
                x, y = move
                if board[y, x] != EMPTY:
                    continue
                board[y, x] = self.player
                if check_win_np(board, x, y, self.player) == self.player:
                    board[y, x] = EMPTY
                    # we treat this as a chosen action, so record features
                    if self.train:
                        self._last_state_board = board.copy()
                        self._last_action = move
                        self._last_features = self._features(board, move)
                    return (x, y)
                board[y, x] = EMPTY

            # --- 2. If opponent can win in one move, block it ---
            for move in legal_moves:
                x, y = move
                if board[y, x] != EMPTY:
                    continue
                board[y, x] = self.opponent
                if check_win_np(board, x, y, self.opponent) == self.opponent:
                    board[y, x] = EMPTY
                    if self.train:
                        self._last_state_board = board.copy()
                        self._last_action = move
                        self._last_features = self._features(board, move)
                    return (x, y)
                board[y, x] = EMPTY

        # --- 3. Otherwise, fall back to epsilon-greedy Q policy ---
        if self.train and np.random.rand() < self.epsilon:
            move = legal_moves[np.random.randint(len(legal_moves))]
        else:
            q_values = [self._q_value(board, m) for m in legal_moves]
            best_idx = int(np.argmax(q_values))
            move = legal_moves[best_idx]

        if self.train:
            self._last_state_board = board.copy()
            self._last_action = move
            self._last_features = self._features(board, move)

        x, y = move
        return (x, y)

    # ------------------------------------------------------------------
    # Q-learning update interface
    # ------------------------------------------------------------------
    def notify_transition(self, next_board: np.ndarray, reward: float, done: bool) -> None:
        """
        Perform a one-step Q-learning update based on the transition:

            (last_state, last_action) --reward--> next_board
        """
        if not self.train:
            return
        if (self._last_state_board is None or
                self._last_action is None or
                self._last_features is None):
            return

        # Compute TD target
        if done:
            target = reward
        else:
            legal_moves_next = self._legal_moves(next_board)
            if not legal_moves_next:
                target = reward
            else:
                q_next = [self._q_value(next_board, m) for m in legal_moves_next]
                target = reward + self.gamma * max(q_next)

        # Current estimate
        current_q = float(np.dot(self.weights, self._last_features))

        # Q-learning update
        td_error = target - current_q
        self.weights += self.alpha * td_error * self._last_features

        # Clear cache – next step will set it again
        self._last_state_board = None
        self._last_action = None
        self._last_features = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _legal_moves(self, board: np.ndarray) -> List[Coord]:
        ys, xs = np.where(board == EMPTY)
        return [(int(x), int(y)) for x, y in zip(xs, ys)]

    def _q_value(self, board: np.ndarray, move: Coord) -> float:
        features = self._features(board, move)
        return float(np.dot(self.weights, features))

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def _features(self, board: np.ndarray, move: Coord) -> np.ndarray:
        """
        Compute a feature vector for placing a stone at `move`
        (x, y) for self.player on the given board.
        """
        x, y = move

        # If illegal, strongly discourage
        if board[y, x] != EMPTY:
            f = np.zeros(self.num_features, dtype=float)
            f[0] = 1.0  # bias
            return f

        # Clone board and make the move for self.player
        new_board = board.copy()
        new_board[y, x] = self.player

        size = board.shape[0]
        center = (size - 1) / 2.0

        # 1) Bias
        bias = 1.0

        # 2) Longest run for self that passes through (x, y)
        max_run_self = self._max_run_through(new_board, x, y, self.player)

        # 3) Self "big threat" indicator (>=4 in a row)
        self_threat_4plus = 1.0 if max_run_self >= 4 else 0.0

        # 4–5) Opponent longest run if they had played here (on original board)
        tmp_board = board.copy()
        tmp_board[y, x] = self.opponent
        max_run_opp_if_not_blocked = self._max_run_through(
            tmp_board, x, y, self.opponent
        )
        opp_threat_4plus = 1.0 if max_run_opp_if_not_blocked >= 4 else 0.0

        # 6–7) Nearby stones in 3x3 neighborhood
        friends, enemies = self._nearby_counts(new_board, x, y, radius=1)

        # 8) Board occupation ratio
        total_cells = float(board.size)
        occupied_cells = float(np.count_nonzero(board != EMPTY))
        occupation_ratio = occupied_cells / total_cells

        # 9) Center preference: 1 at center, ~0 at furthest corners
        dx = abs(x - center)
        dy = abs(y - center)
        max_dist = np.sqrt(2) * center  # distance from center to corner
        dist = np.sqrt(dx * dx + dy * dy)
        center_pref = 1.0 - dist / max_dist  # in [0,1]

        # 10) Stone difference ratio (self stones - opp stones)
        self_count = float(np.count_nonzero(board == self.player))
        opp_count = float(np.count_nonzero(board == self.opponent))
        stone_diff_ratio = (self_count - opp_count) / total_cells

        f = np.zeros(self.num_features, dtype=float)
        f[0] = bias
        f[1] = max_run_self / 5.0
        f[2] = self_threat_4plus
        f[3] = max_run_opp_if_not_blocked / 5.0
        f[4] = opp_threat_4plus
        f[5] = friends / 8.0
        f[6] = enemies / 8.0
        f[7] = occupation_ratio
        f[8] = center_pref
        f[9] = stone_diff_ratio

        return f

    def _max_run_through(self, board: np.ndarray, x: int, y: int, player: int) -> int:
        """Max number of consecutive stones for `player` incl. (x, y)."""
        if board[y, x] != player:
            return 0

        directions = [
            (1, 0),   # horizontal
            (0, 1),   # vertical
            (1, 1),   # diag /
            (1, -1),  # diag \
        ]
        size = board.shape[0]
        best = 1
        for dx, dy in directions:
            count = 1

            cx, cy = x + dx, y + dy
            while 0 <= cx < size and 0 <= cy < size and board[cy, cx] == player:
                count += 1
                cx += dx
                cy += dy

            cx, cy = x - dx, y - dy
            while 0 <= cx < size and 0 <= cy < size and board[cy, cx] == player:
                count += 1
                cx -= dx
                cy -= dy

            if count > best:
                best = count
        return best

    def _nearby_counts(
        self,
        board: np.ndarray,
        x: int,
        y: int,
        radius: int = 1
    ) -> Tuple[int, int]:
        """Count number of self/opponent stones in a (2r+1)x(2r+1) box."""
        size = board.shape[0]
        friends = 0
        enemies = 0
        for yy in range(max(0, y - radius), min(size, y + radius + 1)):
            for xx in range(max(0, x - radius), min(size, x + radius + 1)):
                if board[yy, xx] == self.player:
                    friends += 1
                elif board[yy, xx] == self.opponent:
                    enemies += 1
        return friends, enemies


# ----------------------------------------------------------------------
# Offline training via self-play vs random opponent
# ----------------------------------------------------------------------
class RandomOpponent:
    """Very simple random opponent, used for training."""
    def __init__(self, player: int) -> None:
        self.player = player

    def getAction(self, board: np.ndarray) -> Coord:
        ys, xs = np.where(board == EMPTY)
        if len(xs) == 0:
            return (SIZE // 2, SIZE // 2)
        idx = np.random.randint(len(xs))
        return (int(xs[idx]), int(ys[idx]))


def play_one_episode(
    rl_agent: RLAgent,
    rl_color: int,
) -> float:
    """
    Play a single training episode of RLAgent vs RandomOpponent.
    Returns +1 if RL wins, -1 if RL loses, 0 for draw.
    """
    board = np.zeros((SIZE, SIZE), dtype=int)

    rl_agent.train = True
    rl_agent._last_state_board = None
    rl_agent._last_action = None
    rl_agent._last_features = None

    rl_player = rl_color
    opp_player = WHITE if rl_player == BLACK else BLACK
    opponent = RandomOpponent(player=opp_player)

    current_player = BLACK  # always start with BLACK
    winner = 0

    while True:
        if current_player == rl_player:
            # RL's turn
            x, y = rl_agent.getAction(board)

            if board[y, x] != EMPTY:
                # illegal move, punish hard and end
                rl_agent.notify_transition(board, reward=-1.0, done=True)
                winner = opp_player
                break

            board[y, x] = rl_player

            # Did RL just win?
            win_id = check_win_np(board, x, y, rl_player)
            if win_id == rl_player:
                rl_agent.notify_transition(board, reward=1.0, done=True)
                winner = rl_player
                break

            # Board full?
            if not (board == EMPTY).any():
                rl_agent.notify_transition(board, reward=0.0, done=True)
                winner = 0
                break

            # Opponent moves
            ox, oy = opponent.getAction(board)
            if board[oy, ox] == EMPTY:
                board[oy, ox] = opp_player
                win_id = check_win_np(board, ox, oy, opp_player)
                if win_id == opp_player:
                    rl_agent.notify_transition(board, reward=-1.0, done=True)
                    winner = opp_player
                    break

            # Intermediate step: reward 0, not done
            rl_agent.notify_transition(board.copy(), reward=0.0, done=False)

        else:
            # Opponent plays first when RL is WHITE
            ox, oy = opponent.getAction(board)
            if board[oy, ox] == EMPTY:
                board[oy, ox] = opp_player
                win_id = check_win_np(board, ox, oy, opp_player)
                if win_id == opp_player:
                    winner = opp_player
                    break

        # Draw?
        if not (board == EMPTY).any():
            winner = 0
            break

        # Switch turns
        current_player = WHITE if current_player == BLACK else BLACK

    if winner == rl_player:
        return 1.0
    elif winner == opp_player:
        return -1.0
    else:
        return 0.0


def train(
    episodes: int = 3000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.2,
    rl_color: int = BLACK,
    save_path: str = "rl_weights.npy",
) -> np.ndarray:
    """
    Train RLAgent vs RandomOpponent for a number of episodes and save weights.
    """
    agent = RLAgent(
        player=rl_color,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        train=True,
        use_tactics=False,   # <<< PURE RL DURING TRAINING
    )

    rewards = []
    for ep in range(1, episodes + 1):
        r = play_one_episode(agent, rl_color)
        rewards.append(r)

        # simple epsilon decay
        agent.epsilon = max(0.05, agent.epsilon * 0.999)

        if ep % 50 == 0:
            avg = np.mean(rewards[-50:])
            print(f"Episode {ep:4d} / {episodes}, last-50 avg reward = {avg: .3f}")

    np.save(save_path, agent.weights)
    print(f"Saved weights to {save_path}")
    print("Learned weights:", agent.weights)
    return agent.weights


if __name__ == "__main__":
    train(episodes=3000)