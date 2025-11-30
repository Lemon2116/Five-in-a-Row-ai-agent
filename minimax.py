# minimax.py

import math
import numpy as np

EMPTY = 0
BLACK = 1
WHITE = 2


class MinimaxAgent:
    """
    Minimax + alpha-beta pruning agent for Five-in-a-Row.
    Uses a hand-crafted heuristic (no learning).
    """

    def __init__(self, player=BLACK, max_depth=2):
        """
        player: BLACK or WHITE (which side this agent plays)
        max_depth: search depth in plies
        """
        self.player = player
        self.opponent = BLACK if player == WHITE else WHITE
        self.max_depth = max_depth

    # ---------- Public API ----------

    def get_move(self, grid):
        """
        Pick the best move for self.player on the given board.
        grid: 2D numpy array (15x15), values in {EMPTY, BLACK, WHITE}.
        Returns (x, y) or None if no moves.
        """
        # --- enforce opening in the center ---
        if not np.any(grid != EMPTY):  # board is completely empty
            size = grid.shape[0]
            c = size // 2
            return (c, c)
        # -------------------------------------

        # --- 1) Immediate tactical win if possible ---
        for (x, y) in self._generate_candidate_moves(grid):
            grid[y, x] = self.player
            if self._has_five(grid, self.player):
                grid[y, x] = EMPTY
                return (x, y)
            grid[y, x] = EMPTY

        # --- 2) Block opponent's immediate win if possible ---
        for (x, y) in self._generate_candidate_moves(grid):
            grid[y, x] = self.opponent
            if self._has_five(grid, self.opponent):
                grid[y, x] = EMPTY
                return (x, y)
            grid[y, x] = EMPTY

        # --- 3) Otherwise, run minimax search ---
        best_score = -math.inf
        best_move = None

        for (x, y) in self._generate_candidate_moves(grid):
            grid[y, x] = self.player
            score = self._min_value(
                grid,
                depth=1,
                alpha=-math.inf,
                beta=math.inf
            )
            grid[y, x] = EMPTY

            if score > best_score:
                best_score = score
                best_move = (x, y)

        return best_move

    # alias so it works with Game.run() (agent.getAction)
    def getAction(self, grid):
        return self.get_move(grid)

    # ---------- Minimax core ----------

    def _max_value(self, grid, depth, alpha, beta):
        if self._is_terminal(grid) or depth == self.max_depth:
            return self._evaluate(grid)

        value = -math.inf
        for (x, y) in self._generate_candidate_moves(grid):
            grid[y, x] = self.player
            value = max(value, self._min_value(grid, depth + 1, alpha, beta))
            grid[y, x] = EMPTY

            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value

    def _min_value(self, grid, depth, alpha, beta):
        if self._is_terminal(grid) or depth == self.max_depth:
            return self._evaluate(grid)

        value = math.inf
        for (x, y) in self._generate_candidate_moves(grid):
            grid[y, x] = self.opponent
            value = min(value, self._max_value(grid, depth + 1, alpha, beta))
            grid[y, x] = EMPTY

            if value <= alpha:
                return value
            beta = min(beta, value)
        return value

    # ---------- Helpers: terminal, candidates, evaluation ----------

    def _is_terminal(self, grid):
        """
        True if someone has 5 in a row or board is full.
        """
        return (
            self._has_five(grid, self.player)
            or self._has_five(grid, self.opponent)
            or not (grid == EMPTY).any()
        )

    def _generate_candidate_moves(self, grid):
        """
        Candidate generation:
        - If board empty -> play center.
        - Else: all empty cells within distance 2 of any stone.
        Then sort by distance to center so moves look more reasonable.
        """
        size = grid.shape[0]
        stones = list(zip(*np.where(grid != EMPTY)))
        if not stones:
            center = size // 2
            return [(center, center)]

        candidates = set()
        for (y, x) in stones:
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    nx, ny = x + dx, y + dy
                    if (
                        0 <= nx < size
                        and 0 <= ny < size
                        and grid[ny, nx] == EMPTY
                    ):
                        candidates.add((nx, ny))

        if not candidates:  # fallback
            empties = list(zip(*np.where(grid == EMPTY)))
            return empties

        center = size // 2
        # sort by Manhattan distance to board center to make play more "sensible"
        return sorted(
            candidates,
            key=lambda p: abs(p[0] - center) + abs(p[1] - center),
        )

    def _has_five(self, grid, player):
        """
        Check if player has any five-in-a-row on the board.
        """
        size = grid.shape[0]

        # horizontal
        for y in range(size):
            for x in range(size - 4):
                if np.all(grid[y, x:x + 5] == player):
                    return True

        # vertical
        for x in range(size):
            for y in range(size - 4):
                if np.all(grid[y:y + 5, x] == player):
                    return True

        # diag down-right
        for y in range(size - 4):
            for x in range(size - 4):
                if all(grid[y + k, x + k] == player for k in range(5)):
                    return True

        # diag up-right
        for y in range(4, size):
            for x in range(size - 4):
                if all(grid[y - k, x + k] == player for k in range(5)):
                    return True

        return False

    # ---------- Heuristic evaluation ----------

    def _evaluate(self, grid):
        """
        Static evaluation function.
        Positive = good for self.player, negative = good for opponent.
        """
        if self._has_five(grid, self.player):
            return 10_000_000
        if self._has_five(grid, self.opponent):
            return -10_000_000

        score_self = self._score_player(grid, self.player)
        score_opp = self._score_player(grid, self.opponent)
        return score_self - score_opp

    def _score_player(self, grid, player):
        score = 0
        lines = self._get_all_lines(grid)
        for line in lines:
            score += self._score_line(line, player)
        return score

    def _get_all_lines(self, grid):
        """
        Collect all rows, columns, and both diagonal directions as 1D arrays.
        """
        size = grid.shape[0]
        lines = []

        # rows & columns
        for y in range(size):
            lines.append(grid[y, :])
        for x in range(size):
            lines.append(grid[:, x])

        # diag down-right
        for d in range(-size + 5, size - 4):
            diag = np.diagonal(grid, offset=d)
            if len(diag) >= 5:
                lines.append(diag)

        # diag up-right: flip vertically then take diagonals
        flipped = np.flipud(grid)
        for d in range(-size + 5, size - 4):
            diag = np.diagonal(flipped, offset=d)
            if len(diag) >= 5:
                lines.append(diag)

        return lines

    def _score_line(self, line, player):
        """
        Score a single 1D line for 'player' using simple pattern counts.
        """
        line = np.array(line, dtype=int)
        n = len(line)
        if n < 5:
            return 0

        score = 0
        w_two = 10
        w_three = 100
        w_four = 1000

        enemy = BLACK if player == WHITE else WHITE

        for i in range(n - 4):
            window = line[i:i + 5]

            # ignore mixed windows that contain both player and enemy
            if np.any(window == enemy) and np.any(window == player):
                continue

            stones = np.count_nonzero(window == player)
            empties = np.count_nonzero(window == EMPTY)

            if stones == 4 and empties == 1:
                score += w_four
            elif stones == 3 and empties == 2:
                score += w_three
            elif stones == 2 and empties == 3:
                score += w_two

        return score
