# ========================= NPCagent.py ============================
import random
import math
import copy

BOARD_SIZE = 15

class GomokuAgent:
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

class ExpectimaxAgent(GomokuAgent):

    def __init__(self, player_color, depth=2):
        super().__init__(player_color)
        self.depth = depth

    # ----------------- API ENTRY -------------------------
    def getAction(self, board):
        """Return best move chosen via Expectimax with weighted chance nodes."""
        _, move = self.expectimax(board, depth=self.depth, maximizing=True)
        return move

    # =====================================================
    # EXPECTIMAX SEARCH
    # =====================================================
    def expectimax(self, board, depth, maximizing):
        winner = self.check_win_state(board)
        if winner or depth == 0:
            return self.evaluate(board), None

        legal_moves = self.nearby_weighted_moves(board)

        if maximizing:
            best_score = -math.inf
            best_move = None

            for _,move in legal_moves:
                next_board = self.simulate_move(board, move, self.color)
                score,_ = self.expectimax(next_board, depth-1, maximizing=False)
                if score > best_score:
                    best_score, best_move = score, move
            return best_score, best_move

        else:
            # EXPECTATION NODE (Opponent)
            values = []
            for prob, move in legal_moves:
                opponent = "black" if self.color=="white" else "white"
                next_board = self.simulate_move(board, move, opponent)
                score,_ = self.expectimax(next_board, depth-1, maximizing=True)
                values.append(prob * score)
            return sum(values), None


    # ================= GAME STATE UTILS =================
    def simulate_move(self, board, move, color):
        new_board = copy.deepcopy(board)
        r,c = move
        new_board[r][c] = color
        return new_board


    def evaluate(self, board):
        """
        Heuristic evaluation for win/loss, number of open threats, etc.
        You CAN replace this with stronger scoring later.
        """
        winner = self.check_win_state(board)
        if winner == self.color: return 99999
        elif winner is not None: return -99999
        else:
            # Simple score: number of 2|3|4-length chains
            return self.heuristic_score(board)


    def heuristic_score(self, board):
        score = 0
        # You should later implement pattern-based scoring (OPEN-3, OPEN-4, etc.)
        return score


    # -------- INLINE win check used by agent ------------
    def check_win_state(self, board):
        directions = [(1,0),(0,1),(1,1),(1,-1)]
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r][c]:
                    color = board[r][c]
                    for dr,dc in directions:
                        count=1
                        for i in range(1,5):
                            nr,nc=r+dr*i,c+dc*i
                            if 0<=nr<BOARD_SIZE and 0<=nc<BOARD_SIZE and board[nr][nc]==color:
                                count+=1
                            else:
                                break
                        if count>=5: return color
        return None
