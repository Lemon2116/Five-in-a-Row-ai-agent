# five_in_a_row.py (fixed)
import pygame
import sys
import numpy as np
import random

# Board constants
SIZE = 15                 # intersections per side
EMPTY = 0
BLACK = 1
WHITE = 2

# GUI constants
WINDOW_SIZE = 720         # pixels (square window)
MARGIN = 40               # margin from window edge to first grid line
LINE_COLOR = (90, 60, 30) # dark brown for grid lines
BG_COLOR = (205, 170, 125) # light brown board color
BLACK_COLOR = (20, 20, 20)
WHITE_COLOR = (230, 230, 230)
HIGHLIGHT_COLOR = (200, 30, 30)  # last move marker
FPS = 60

class Board:
    """
    Board representation and basic operations.
    grid[y,x] with 0 <= x,y < SIZE
    (The physical board state and history)
    """
    def __init__(self, size=SIZE):
        self.size = size
        self.grid = np.zeros((size, size), dtype=np.int8)
        self.move_history = []  # list of (x,y,player)

    def reset(self):
        self.grid.fill(EMPTY)
        self.move_history.clear()

    def place(self, x, y, player):
        """Place a stone at (x,y) for player (BLACK/WHITE). Returns True on success."""
        if not (0 <= x < self.size and 0 <= y < self.size):
            return False
        if self.grid[y, x] != EMPTY:
            return False
        self.grid[y, x] = player
        self.move_history.append((x, y, player))
        return True

    def undo(self):
        """Undo last move. Returns (x,y,player) or None."""
        if not self.move_history:
            return None
        x, y, player = self.move_history.pop()
        self.grid[y, x] = EMPTY
        return (x, y, player)

# --- GameState for AI Search / Terminal Checks ---
class GameState:
    """A minimal state representation for the AI search tree.
    Coordinate convention used throughout: (x, y) == (col, row).
    'board' here is a NumPy 2D array accessed as board[y, x].
    """
    def __init__(self, board, current_player, last_move=None):
        # 'board' here is the NumPy grid array (shape: size x size)
        self.board = board
        self.current_player = current_player  # player to move next
        # last_move stored as (x, y) OR None, representing the move that produced this state
        self.last_move = last_move
        self.size = board.shape[0]

    def get_legal_actions(self):
        """
        Return empty positions (x, y) that are within Manhattan distance
        <= 3 of ANY existing stone.
        Much smaller action space → Expectimax becomes feasible.
        """

        board = self.board
        size = board.shape[0]
        stones = np.argwhere(board != EMPTY)

        # Case 1: empty board → only center
        if stones.shape[0] == 0:
            mid = size // 2
            return [(mid, mid)]

        stone_y = stones[:, 0]
        stone_x = stones[:, 1]

        legal = []

        # Iterate through ALL empty board cells
        empty_y, empty_x = np.where(board == EMPTY)

        for y, x in zip(empty_y, empty_x):
            # Manhattan distance to nearest stone
            dmin = np.min(np.abs(stone_y - y) + np.abs(stone_x - x))
            if dmin <= 2:
                legal.append((x, y))  # (x, y) = (col, row)

        # Fallback: if no filtered moves (rare)
        if len(legal) == 0:
            return list(zip(empty_x.tolist(), empty_y.tolist()))

        return legal

    def generate_successor(self, action, player_color):
        """Returns a new GameState after placing a stone at action (x, y)."""
        if action is None:
            raise ValueError("generate_successor called with None action")
        x, y = action  # action is (x, y)
        # copy board and place at [y, x]
        new_board = self.board.copy()
        new_board[y, x] = player_color

        # next player is opponent
        next_player = WHITE if player_color == BLACK else BLACK
        return GameState(new_board, next_player, last_move=(x, y))

    def is_terminal(self):
        """
        True if the LAST MOVE resulted in a win, or if the board is full.
        Assumes self.last_move is set to the move that led to this state.
        """
        if self.last_move is not None:
            x, y = self.last_move
            # The player who just moved is the opposite of the current_player
            last_mover = WHITE if self.current_player == BLACK else BLACK
            if self.check_win(x, y, board_grid=self.board, player=last_mover):
                 return True

        # board full -> terminal (draw)
        return not (self.board == EMPTY).any()

    def check_win(self, x, y, board_grid=None, player=None):
        """
        After a move at (x,y), check if that move created a win.
        Uses (x,y) == (col,row) convention: index grid[y, x].
        Returns player id (BLACK/WHITE) if win, else 0.
        """
        grid = board_grid if board_grid is not None else self.board

        # Determine the player who just moved at (x,y)
        player = player if player is not None else int(grid[y, x])
        if player == EMPTY:
            return 0

        # 4 directional vectors: horizontal, vertical, diag down-right, diag up-right
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dx, dy in directions:
            count = 1  # count the stone at (x,y)
            # check forward direction
            nx, ny = x + dx, y + dy
            while 0 <= nx < self.size and 0 <= ny < self.size and grid[ny, nx] == player:
                count += 1
                nx += dx
                ny += dy
            # check backward direction
            nx, ny = x - dx, y - dy
            while 0 <= nx < self.size and 0 <= ny < self.size and grid[ny, nx] == player:
                count += 1
                nx -= dx
                ny -= dy

            if count >= 5:
                return player
        return 0

class EvaluationFunction:
    def __init__(self, agent_color):
        self.agent_color = agent_color
        self.opponent_color = WHITE if agent_color == BLACK else BLACK
        self.size = SIZE

        # Strong defensive bias
        self.SELF_5 = 1e8
        self.OPP_5 = -1e10
        
        self.SELF_4 = 5e5
        self.OPP_4 = -5e7
        
        self.SELF_3 = 5e4
        self.OPP_3 = -5e5
        
        self.SELF_2 = 2e3
        self.OPP_2 = -4e3

    def evaluate(self, board, player_color):
        score = 0
        directions = [(1,0), (0,1), (1,1), (1,-1)]

        for y in range(self.size):
            for x in range(self.size):
                color = board[y][x]
                if color == EMPTY:
                    continue

                for dx, dy in directions:
                    # ---- SKIP if this is NOT the beginning of a line ----
                    prev_x = x - dx
                    prev_y = y - dy
                    if 0 <= prev_x < self.size and 0 <= prev_y < self.size:
                        if board[prev_y][prev_x] == color:
                            continue

                    # ---- Count only forward ----
                    line = self._count_line(board, x, y, dx, dy, color)

                    if color == player_color:
                        if line >= 5:  score += self.SELF_5
                        elif line == 4: score += self.SELF_4
                        elif line == 3: score += self.SELF_3
                        elif line == 2: score += self.SELF_2

                    else:   # opponent
                        if line >= 5:  score += self.OPP_5
                        elif line == 4: score += self.OPP_4
                        elif line == 3: score += self.OPP_3
                        elif line == 2: score += self.OPP_2

        return score

    def _count_line(self, board, x, y, dx, dy, color):
        count = 1
        i = 1
        while True:
            nx = x + dx * i
            ny = y + dy * i
            if 0 <= nx < self.size and 0 <= ny < self.size and board[ny][nx] == color:
                count += 1
                i += 1
            else:
                break
        return count


class Game:
    def __init__(self, size=SIZE, window_size=WINDOW_SIZE, margin=MARGIN,
                 human=True, npc=None):
        pygame.init()
        pygame.display.set_caption("Five-in-a-Row + AI mode")

        self.screen = pygame.display.set_mode((window_size, window_size))
        self.clock = pygame.time.Clock()

        self.size = size
        self.margin = margin
        self.window_size = window_size

        self.human = human        # True = Human plays (Black by default)
        self.npc = npc            # None / one agent / [agent1, agent2]

        # support multiple AI modes
        self.ai_vs_ai = isinstance(npc, list) and len(npc) == 2
        self.one_ai = (npc is not None) and not self.ai_vs_ai

        available = window_size - 2 * margin
        self.cell = available / (size - 1)
        self.stone_radius = int(self.cell * 0.4)
        self.click_tolerance = int(self.cell * 0.5)

        self.board = Board(size=size)
        self.current_player = BLACK
        self.winner = 0
        self.running = True

        # Optionally enforce initial center placement (if you want agent/human to always start center)
        # If desired, uncomment:
        # cx = cy = size // 2
        # self.board.place(cx, cy, BLACK)
        # self.current_player = WHITE

    def pixel_to_grid(self, px, py):
        fx = (px - self.margin) / self.cell
        fy = (py - self.margin) / self.cell
        gx = int(round(fx))
        gy = int(round(fy))
        if 0 <= gx < self.size and 0 <= gy < self.size:
            grid_x_pix = self.margin + gx * self.cell
            grid_y_pix = self.margin + gy * self.cell
            dist_sq = (grid_x_pix - px) ** 2 + (grid_y_pix - py) ** 2
            if dist_sq <= self.click_tolerance ** 2:
                return gx, gy
        return None

    def draw(self):
        # background
        self.screen.fill(BG_COLOR)

        # draw grid lines
        for i in range(self.size):
            x = int(self.margin + i * self.cell)
            y = int(self.margin + i * self.cell)
            pygame.draw.line(self.screen, LINE_COLOR, (x, self.margin), (x, self.window_size - self.margin), 2)
            pygame.draw.line(self.screen, LINE_COLOR, (self.margin, y), (self.window_size - self.margin, y), 2)

        # draw star points
        if self.size >= 15:
            pts = [3, 7, 11] if self.size == 15 else [self.size // 2]
            for rx in pts:
                for ry in pts:
                    px = int(self.margin + rx * self.cell)
                    py = int(self.margin + ry * self.cell)
                    pygame.draw.circle(self.screen, LINE_COLOR, (px, py), 5)

        # draw stones
        for y in range(self.size):
            for x in range(self.size):
                val = int(self.board.grid[y, x])
                if val != EMPTY:
                    px = int(self.margin + x * self.cell)
                    py = int(self.margin + y * self.cell)
                    if val == BLACK:
                        pygame.draw.circle(self.screen, BLACK_COLOR, (px, py), self.stone_radius)
                    else:
                        pygame.draw.circle(self.screen, WHITE_COLOR, (px, py), self.stone_radius)
                        pygame.draw.circle(self.screen, LINE_COLOR, (px, py), self.stone_radius, 2)

        # highlight last move
        if self.board.move_history:
            lx, ly, lp = self.board.move_history[-1]
            px = int(self.margin + lx * self.cell)
            py = int(self.margin + ly * self.cell)
            pygame.draw.rect(self.screen, HIGHLIGHT_COLOR, (px-6, py-6, 12, 12), 2)

        # draw status text
        font = pygame.font.SysFont(None, 24)
        if self.winner != 0:
            win_text = "Black wins!" if self.winner == BLACK else "White wins!"
            txt = font.render(win_text + "  (press 'r' to restart)", True, (10, 10, 10))
        else:
            turn_text = "Black" if self.current_player == BLACK else "White"
            txt = font.render(f"Turn: {turn_text}   (r reset, u undo, q quit)", True, (10, 10, 10))
        self.screen.blit(txt, (10, 10))

        pygame.display.flip()

    def place_and_check(self, x, y):
        """Places stone, updates history, and checks for a win."""
        if self.board.place(x, y, self.current_player):
            # Create a temporary GameState to check for win
            # Note: GameState.check_win expects (x,y) coordinates (col,row)
            temp_state = GameState(self.board.grid, 
                                   WHITE if self.current_player == BLACK else BLACK,
                                   last_move=(x, y))

            # We pass player who just moved
            winner = temp_state.check_win(x, y, board_grid=self.board.grid, player=self.current_player)

            if winner:
                self.winner = winner
            else:
                self.current_player = WHITE if self.current_player == BLACK else BLACK

    def handle_mouse_down(self, pos):
        if self.winner != 0:
            return  # no more moves after a win
        g = self.pixel_to_grid(*pos)
        if g is None:
            return
        self.place_and_check(*g)

    def handle_key(self, key):
        if key == pygame.K_r:
            self.board.reset()
            self.current_player = BLACK
            self.winner = 0
        elif key == pygame.K_u:
            undone = self.board.undo()
            if undone:
                # if we undo, switch player (since last move was removed)
                self.current_player = WHITE if self.current_player == BLACK else BLACK
                self.winner = 0
        elif key == pygame.K_q:
            self.running = False

    def _agent_choose_fallback(self):
        """Choose a fallback move: center if empty else random legal move."""
        cx = cy = self.size // 2
        if self.board.grid[cy, cx] == EMPTY:
            return (cx, cy)
        rows, cols = np.where(self.board.grid == EMPTY)
        if len(rows) == 0:
            return None
        # return (x, y)
        idx = random.randrange(len(rows))
        return (int(cols[idx]), int(rows[idx]))

    def run(self):
        while self.running:
            self.clock.tick(FPS)

            # ---------------- AI VS AI -----------------
            if self.ai_vs_ai and self.winner == 0:
                agent = self.npc[0] if self.current_player == BLACK else self.npc[1]
                try:
                    move = agent.getAction(self.board.grid)
                except Exception as e:
                    print("Agent getAction threw exception:", e)
                    move = None
                if move is None:
                    move = self._agent_choose_fallback()
                if move:
                    x, y = move
                    self.place_and_check(x, y)
                continue

            # ---------------- HUMAN VS AI --------------
            if self.one_ai and self.current_player == WHITE and self.winner == 0:
                try:
                    move = self.npc.getAction(self.board.grid)
                except Exception as e:
                    print("Agent getAction threw exception:", e)
                    move = None
                if move is None:
                    move = self._agent_choose_fallback()
                if move:
                    x, y = move
                    self.place_and_check(x, y)
                continue

            # ---------------- HUMAN INPUT --------------
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.human or self.current_player == BLACK:
                        g = self.pixel_to_grid(*event.pos)
                        if g:
                            self.place_and_check(*g)
                elif event.type == pygame.KEYDOWN:
                    self.handle_key(event.key)

            self.draw()

        pygame.quit()
        sys.exit()

