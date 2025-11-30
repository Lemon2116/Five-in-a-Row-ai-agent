# five_in_a_row.py
# Playable Five-in-a-Row (15x15) with mouse clicks (pygame)
# Requirements: pygame, numpy
#
# Controls:
#  - Left mouse click on an intersection -> place stone (if empty)
#  - 'r' -> reset board
#  - 'u' -> undo last move
#  - 'q' or close window -> quit

import pygame
import sys
import numpy as np

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

    def is_full(self):
        return not (self.grid == EMPTY).any()

    def check_win(self, x, y):
        """
        After a move at (x,y), check if that move created a win.
        Returns player id (BLACK/WHITE) if win, else 0.
        Efficient: only checks lines through (x,y).
        """
        player = int(self.grid[y, x])
        if player == EMPTY:
            return 0

        # 4 directional vectors: horizontal, vertical, diag down-right, diag up-right
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dx, dy in directions:
            count = 1  # count the stone at (x,y)
            # check forward direction
            nx, ny = x + dx, y + dy
            while 0 <= nx < self.size and 0 <= ny < self.size and self.grid[ny, nx] == player:
                count += 1
                nx += dx
                ny += dy
            # check backward direction
            nx, ny = x - dx, y - dy
            while 0 <= nx < self.size and 0 <= ny < self.size and self.grid[ny, nx] == player:
                count += 1
                nx -= dx
                ny -= dy

            if count >= 5:
                return player
        return 0

class Game:
    def __init__(self, size=SIZE, window_size=WINDOW_SIZE, margin=MARGIN):
        pygame.init()
        pygame.display.set_caption("Five-in-a-Row - Click intersections to place stones")
        self.screen = pygame.display.set_mode((window_size, window_size))
        self.clock = pygame.time.Clock()

        self.size = size
        self.margin = margin
        self.window_size = window_size

        # compute cell spacing so grid fits nicely
        available = window_size - 2 * margin
        self.cell = available / (size - 1)  # spacing between intersections
        self.stone_radius = int(self.cell * 0.4)  # stone radius in pixels
        self.click_tolerance = int(self.cell * 0.5)  # tolerance to accept a click near intersection

        self.board = Board(size=size)
        self.current_player = BLACK
        self.winner = 0
        self.running = True

    def pixel_to_grid(self, px, py):
        """
        Convert pixel (px,py) to the nearest grid intersection (x,y).
        Return (x,y) if within tolerance, else None.
        """
        # compute nearest grid index
        # grid line i pixel coordinate: margin + i * cell
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
            # vertical line
            pygame.draw.line(self.screen, LINE_COLOR, (x, self.margin), (x, self.window_size - self.margin), 2)
            # horizontal line
            pygame.draw.line(self.screen, LINE_COLOR, (self.margin, y), (self.window_size - self.margin, y), 2)

        # draw star points (optional, typical at certain intersections)
        star_points = []
        if self.size >= 15:
            # common star point positions for 15x15: 4, 7, 10 (0-based indices)
            pts = [3, 7, 11] if self.size == 15 else [self.size // 2]
            for rx in pts:
                for ry in pts:
                    px = int(self.margin + rx * self.cell)
                    py = int(self.margin + ry * self.cell)
                    star_points.append((px, py))
            for (px, py) in star_points:
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
                        # add a thin border to white stones
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

    def handle_mouse_down(self, pos):
        if self.winner != 0:
            return  # no more moves after a win
        g = self.pixel_to_grid(*pos)
        if g is None:
            return
        x, y = g
        if self.board.place(x, y, self.current_player):
            # check win
            winner = self.board.check_win(x, y)
            if winner != 0:
                self.winner = winner
            else:
                # toggle player
                self.current_player = WHITE if self.current_player == BLACK else BLACK

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

    def run(self):
        while self.running:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # left click
                    self.handle_mouse_down(event.pos)
                elif event.type == pygame.KEYDOWN:
                    self.handle_key(event.key)
            self.draw()
        pygame.quit()
        sys.exit()

