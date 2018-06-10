import random
import numpy as np

from PIL import Image, ImageDraw

CELL = 16
BORDER = 10
WIDTH = 2
COLORS = ["#0000FF", "#008200", "#FF0000", "#000084", "#840000", "#008284", "#840084", "#000000"]

class GameOver(Exception):
    pass

# config for initial state and whether to render board
class MinesweeperState:
    def __init__(self, shape, mines, start=None, render=False, seed=None):
        self.covered = np.ones(shape, dtype=bool)
        self.adjacent_mines = np.zeros(shape, dtype=np.int8)
        self.shape = shape
        self.mine_count = mines
        self.move = 0
        self._loss = False

        # create seed
        if seed:
            np.random.seed(seed)
            random.seed(seed)

        indices = [tuple(index) for index in np.ndindex(shape)]
        if start:
            if start is True:
                start = (random.randint(0, shape[0]-1), random.randint(0, shape[1]-1))

            self.covered[start] = False
            indices.remove(start)
            self.move += 1

        # initialize mines
        assert mines < len(indices)
        indices = random.sample(indices, mines)
        for index in indices:
            self.adjacent_mines[index] = -1

        # create neighboring mines
        for index in indices:
            for neighbor in get_neighbors(index, shape):
                if self.adjacent_mines[neighbor] >= 0:
                    self.adjacent_mines[neighbor] += 1

        # create graphics for board state
        self.render = render
        if self.render:
            self._init_image(shape)
            if start:
                self._draw_cell(start)


    def _init_image(self, shape):
        # create image and draw background
        image_shape = (CELL*shape[0] + 2*BORDER, CELL*shape[1] + 2*BORDER)
        self.image = Image.new('RGB', image_shape, color="#BDBDBD")
        self.drawer = ImageDraw.Draw(self.image)

        # draw covered cells
        for i in range(shape[0]):
            white_path = (CELL*i + BORDER + WIDTH, BORDER, CELL*i + BORDER + WIDTH, CELL * shape[1] + BORDER)
            self.drawer.line(white_path, fill="#FFFFFF", width=WIDTH)

        for j in range(shape[1]):
            white_path = (BORDER, CELL*j + BORDER + WIDTH, CELL * shape[0] + BORDER, CELL*j + BORDER + WIDTH)
            self.drawer.line(white_path, fill="#FFFFFF", width=WIDTH)

        for i in range(shape[0]):
            black_path = (CELL*i + BORDER, BORDER, CELL*i + BORDER, CELL * shape[1] + BORDER)
            self.drawer.line(black_path, fill="#7B7B7B", width=WIDTH)

        black_path = (CELL*shape[0] + BORDER, BORDER, CELL*shape[0] + BORDER, CELL * shape[1] + BORDER)
        self.drawer.line(black_path, fill="#7B7B7B", width=WIDTH)

        for j in range(shape[0]):
            black_path = (BORDER, CELL*j + BORDER, CELL * shape[0] + BORDER, CELL*j + BORDER)
            self.drawer.line(black_path, fill="#7B7B7B", width=WIDTH)

        black_path = (BORDER, CELL*shape[1] + BORDER, CELL * shape[0] + BORDER, CELL*shape[1] + BORDER)
        self.drawer.line(black_path, fill="#7B7B7B", width=WIDTH)


    def reveal(self, pos):
        if self._loss or self.is_goal():
            raise GameOver

        assert self.covered[pos], "chosen position has already been uncovered"
        self.move += 1
        if self.render:
            self._draw_cell(pos) # alter image

        self.covered[pos] = False
        if self.adjacent_mines[pos] < 0:
            self._loss = True

        return self.adjacent_mines[pos]


    def is_goal(self):
        return (self.covered.sum() == self.mine_count) and not self._loss


    def _draw_cell(self, pos):
        # draw covered cell
        cell_border = (CELL*pos[0] + BORDER + WIDTH, CELL*pos[1] + BORDER + WIDTH, CELL*pos[0] + BORDER + CELL, CELL*pos[1] + BORDER + CELL)
        self.drawer.rectangle(cell_border, fill="#BDBDBD")
        val = self.adjacent_mines[pos]
        if val > 0:
            text_size = self.drawer.textsize(str(val))
            text_anchor = (cell_border[0] + (CELL - text_size[0]) / 2, cell_border[1] + (CELL - text_size[1]) / 2)
            self.drawer.text(text_anchor, str(val), fill=COLORS[val - 1], align="center")
        elif val < 0:
            # draw a nice mine
            self.drawer.rectangle(cell_border, fill="#FF0000")
            mine_border = (cell_border[0] + 3, cell_border[1] + 3, cell_border[2] - 3, cell_border[3] - 3)
            self.drawer.ellipse(mine_border, fill="#000000")
            self.drawer.line(mine_border, fill="#000000")
            self.drawer.line((mine_border[0], mine_border[1] + 8, mine_border[2], mine_border[3] - 8), fill="#000000")
            self.drawer.line((mine_border[0] - 2, mine_border[1] + 4, mine_border[2] + 2, mine_border[3] - 4), fill="#000000")
            self.drawer.line((mine_border[0] + 4, mine_border[1] - 2, mine_border[2] - 4, mine_border[3] + 2), fill="#000000")
            self.drawer.rectangle((cell_border[0] + 5, cell_border[1] + 5, cell_border[0] + 6, cell_border[1] + 6), fill="#FFFFFF")


    def to_image(self, dest=None):
        if not self.render:
            raise RuntimeError("Graphics rendering option is disabled")

        if dest is None:
            self.image.save("board_%d.png" % self.move)
        else:
            self.image.save(dest)


def get_neighbors(pos, shape):
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            neighbor = (pos[0] + i, pos[1] + j)
            if 0 <= neighbor[0] < shape[0] and 0 <= neighbor[1] < shape[1]:
                if neighbor != pos:
                    yield neighbor


def compute_adjacent_mines(mine_positions):
    mine_positions = mine_positions.astype(bool)

    adjacent_mines = np.zeros_like(mine_positions, dtype=np.int8)
    adjacent_mines[:-1] += mine_positions[1:]
    adjacent_mines[1:] += mine_positions[:-1]
    adjacent_mines[:,:-1] += mine_positions[:,1:]
    adjacent_mines[:,1:] += mine_positions[:,:-1]
    adjacent_mines[1:,1:] += mine_positions[:-1,:-1]
    adjacent_mines[:-1,:-1] += mine_positions[1:,1:]
    adjacent_mines[:-1,1:] += mine_positions[1:,:-1]
    adjacent_mines[1:,:-1] += mine_positions[:-1,1:]

    for position in zip(*np.where(mine_positions)):
        adjacent_mines[position] = -1

    return adjacent_mines


def draw_state(dest, mine_placement, covered):
    state = MinesweeperState(mine_placement.shape, (mine_placement == -1).sum(), render=True)
    state.adjacent_mines = mine_placement
    
    for position in zip(*np.where(~covered.T)):
        state._draw_cell(position)

    state.to_image(dest=dest)


def test_state():
    shape = (10, 8)
    test = MinesweeperState(shape, 10, render=True)
    test.to_image('test_image_init.png')
    assert (test.adjacent_mines < 0).sum() == 10

    test = MinesweeperState(shape, 25, start=(5,4), render=True)
    test.to_image('test_image_reveal.png')
    print(test.adjacent_mines.T)
    print(test.adjacent_mines[(5,4)])

    for i in range(5):
        pos = (random.randint(0, shape[0]-1), random.randint(0, shape[1]-1))
        while not test.covered[pos]:
            pos = (random.randint(0, shape[0]-1), random.randint(0, shape[1]-1))
        
        val = test.reveal(pos)
        test.to_image("test_image_%d.png" % (i + 1))
        print(test.covered.T)
        print(val)


if __name__ == '__main__':
    test_state()
