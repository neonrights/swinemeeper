import random
import numpy as np

from PIL import Image, ImageDraw

CELL = 16
BORDER = 10
WIDTH = 2

class MinesweeperState:
    def __init__(self, shape, mines, start=None):
        self.covered = np.ones(shape, dtype=bool)
        self.adjacent_mines = np.zeros(shape, dtype=np.int8)
        self.mine_count = mines
        self._init_image(shape)

        indices = [tuple(index) for index in np.ndindex(shape)]
        if start:
            self.covered[start] = False
            indices.remove(start)
            self._draw_cell(start)

        # initialize mines
        assert mines < len(indices)
        indices = random.sample(indices, mines)
        for index in indices:
            self.adjacent_mines[index] = -1

        # create neighboring mines
        for index in indices:
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    neighbor = (index[0] + i, index[1] + j)
                    if index != neighbor:
                        try:
                            if self.adjacent_mines[neighbor] >= 0:
                                self.adjacent_mines[neighbor] += 1
                        except IndexError:
                            continue

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
        self._draw_cell(pos) # alter image
        self.covered[pos] = False
        return self.neighboring_mines[pos]

    def is_goal(self):
        return self.covered.sum() == self.mine_count

    def _draw_cell(self, pos):
        # draw covered cell
        val = self.neighboring_mines[pos]
        if val > 0:
            pass # draw text
        elif val < 0:
            pass # draw mine

    def to_image(self, dest):
        self.image.save(dest)


def test_state():
    shape = (20, 16)
    test = MinesweeperState(shape, 100)
    test.to_image('test_image_1.png')


if __name__ == '__main__':
    test_state()
