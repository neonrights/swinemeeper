import random
import numpy as np


class MinesweeperState:
    def __init__(self, shape, mines, start=None):
        self.revealed = np.zeros(shape, dtype=bool)
        self.values = np.zeros(shape, dtype=np.int8)
        self.mines = mines

        
        indices = [tuple(index) for index in np.ndindex(shape)]
        if start:
            self.revealed[start] = True
            indices.remove(start)

        assert mines < len(indices)
        indices = random.sample(indices, mines)
        for index in indices:
            self.values[index] = -1

        for index in indices:
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    neighbor = (index[0] + i, index[1] + j)
                    if index != neighbor and 0 <= neighbor[0] < shape[0] and 0 <= neighbor[1] < shape[1]:
                        if self.values[neighbor] >= 0:
                            self.values[neighbor] += 1

    def reveal(self, x, y):
        self.revealed[(x,y)] = True
        return self.values[(x,y)]

    def is_goal(self):
        return (not self.revealed).sum() == self.mines

    def to_image(self, path):
        raise NotImplementedError # output board state as image


if __name__ == '__main__':
    shape = (10, 10)
    test1 = MinesweeperState(shape, 40, (5,5))
    pdb.set_trace()

