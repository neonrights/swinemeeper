import os
import numpy as np

from collections import Queue
from state import MinesweeperState


class MinesweeperSolver:
    def __init__(self, shape, mines, start=None, name='solver'):
        self.board = MinesweeperState(shape, mines, start)
        self.move = 1 if start else 0
        self.name = name
        # other shared variables, like remaining mines

    def act(self):
        # determine which space to click on
        # determine if game is over
        raise NotImplementedError

    def save_state(self, path):
        self.board.to_image(os.path.join(path, self.name, "board_%d" $ self.move))


class CSPSolver(MinesweeperSolver):
    def __init__(self, shape, mines, start=None, name='csp'):
        super(self, MinesweeperSolver).__init__(shape, mines, start)
        # initialize variable/probability tracker
        self.probs = -np.ones(shape) # < 0 for unknown
        self.safe = Queue()

        # initialize constraint list
        all_variables = set([tuple(index) for index in np.ndindex(shape)])
        self.constraints = [[all_variables, mines]]
        if start:
            self.variables[start] = 0
            self._add_constraint(start)

    def _calculate_probabilities(self):
        raise NotImplementedError
        # calculate the probability of a space containing a mine using constraint list

    def _add_constraint(self, var):
        raise NotImplementedError
        # determine unknown neighbors
        # if 0, label all as known 0 add to safe queue
        # if equal to neighbors, label all as known mines
        # in both cases prune constraint list of known variables

    def act(self):
        # view constraint list, determine which space to choose
        if self.safe.isEmpty():
            raise NotImplementedError
        else:
            pos = self.safe.pop()
            val = self.board.reveal(pos)
            assert val >= 0
            self._add_constraint(pos)
 
