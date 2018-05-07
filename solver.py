import os
import numpy as np

from state import MinesweeperState


class MinesweeperSolver:
    def __init__(self, shape, mines, start=None, name='solver'):
        self.board = MinesweeperState(shape, mines, start)
        self.move = 1 if start else 0
        self.name = name
        # other shared variables, like remaining mines

    def act(self):
        # determine which space to click on
        # determine if game s over
        raise NotImplementedError

    def save_state(self, path):
        self.board.to_image(os.path.join(path, self.name, "board_%d" $ self.move))


class CSPSolver(MinesweeperSolver):
    def __init__(self, shape, mines, start=None, name='csp'):
        super(self, MinesweeperSolver).__init__(shape, mines, start)
        # initialize variable/probability tracker
        self.probs = -np.ones(shape) # < 0 for unknown
        self.safe_moves = set() # set of safe spaces for fast moves
        self.known_mines = set() # set of known mines

        # initialize constraint list
        all_variables = set([tuple(index) for index in np.ndindex(shape)])
        self.constraints = [[all_variables, mines]]
        if start:
            self.variables[start] = 0
            self._add_constraint(start)

    def _calculate_probabilities(self):
        raise NotImplementedError
        # calculate the probability of a space containing a mine using constraint list

    def _add_constraint(self, position):
        # determine unknown neighbors
        constraint_vars = set()
        assert self.board.revealed[position]
        constraint_val = self.board.neighboring_mines[position]
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                neighbor = (position[0] + i, position[1] + j)
                if neighbor != position:
                    try:
                        if not self.board.revealed[neighbor]:
                            constraint_var.add(neighbor)
                    except IndexError:
                        continue

        if not contraint_vars:
            assert constraint_val == 0
            return # no variables in constraint, no need to add

        # remove known mines from constraint, update constraint
        constraint_val -= len(constraint_vars.intersection(self.known_mines))
        constraint_vars = constraint_vars.difference(self.known_mines)
        assert constraint_val >= 0

        prune_mines = set()
        prune_safe = set()
        if constraint_val == 0:
            # prune safe variables
            prune_safe = constraint_vars
        elif len(constraint_vars) == constraint_val:
            # prune known mines
            prune_mines = constraint_vars
            self.known_mines.union(constraint_vars)
        else:
            self.constraint_list.append([constraint_vars, constraint_val])
            return # constraint not resolved, add to list

        # continue while there are still variables to prune
        while not prune_mines and not prune_safe:
            new_safe = set()
            new_mines = set()
            for i in range(len(self.constraint_list)):
                self.constraint_list[i][0] = self.constraint_list[i][0].difference(prune_safe)
                self.constraint_list[i][1] -= len(self.constraint_list[i][0].intersection(prune_mines))
                self.constraint_list[i][0] = self.constraint_list[i][0].difference(prune_mines)

                if not self.constraint_list[i][0]:
                    assert self.constraint_list[i][1] == 0
                    del self.constraint_list[i] # empty constraint
                    continue

                if self.constraint_list[i][1] == 0:
                    new_safe = new_safe.union(self.constraint_list[i][0])
                    del self.constraint_list[i]
                elif self.constraint_list[i][0]) == self.constraint_list[i][1]:
                    new_mines = new_mines.union(self.constraint_list[i][0])
                    self.known_mines = self.known_mines.union(self.constraint_list[i][0])
                    del self.constraint_list[i]

            prune_mines = new_mines
            prune_safe = new_safe

    def act(self):
        # view constraint list, determine which space to choose
        if self.safe.isEmpty():
            raise NotImplementedError
        else:
            pos = self.safe.pop()
            val = self.board.reveal(pos)
            assert val >= 0
            self._add_constraint(pos)
 
