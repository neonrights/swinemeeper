import os
import copy
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


class PlayerSolver(MinesweeperSolver):
    def __init__(self, shape, mines, start=None, name='player'):
        super(PlayerSolver, self).__init__(shape, mines, start)
        # initialize solver to display changes to grid using image
        # initialize solver to track mouse when clicking on image
        # basically make a working game of minesweeper minus time and flagging


class CSPSolver(MinesweeperSolver):
    def __init__(self, shape, mines, start=None, name='csp'):
        super(CSPSolver, self).__init__(shape, mines, start)
        # initialize variable/probability tracker
        self.probs = np.zeros(shape) + 10 # < 0 for unknown

        self.safe_moves = set() # set of safe spaces for fast moves
        self.known_mines = set() # set of known mines

        # initialize constraint list
        all_variables = set([tuple(index) for index in np.ndindex(shape)])
        self.constraints = [[all_variables, mines]]
        # add dictionary for faster constraint pruning
        #self.var_dict = dict()
        #for var in all_variables:
        #    self.var_dict[var] = set([copy.copy(self.constraints[0])])

        if start:
            self._add_constraint(start)

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
            self.constraints.append([constraint_vars, constraint_val])
            return # constraint not resolved, add to list

        # continue while there are still variables to prune
        while not prune_mines and not prune_safe:
            new_safe = set()
            new_mines = set()
            for i in range(len(self.constraints)):
                self.constraints[i][0] = self.constraints[i][0].difference(prune_safe)
                self.constraints[i][1] -= len(self.constraints[i][0].intersection(prune_mines))
                self.constraints[i][0] = self.constraints[i][0].difference(prune_mines)

                if not self.constraints[i][0]:
                    assert self.constraints[i][1] == 0
                    del self.constraints[i] # empty constraint
                    continue

                # if constraint is satisfied, add new variables to list
                if self.constraints[i][1] == 0:
                    new_safe = new_safe.union(self.constraints[i][0])
                    del self.constraints[i]
                elif self.constraints[i][0]) == self.constraints[i][1]:
                    new_mines = new_mines.union(self.constraints[i][0])
                    self.known_mines = self.known_mines.union(self.constraints[i][0])
                    del self.constraints[i]

            prune_mines = new_mines
            prune_safe = new_safe

    def _calculate_probabilities(self):
        raise NotImplementedError
        # calculate the probability of a space containing a mine using constraint list
        # exception for first constraint, will stay until game is over
        #   calculation is easy as it is val / |vars|
        # figure out how to calculate probs for other constraints
        # get unknown variables
        # perform dfs on unknown using constraint list as conditional

    def _satisfies_constraints(self, vars, vals):
        raise NotImplementedError # returns true if variable-value pairs satisfy constraints

    def act(self):
        # view constraint list, determine which space to choose
        if self.safe.isEmpty():
            self._calculate_probabilities()
            # return position of variable with min prob
        else:
            pos = self.safe.pop()

        val = self.board.reveal(pos)
        assert val >= 0
        self._add_constraint(pos)


def test_cps():
    raise NotImplementedError
    # test initialization
    # test constraint addition
    # test probability calculation
    # test act/decision function


if __name__ == '__main__':
    test_cps()
