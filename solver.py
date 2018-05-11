import os
import collections
import numpy as np

from state import MinesweeperState
import pdb


class MinesweeperSolver(object):
    def __init__(self, shape, mines, start=None, name='solver'):
        self.board = MinesweeperState(shape, mines, start, render=True)
        self.move = 1 if start else 0
        self.name = name
        # other shared variables, like remaining mines

    def act(self):
        # determine which space to click on
        # determine if game s over
        raise NotImplementedError

    def save_state(self):
        self.board.to_image("board_%d.png" % self.move)


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
        all_variables = set(tuple(index) for index in np.ndindex(shape))
        self.constraints = [[all_variables, mines]]
        # add dictionary for faster constraint pruning
        self.var_dict = dict()
        for var in all_variables:
            self.var_dict[var] = [self.constraints[0]]

        if start:
            self._add_constraint(start)

    def _add_constraint(self, position):
        # determine unknown neighbors
        constraint_vars = set()
        assert not self.board.covered[position]
        constraint_val = self.board.adjacent_mines[position]
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                neighbor = (position[0] + i, position[1] + j)
                if neighbor != position:
                    try:
                        if self.board.covered[neighbor]:
                            constraint_vars.add(neighbor)
                    except IndexError:
                        continue

        # remove known mines from constraint, update constraint
        constraint_val -= len(constraint_vars.intersection(self.known_mines))
        constraint_vars = constraint_vars.difference(self.known_mines)
        assert constraint_val >= 0

        new_constraints = collections.deque()
        new_constraints.append(set([position]), 0) # prune newly revealed space
        if constraint_vars:
            new_constraints.append([constraint_vars, constraint_val]) # prune 
            if constraint_val == 0:
                self.save_moves = self.safe_moves.union(constraint_vars)
            elif len(constraint_vars) == constraint_val:
                self.known_mines = self.known_mines.union(constraint_vars)
        
        # continue while there are still newly formed constraints
        while not new_constraints:
            constraint_vars, constraint_val = new_constraints.popleft()
            for i in range(len(self.constraints)):
                if constraint_val == 0:
                    # resolved constraint, all safe
                    self.constraints[i][0] = self.constraints[i][0].difference(constraint_vars)
                elif len(constraint_vars) == constraint_val:
                    # resolved constraint, all mines
                    self.constraints[i][1] -= len(self.constraints[i][0].intersection(constraint_vars))
                    self.constraints[i][0] = self.constraints[i][0].difference(constraint_vars)
                else:
                    # unresolved constraint
                    if self.constraints[i][0].issuperset(constraint_vars):
                        # new constraint is subset of old constraint
                        self.constraints[i][0] = self.constraints[i][0].difference(constraint_vars)
                        self.constraints[i][1] -= constraint_val
                    elif self.constraints[i][0].issubset(constraint_vars):
                        # old constraint is subset of new constraint
                        new_vars = constraint_vars.difference(self.constraints[i][0])
                        new_val = constraint_val - self.constraints[i][1]
                        new_constraints.append(new_vars, new_val)
                        # edit constraint variables, see if resolved
                        # if resolved, add to resolved list and break?
                        # if not continue? restart? with new constraints

                if not self.constraints[i][0]:
                    assert self.constraints[i][1] == 0
                    del self.constraints[i] # empty constraint, remove
                    continue

                # if constraint is resolved, add new variables to list
                if self.constraints[i][1] == 0:
                    new_constraints.append(self.constraints[i])
                    del self.constraints[i]
                elif self.constraints[i][0] == self.constraints[i][1]:
                    new_constraints.append(self.constraints[i])
                    self.known_mines = self.known_mines.union(self.constraints[i][0])
                    del self.constraints[i]

            if constraint_val == 0 or len(constraint_vars) == constraint_val:
                # add constraint if not resolved, not altered, and not repeat
                self.constraints.append([constraint_vars, constraint_val])

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
    test_solver = CSPSolver((20, 16), 100)
    assert test_solver.constraints[0][1] == 100
    assert len(test_solver.constraints[0][0]) == 20*16
    test_solver.save_state()

    test_solver = CSPSolver((20, 16), 100, start=(5,4))
    test_solver.save_state()
    pdb.set_trace()
    # test initialization
    # test constraint addition
    # test probability calculation
    # test act/decision function


if __name__ == '__main__':
    test_cps()
