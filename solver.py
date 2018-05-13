import os
import sys
import collections
import numpy as np

from state import MinesweeperState
from debugger import assert_debugger
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
        if not os.path.exists("images"):
            os.mkdir("images")

        self.board.to_image("images/board_%d.png" % self.move)


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
        if self.board.covered[position]:
            pdb.set_trace()

        # determine unknown neighbors
        constraint_vars = set()
        constraint_val = self.board.adjacent_mines[position]
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                neighbor = (position[0] + i, position[1] + j)
                if 0 <= neighbor[0] < self.board.covered.shape[0] and 0 <= neighbor[1] < self.board.covered.shape[1]:
                    if neighbor != position and self.board.covered[neighbor]:
                        constraint_vars.add(neighbor)

        # remove known mines from constraint, update constraint
        constraint_val -= len(constraint_vars.intersection(self.known_mines))
        constraint_vars -= self.known_mines
        assert constraint_val >= 0, "invalid constraint value after pruning known mines"

        new_constraints = collections.deque()
        new_constraints.append([set([position]), 0]) # prune newly revealed space
        if constraint_vars:
            new_constraints.append([constraint_vars, constraint_val]) # prune
        
        # continue while there are still newly formed constraints
        while new_constraints:
            constraint_vars, constraint_val = new_constraints.popleft()
            delete_set = set()
            for i in range(len(self.constraints)):
                if constraint_val == 0:
                    # resolved constraint, all safe
                    self.constraints[i][0] -= constraint_vars
                elif len(constraint_vars) == constraint_val:
                    # resolved constraint, all mines
                    self.constraints[i][1] -= len(self.constraints[i][0].intersection(constraint_vars))
                    self.constraints[i][0] -= constraint_vars
                else:
                    # unresolved constraint
                    if self.constraints[i][0].issuperset(constraint_vars):
                        # new constraint is subset of old constraint
                        self.constraints[i][0] -= constraint_vars
                        self.constraints[i][1] -= constraint_val
                    elif self.constraints[i][0].issubset(constraint_vars):
                        # old constraint is subset of new constraint
                        new_vars = constraint_vars.difference(self.constraints[i][0])
                        new_val = constraint_val - self.constraints[i][1]
                        new_constraints.append([new_vars, new_val])
                        continue # skip remaining? must not add unaltered constraint

                if not self.constraints[i][0]:
                    assert self.constraints[i][1] == 0, "constraint has no variables but value is not 0"
                    delete_set.add(i) # empty constraint, remove

                # if constraint is resolved, add new variables to list
                if self.constraints[i][1] == 0:
                    new_constraints.append(self.constraints[i])
                    self.safe_moves = self.safe_moves.union(self.constraints[i][0])
                    delete_set.add(i)
                elif len(self.constraints[i][0]) == self.constraints[i][1]:
                    new_constraints.append(self.constraints[i])
                    self.known_mines = self.known_mines.union(self.constraints[i][0])
                    delete_set.add(i)

            for i in sorted(delete_set, reverse=True):
                del self.constraints[i]

            # add constraint if not resolved, otherwise add to known mines or safe moves
            if constraint_val == 0:
                for move in constraint_vars:
                    if self.board.covered[move]:
                        self.safe_moves.add(move)
            elif len(constraint_vars) == constraint_val:
                self.known_mines = self.known_mines.union(constraint_vars)
            elif constraint_vars:
                self.constraints.append([constraint_vars, constraint_val])

    def _calculate_probabilities(self):
        raise NotImplementedError
        # get unknown variables
        # split unknown variables into disjoint sets and constraints
        # perform dfs with backtracking on each set of variables and constraints
        # get probabilities, add safe moves if any become available
        # return min probability space

    def _satisfies_constraints(self, vars, vals):
        raise NotImplementedError # returns true if variable-value pairs are satisfied

    def act(self):
        # view constraint list, determine which space to choose
        self.move += 1
        if self.safe_moves:
            pos = self.safe_moves.pop()
        else:
            pos = self._calculate_probabilities()

        val = self.board.reveal(pos)
        if val >= 0:
            self._add_constraint(pos)

        return val


@exception_debugger
def test_cps():
    test_solver = CSPSolver((20, 16), 100)
    assert test_solver.constraints[0][1] == 100
    assert len(test_solver.constraints[0][0]) == 20*16
    test_solver.save_state()

    test_solver = CSPSolver((20, 20), 40, start=(5,4))
    assert len(test_solver.constraints[0][0]) == (20**2 - 9), "improper pruning"
    test_solver.save_state()
    while True:
        try:
            test_solver.act()
            test_solver.save_state()
        except NotImplementedError:
            break
    pdb.set_trace()


if __name__ == '__main__':
    test_cps()
