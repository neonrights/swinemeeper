import operator
import numpy as np

from MinesweeperState import *
from CCCSPSolver import *
from debugger import *


class ExpectedConstraintSolver(CCCSPSolver):
	def _probabilistic_guess(self):
		probabilities = dict() # calculate probabilities

		# get unknown variables
		remaining_variables = set([(pos1, pos2) for pos1, pos2 in zip(*np.where(self.board.covered))])
		remaining_variables -= self.known_mines
		remaining_variables -= self.safe_moves

		# split unknown variables into disjoint sets of variables and constraints
		disjoint_sets = self._get_disjoint_sets(remaining_variables)
		
		for variables, constraints in disjoint_sets:
			# check if single constraint
			if len(constraints) == 1:
				assert constraints[0][1] > 0, "uncaught resolved/invalid constraint"
				# determine if crapshoot, if so, guess immediately
				prob = float(constraints[0][1]) / len(constraints[0][0])
				for var in constraints[0][0]:
					probabilities[var] = prob
			else:
				_, max_constraints = self._get_max_hyper_vars(variables, constraints)

				# use dfs to calculate probabilities
				sums, total = self._constraint_dfs(max_constraints, dict(), 0, list())
				for max_var, val in sums.items():
					set_size = len(max_var)
					for var in max_var:
						probabilities[var] = float(val) / (set_size * total)

		# find 0's and 1's
		for pos, val in probabilities.items():
			if val == 0:
				self.safe_moves.add(pos)
			elif val == 1:
				self.known_mines.add(pos)

		if self.safe_moves:
			guess = self.safe_moves.pop()
		else:
			# calculate expected constraint value for all min probs
			# select min expected constraint
			guess = min(probabilities.items(), key=operator.itemgetter(1))[0]

		return guess, probabilities


def test_reordering():
	pass


if __name__ == '__main__':
	test_reordering()
