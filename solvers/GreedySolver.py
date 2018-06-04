import copy
import operator
import numpy as np

from CCCSPSolver import *
from MinesweeperState import *

# reweights min probabilities by probability of 0 + max neighboring mines
class GreedySolver(CCCSPSolver):
	def __init__(self, board, name='greedy'):
		super(CCCSPSolver, self).__init__(board, name)


	def _valid_constraint(vars, val):
		constraint_list = copy.deepcopy(self.constraints)
		for constraint in constraint_list:
			pass # edit every constraint, see if conflicts arise

		return True


	def _probabilistic_guess(self):
		probabilities = dict() # calculate probabilities
		zero_probs = dict() # probability of position containing 0 or max as constraint value
		max_probs = dict()

		# get unknown variables
		remaining_variables = set([(pos1, pos2) for pos1, pos2 in zip(*np.where(self.board.covered))])
		remaining_variables -= self.known_mines
		remaining_variables -= self.safe_moves

		# split unknown variables into disjoint sets of variables and constraints
		disjoint_sets = self._get_disjoint_sets(set(remaining_variables))
		
		for variables, constraints in disjoint_sets:
			# check if single constraint
			if len(constraints) == 1:
				assert constraints[0][1] > 0, "uncaught resolved/invalid constraint"
				# determine if crapshoot, if so, guess immediately
				prob = float(constraints[0][1]) / len(constraints[0][0])
				for var in constraints[0][0]:
					probabilities[var] = prob
					for neighbor in get_neighbors(var):
						if neighbor in remaining_variables:
							pass # calculate min and max probs assuming valid

			else:
				_, max_constraints = self._get_max_hyper_vars(variables, constraints)

				# use dfs to calculate probabilities
				sums, total = self._constraint_dfs(max_constraints, dict(), 0, list())
				for max_var, val in sums.items():
					set_size = len(max_var)
					for var in max_var:
						probabilities[var] = float(val) / (set_size * total)
						for neighbor in get_neighbors(var):
							if neighbor in remaining_variables:
								pass # calculate min and max probs assuming valid

		# find 0's and 1's
		for pos, val in probabilities.items():
			if val == 0:
				self.safe_moves.add(pos)
			elif val == 1:
				self.known_mines.add(pos)

		if self.safe_moves:
			guess = self.safe_moves.pop()
		else:
			# get all variables of min probability
			sorted_probs = sorted(probabilities.items(), key=operator.itemgetter(1))
			min_prob = sorted_probs[0][1]
			min_pos = list()
			for pos, val in sorted_probs:
				if val == min_prob:
					min_pos.append(pos)
				elif val > min_prob:
					break
				else:
					raise Exception # should never get here

			if len(min_pos == 1):
				guess = min_pos[0]
			else:
				# calculate probability of each space being min or max constraint
				# set guess as highest probability
				pos_gain = list()
				for pos in min_pos:
					pos_gain.append((pos, self._zero_one_state_count(pos, disjoint_sets[:])))

				guess = max(pos_gain, key=lambda x : x[1])[0]

		return guess, probabilities


	def _zero_one_state_count(self, pos, disjoint_sets):
		# edit disjoint_sets to reflect changes
		# perform dfs and return number of viable options
		dream_constraint = set(get_neighbors(pos))
		dream_constraint.add(pos)

		zero_total = 1
		one_total = 1
		for variables, constraints in disjoint_sets:
			if zero_total == 0 and one_total == 0:
				break # both constraints are invalid

			if len(constraints) == 1:
				# edit single constraint
				common = constraints[0][0].intersection(dream_constraint)
				constraints[0][0] -= dream_constraint

				if len(constraints[0][0]) > constraints[0][1]:
					zero_total = 0

				zero_total *= comb(len(constraints[0][0]), constraints[0][1])
				
				if len(common) > constraints[0][1]:
					one_total = 0

				one_total *= comb(len(constraints[0][0]), constraints[0][1] - len(common))
			else:
				variables -= dream_constraint
				zero_constraints, one_constraints = self._adjust_constraints(dream_constraint, variables, constraints)
				if zero_constraints:
					max_zero_constraints = self._get_max_hyper_vars(variables, zero_constraints)
					_, total = self._constraint_dfs(max_zero_constraints, dict(), 0, list())
					zero_total *= total
				else:
					zero_total = 0

				if one_constraints:
					max_one_constraints = self._get_max_hyper_vars(variables, one_constraints)
					_, total = self._constraint_dfs(max_one_constraints, dict(), 0, list())
					one_total *= total
				else:
					one_total = 0

		return zero_total + one_total


	def _adjust_constraints(self, new_constraint, constraints):
		zero_constraints = list()
		one_constraints = list()
		for constraint in constraints:
			common = constraint.intersection(new_constraint)
			constraint[0] -= new_constraint

			if zero_constraints is not None and len(constraint[0]) >= constraint[1]:
				zero_constraints.append(constraint)
			else:
				zero_constraints = None

			if one_constraints is not None and len(common) <= constraint[1]:
				zero_constraints.append((constraint[0], constraint[1] - len(common)))
			else:
				zero_constraints = None

			if zero_constraints is None and one_constraints is None:
				break

		return zero_constraints, one_constraints