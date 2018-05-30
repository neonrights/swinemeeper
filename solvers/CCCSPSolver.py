import copy
import operator
import numpy as np

from collections import deque
from scipy.misc import comb

from MinesweeperState import *
from CSPSolver import *
from debugger import *


class CCCSPSolver(CSPSolver):
	def __init__(self, board, name='cccsp'):
		super(CCCSPSolver, self).__init__(board, name)

	def _get_max_hyper_vars(self, variables, constraints):
		# split variables into maximal sets
		max_var_sets = set()
		remaining_variables = set(variables)
		while remaining_variables:
			var = remaining_variables.pop()
			max_set = set(variables)

			for constraint in constraints:
				if var in constraint[0]:
					max_set = max_set.intersection(constraint[0])
				else:
					max_set -= constraint[0]

			remaining_variables -= max_set
			max_var_sets.add(frozenset(max_set))

		# rewrite constraints in terms of new maximal set variables
		max_constraints = copy.deepcopy(constraints)
		for constraint in max_constraints:
			for max_var in max_var_sets:
				if constraint[0].issuperset(max_var):
					constraint[0] -= max_var
					constraint[0].add(max_var)

		return max_var_sets, max_constraints

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
			guess = min(probabilities.items(), key=operator.itemgetter(1))[0]

		return guess, probabilities


	def _constraint_dfs(self, constraint_list, sums, total, var_val_pairs):
		if not constraint_list: # all constraints resolved
			# record variable value pairs in probabilities
			combinations = 1
			for var, val in var_val_pairs:
				combinations *= comb(len(var), val)

			for var, val in var_val_pairs:
				try:
					sums[var] += val * combinations
				except KeyError:
					sums[var] = val * combinations

			return sums, total + combinations

		# at each recursion, go through constraint list, select which variable to choose next
		constraint_counts = dict()
		for i, constraint in enumerate(constraint_list):
			if constraint[1] == 0:
				# all must be 0, set all as 0
				new_constraint_list = copy.deepcopy(constraint_list)
				del new_constraint_list[i]
				
				# update constraints, look for conflicts
				delete_set = set()
				for j, new_constraint in enumerate(new_constraint_list):
					new_constraint[0] -= constraint[0]

					if new_constraint[1] < 0: # invalid assignment
						return sums, total
					elif new_constraint[1] > 0 and not new_constraint[0]: # invalid assignment
						return sums, total
					elif sum(len(max_set) for max_set in constraint[0]) < constraint[1]:
						return sums, total
					elif not new_constraint[0]:
						delete_set.add(i)

				# delete empty constraints
				for j in sorted(delete_set, reverse=True):
					del new_constraint_list[j]
				
				# recurse
				self.nodes += 1
				new_var_val_pairs = list(var_val_pairs) + [(var, 0) for var in constraint[0]]
				sums, total = self._constraint_dfs(new_constraint_list, sums, total, new_var_val_pairs)
				return sums, total
			elif len(constraint[0]) == 1:
				# only a single superset variable left, must be equal to remainder
				max_var = next(iter(constraint[0]))
				if len(max_var) < constraint[1] or constraint[1] < 0:
					return sums, total

				new_constraint_list = copy.deepcopy(constraint_list)
				del new_constraint_list[i]

				delete_set = set()
				for j, new_constraint in enumerate(new_constraint_list):
					if max_var in new_constraint[0]:
						new_constraint[0].remove(max_var)
						new_constraint[1] -= constraint[1]

					if new_constraint[1] < 0: # invalid assignment
						return sums, total
					elif new_constraint[1] > 0 and not new_constraint[0]: # invalid assignment
						return sums, total
					elif sum(len(max_set) for max_set in constraint[0]) < constraint[1]:
						return sums, total
					elif not new_constraint[0]:
						delete_set.add(i)

				# delete empty constraints
				for j in sorted(delete_set, reverse=True):
					del new_constraint_list[j]
				
				# recurse
				self.nodes += 1
				new_var_val_pairs = list(var_val_pairs)
				new_var_val_pairs.append((max_var, constraint[1]))
				sums, total = self._constraint_dfs(new_constraint_list, sums, total, new_var_val_pairs)
				return sums, total
			elif sum([len(max_set) for max_set in constraint[0]]) == constraint[1]:
				# all must be 1, set all as 1
				new_constraint_list = copy.deepcopy(constraint_list)
				del new_constraint_list[i]
				
				# update constraints, look for conflicts
				delete_set = set()
				for j, new_constraint in enumerate(new_constraint_list):
					# trickier update
					common = constraint[0].intersection(new_constraint[0])
					new_constraint[0] -= common
					for max_set in common:
						new_constraint[1] -= len(max_set)

					if new_constraint[1] < 0: # invalid assignment
						return sums, total
					elif new_constraint[1] > 0 and not new_constraint[0]: # invalid assignment
						return sums, total
					elif sum(len(max_set) for max_set in constraint[0]) < constraint[1]:
						return sums, total
					elif not new_constraint[0]:
						delete_set.add(i)

				# delete empty constraints
				for j in sorted(delete_set, reverse=True):
					del new_constraint_list[j]
				
				# recurse
				self.nodes += 1
				new_var_val_pairs = list(var_val_pairs) + [(var, len(var)) for var in constraint[0]]
				sums, total = self._constraint_dfs(new_constraint_list, sums, total, new_var_val_pairs)
				return sums, total
			
			for var in constraint[0]:
				try:
					constraint_counts[var] += 1
				except KeyError:
					constraint_counts[var] = 1

		chosen_var = max(constraint_counts.items(), key=operator.itemgetter(1))[0]
		for chosen_val in range(len(chosen_var)+1):
			# copy of constraints, update constraints based off of chosen value
			new_constraint_list = copy.deepcopy(constraint_list)
			delete_set = set()
			try:
				for i, new_constraint in enumerate(new_constraint_list):
					if chosen_var in new_constraint[0]:
						new_constraint[0].remove(chosen_var)
						new_constraint[1] -= chosen_val

					if new_constraint[1] < 0: # invalid assignment
						raise InvalidConstraint
					elif new_constraint[1] > 0 and not new_constraint[0]: # invalid assignment
						raise InvalidConstraint
					elif sum(len(max_set) for max_set in constraint[0]) < constraint[1]:
						raise InvalidConstraint
					elif not new_constraint[0]:
						delete_set.add(i)

			except InvalidConstraint:
				continue

			# delete empty constraints
			for i in sorted(delete_set, reverse=True):
				del new_constraint_list[i]
			
			# recurse with newly assigned value
			self.nodes += 1
			new_var_val_pairs = list(var_val_pairs)
			new_var_val_pairs.append((chosen_var, chosen_val))
			sums, total = self._constraint_dfs(new_constraint_list, sums, total, new_var_val_pairs)

		return sums, total # backtrack, no valid options left

@exception_debugger
def test_cccsp_dfs():
	# test dfs
	board = MinesweeperState((5,5),5)
	test_solver = CCCSPSolver(board)
	# test board state
	test_solver.board.covered = np.ones((5,5), dtype=bool)
	test_solver.board.adjacent_mines = np.array([[ 1,  2,  1, 1, 0],
												 [-1,  3, -1, 2, 0],
												 [ 1,  3, -1, 2, 0],
												 [ 1,  3,  3, 2, 0],
												 [ 1, -1, -1, 1, 0]])
	# test constraint list
	test_solver.board.reveal((2,1))
	test_solver._add_constraint((2,1))

	assert len(test_solver.constraints) == 2

	# call probabilistic guess, check probabilities, check if disjoint
	_, probs = test_solver._probabilistic_guess()
	true_probs = np.array([[1./8, 1./8, 1./8, 1./8, 1./8],
						   [3./8, 3./8, 3./8, 1./8, 1./8],
						   [3./8, 0.,   3./8, 1./8, 1./8],
						   [3./8, 3./8, 3./8, 1./8, 1./8],
						   [1./8, 1./8, 1./8, 1./8, 1./8]])

	test_probs = np.zeros((5,5))
	for key in probs:
		test_probs[key] = probs[key]

	assert np.allclose(test_probs, true_probs)
	print("basic probability test passed!")
	
	# do a harder one
	test_solver.board.reveal((2,3))
	test_solver._add_constraint((2,3))

	assert len(test_solver.constraints) == 3

	_, probs = test_solver._probabilistic_guess()
	prob1 = 57./455
	prob2 = 159./455
	prob3 = 38./91
	prob4 = 68./455
	true_probs = np.array([[prob1, prob1, prob1, prob1, prob1],
						   [prob2, prob2, prob3, prob4, prob4],
						   [prob2, 0.,    prob3, 0.,    prob4],
						   [prob2, prob2, prob3, prob4, prob4],
						   [prob1, prob1, prob1, prob1, prob1]])

	test_probs = np.zeros((5,5))
	for key in probs:
		test_probs[key] = probs[key]

	assert np.allclose(test_probs, true_probs)
	print("hard probability test passed!")


if __name__ == '__main__':
	test_cccsp_dfs()
