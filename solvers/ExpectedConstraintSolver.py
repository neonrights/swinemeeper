import operator
import numpy as np

from MinesweeperState import *
from CCCSPSolver import *
from debugger import *


class ExpectedConstraintSolver(CCCSPSolver):
	def __init__(self, board, name='ExpectedConstraint'):
		super(ECSolver, self).__init__(board, name)

	def _probabilistic_guess(self):
		probabilities = dict() # calculate probabilities
		expected_probs = dict()

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
				e_prob = float(constraint[0][1]) / (len(constraint[0][0]) - 1)
				for var in constraints[0][0]:
					probabilities[var] = prob
					for neighbor in get_neighbors(var, self.board.shape):
						try:
							if neighbor in constraint[0][0]:
								expected_probs[neighbor] += e_prob
							else:
								expected_probs[neighbor] += prob
						except KeyError:
							if neighbor in constraint[0][0]:
								expected_probs[neighbor] = e_prob
							else:
								expected_probs[neighbor] = prob
			else:
				_, max_constraints = self._get_max_hyper_vars(variables, constraints)

				# use dfs to calculate probabilities
				sums, total, e_sums, e_totals = self._constraint_dfs(max_constraints, dict(), 0, dict(), dict(), list())
				for max_var, val in sums.items():
					set_size = len(max_var)
					prob = float(val) / (set_size * total)
					for var in max_var:
						probabilities[var] = prob
						for neighbor in get_neighbors(var, self.board.shape):
							# TODO calculate expected probs, uses neighbor and var
							try:
								expected_probs[neighbor] += 1
							except KeyError:
								expected_probs[neighbor] = 1

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
			raise NotImplementedError

		return guess, probabilities

	def _constraint_dfs(self, constraint_list, sums, total, e_sums, e_totals, var_val_pairs):
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

				if len(var) < val: # valid expected value, calculate expected
					e_combs = 1
					for e_var, e_val in var_val_pairs:
						if var is e_var:
							e_combs *= comb(len(e_var) - 1, e_val)
						else:
							e_combs *= comb(len(e_var), e_val)

					# unique total
					try:
						e_totals[var] += e_combs
					except KeyError:
						e_totals[var] = e_combs

					# unique combinations
					e_sums[var] = {var : (var - 1) * e_combs}
					for e_var, e_val in var_val_pairs:
						try:
							if var is not e_var:
								e_sums[var][e_var] += e_val * e_combs
						except KeyError:
							e_sums[var][e_var] = e_val * e_combs

			return sums, total + combinations, e_sums, e_totals

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
						return sums, total, e_sums, e_totals
					elif new_constraint[1] > 0 and not new_constraint[0]: # invalid assignment
						return sums, total, e_sums, e_totals
					elif sum(len(max_set) for max_set in constraint[0]) < constraint[1]:
						return sums, total, e_sums, e_totals
					elif not new_constraint[0]:
						delete_set.add(i)

				# delete empty constraints
				for j in sorted(delete_set, reverse=True):
					del new_constraint_list[j]
				
				# recurse
				self.nodes += 1
				new_var_val_pairs = list(var_val_pairs) + [(var, 0) for var in constraint[0]]
				sums, total = self._constraint_dfs(new_constraint_list, sums, total, new_var_val_pairs)
				return sums, total, e_sums, e_totals
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
						return sums, total, e_sums, e_totals
					elif new_constraint[1] > 0 and not new_constraint[0]: # invalid assignment
						return sums, total, e_sums, e_totals
					elif sum(len(max_set) for max_set in constraint[0]) < constraint[1]:
						return sums, total, e_sums, e_totals
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
				return sums, total, e_sums, e_totals
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
				return sums, total, e_sums, e_totals
			
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

		return sums, total, e_sums, e_totals # backtrack, no valid options left	


def test_reordering():
	pass


if __name__ == '__main__':
	test_reordering()
