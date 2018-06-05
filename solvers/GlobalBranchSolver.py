import copy
import numpy as np

from CSPSolver import *
from MinesweeperState import *


class GlobalBnBSolver(CSPSolver):
	def __init__(self, board, name='globalbnb'):
		super(GlobalBnBSolver, self).__init__(board, name)


	def _probabilistic_guess(self):
		mine_probability = np.zeros(self.board.shape)
		constraint_probability = np.zeros(self.board.shape + (9,))

		# get unknown variables
		variables = set([(pos1, pos2) for pos1, pos2 in zip(*np.where(self.board.covered))])
		variables -= self.known_mines
		variables -= self.safe_moves
		
		# do initial ordering of variables by number of constraints they appear in
		mine_sums = np.zeros_like(mine_probability, dtype=np.uint)
		constraint_sums = np.zeros_like(constraint_probability, dtype=np.uint)
		total = 0
		board_values = np.zeros(self.board.shape, dtype=np.int8)
		mine_sums, constraint_sum, total = self._constraint_dfs(self.constraints, win_sums, total, board_values, variables)
		for var in variables:
			mine_probability[var] = float(mine_sums[var]) / total
			for i in range(9):
				constraint_probability[var + (i,)] = float(constraint_sums[var + (i,)]) / total

		# find 0's and 1's
		for pos, val in probabilities.items():
			if val == 0:
				self.safe_moves.add(pos)
			elif val == 1:
				self.known_mines.add(pos)

		if self.safe_moves:
			guess = self.safe_moves.pop()
		else:
			# secondary sort by some heuristic
			var_ordering = sorted(variables, key=lambda var : mine_probability[var], reverse=True)
			
			max_win_prob = 0.
			guess = var_ordering[0]
			for var in var_ordering:
				if max_prob > 1 - mine_probability[var]:
					break # cannot do better, prune

				win_prob = 0.
				for i in range(9):
					if constraint_probability[var + (i,)] > 0:
						win_prob += constraint_probability[var + (i,)] * self._calculate_win_prob(var, i)

				if win_prob > max_win_prob:
					max_win_prob = win_prob
					guess = var

		return guess, probabilities


	def _calculate_win_prob(self, dream_board, variables):
		win_probs = np.ones_like(dream_board)
		for position in variables:
			# reveal position
			# fully realize constraints
			# check if game is won
			# if guess is needed, recursively call to calculate win prob



	def _constraint_dfs(self, constraint_list, win_sums, total, board_values, variables):
		if not constraint_list: # all constraints resolved
			total += 1
			mine_sums += board_values
			adjacent_mines = compute_adjacent_mines(board_values)
			win_sums += self._calculate_win_prob(adjacent_mines, variables)
			return win_sums, total

		# at each recursion, go through constraint list, select which variable to choose next
		self.nodes += 1
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
						return win_sums, total
					elif new_constraint[1] > 0 and not new_constraint[0]: # invalid assignment
						return win_sums, total
					elif len(constraint[0]) < constraint[1]:
						return win_sums, total
					elif not new_constraint[0]:
						delete_set.add(j)

				# delete empty constraints
				for j in sorted(delete_set, reverse=True):
					del new_constraint_list[j]
				
				# recurse
				new_board_values = board_values.copy()
				for var in constraint[0]:
					new_board_values[var] = 0
				win_sums, total = self._constraint_dfs(new_constraint_list, win_sums, total, new_board_values, variables)
				return win_sums, total
			elif len(constraint[0]) == constraint[1]:
				# all must be 1, set all as 1
				new_constraint_list = copy.deepcopy(constraint_list)
				del new_constraint_list[i]
				
				# update constraints, look for conflicts
				delete_set = set()
				for j, new_constraint in enumerate(new_constraint_list):
					new_constraint[1] -= len(constraint[0].intersection(new_constraint[0]))
					new_constraint[0] -= constraint[0]

					if new_constraint[1] < 0: # invalid assignment
						return win_sums, total
					elif new_constraint[1] > 0 and not new_constraint[0]: # invalid assignment
						return win_sums, total
					elif len(constraint[0]) < constraint[1]:
						return win_sums, total
					elif not new_constraint[0]:
						delete_set.add(j)

				# delete empty constraints
				for j in sorted(delete_set, reverse=True):
					del new_constraint_list[j]
				
				# recurse
				new_board_values = board_values.copy()
				for var in constraint[0]:
					new_board_values[var] = 1
				win_sums, total = self._constraint_dfs(new_constraint_list, win_sums, total, new_board_values, variables)
				return win_sums, total
			
			for var in constraint[0]:
				try:
					constraint_counts[var] += 1
				except KeyError:
					constraint_counts[var] = 1

		chosen_var = max(constraint_counts.items(), key=operator.itemgetter(1))[0]
		for chosen_val in [0, 1]:
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
					elif len(constraint[0]) < constraint[1]:
						raise InvalidConstraint
					elif not new_constraint[0]:
						delete_set.add(i)
			except InvalidConstraint:
				continue

			# delete empty constraints
			for i in sorted(delete_set, reverse=True):
				del new_constraint_list[i]
			
			# recurse with newly assigned value
			new_board_values = board_values.copy()
			new_board_values[chosen_var] = chosen_val
			win_sums, total = self._constraint_dfs(new_constraint_list, win_sums, total, new_board_values, variables)

		return win_sums, total # backtrack, no valid options left