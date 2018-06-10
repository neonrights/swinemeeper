import copy
import numpy as np

from collections import deque

from CSPSolver import *
from MinesweeperState import *
from debugger import *


class GlobalSolver(CSPSolver):
	def __init__(self, board, name='global'):
		super(GlobalSolver, self).__init__(board, name)


	def _probabilistic_guess(self):
		win_probability = np.zeros(self.board.shape)

		# get unknown variables
		variables = set(zip(*np.where(self.board.covered)))
		variables -= self.known_mines
		variables -= self.safe_moves
		
		# do initial ordering of variables by number of constraints they appear in
		win_sums = np.zeros_like(win_probability)
		total = 0
		board_values = np.zeros(self.board.shape, dtype=np.int8)
		win_sums, total = self._constraint_dfs(self.constraints, win_sums, total, board_values, variables)
		for var in variables:
			win_probability[var] = float(win_sums[var]) / total

		if self.safe_moves:
			guess = self.safe_moves.pop()
		else:
			guess = np.unravel_index(win_probability.argmax(), win_probability.shape)

		return guess, win_probability


	def _calculate_win_prob(self, dream_board, constraints, variables, known_mines, safe_moves):
		win_probs = np.ones_like(dream_board, dtype=np.float)
		for position in variables:
			if dream_board[position] < 0:
				win_probs[position] = 0
				continue

			# reveal position
			new_constraint_vars = set(get_neighbors(position, self.board.shape))
			new_constraint_val = dream_board[position]
			# TODO, bug with known variables and set of unknown variables

			new_constraint_vars -= safe_moves
			new_constraint_val -= len(new_constraint_vars.intersection(known_mines))
			new_constraint_vars -= known_mines
			new_constraint_vars = new_constraint_vars.intersection(variables)
			if new_constraint_val < 0:
				raise InvalidConstraint

			# add new constraint knowledge
			new_constraints = [[set([position]), 0], [new_constraint_vars, new_constraint_val]]
			dream_constraints, dream_known_mines, dream_safe_moves = self.add_new_constraints(copy.deepcopy(constraints), new_constraints)
			dream_variables = variables - dream_known_mines - dream_safe_moves
			
			if not dream_variables:
				win_probs[position] = 1
			else:
				# recursively call to calculate win prob
				win_sums = np.zeros_like(win_probs)
				total = 0
				board_values = np.zeros(self.board.shape, dtype=np.int8)
				win_sums, total = self._constraint_dfs(dream_constraints, win_sums, total, board_values, dream_variables)

				win_probs[position] = float(np.max([win_sums[pos] for pos in dream_variables])) / total

		return win_probs


	def _constraint_dfs(self, constraint_list, win_sums, total, board_values, variables):
		if not constraint_list: # all constraints resolved
			total += 1

			dream_board = compute_adjacent_mines(board_values)
			#pdb.set_trace()
			win_prob = self._calculate_win_prob(dream_board, self.constraints, variables, set(), set())
			return win_sums + win_prob, total

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

					if not self.valid_constraint(new_constraint):
						return sums, total
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

					if not self.valid_constraint(new_constraint):
						return sums, total
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

					if not self.valid_constraint(new_constraint):
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


@exception_debugger
def test_win_prob():
	# test dfs
	board = MinesweeperState((3,2),2)
	test_solver = GlobalSolver(board)
	assert len(test_solver.constraints) == 1
	# test board state
	test_solver.board.covered = np.ones((3,2), dtype=bool)
	test_solver.board.adjacent_mines = np.array([[1,-1],
												 [2, 2],
												 [1,-1]])

	test_solver.board.reveal((1,0))
	test_solver._add_constraint((1,0))
	assert len(test_solver.constraints) == 1

	_, test_probs = test_solver._probabilistic_guess()
	print(test_probs)
	true_probs = np.array([[0.35, 0.35],
						   [0.   , 0.2],
						   [0.35, 0.35]])
	assert np.allclose(test_probs, true_probs)


	board = MinesweeperState((3,3),2)
	test_solver = GlobalSolver(board)
	assert len(test_solver.constraints) == 1
	# test board state
	test_solver.board.covered = np.ones((3,3), dtype=bool)
	test_solver.board.adjacent_mines = np.array([[1,-1, 2],
												 [1, 2,-1],
												 [0, 1, 1]])

	test_solver.board.reveal((0,0))
	test_solver._add_constraint((0,0))
	assert len(test_solver.constraints) == 2

	_, test_probs = test_solver._probabilistic_guess()
	print(test_probs)
	true_probs = np.array([[0., 0.24, 0.35550265],
						   [0.24, 0.26708995, 0.35608466],
						   [0.35550265, 0.35608466, 0.36888889]])
	assert np.allclose(test_probs, true_probs)


if __name__ == '__main__':
	test_win_prob()