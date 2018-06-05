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

			new_constraint_vars -= safe_moves
			new_constraint_val -= len(new_constraint_vars.intersection(known_mines))
			new_constraint_vars -= known_mines
			new_constraint_vars = new_constraint_vars.intersection(variables)
			if new_constraint_val < 0:
				raise InvalidConstraint

			dream_known_mines = set(known_mines)
			dream_safe_moves = set(safe_moves)

			new_constraints = deque()
			new_constraints.append([set([position]), 0]) # prune newly revealed space
			if new_constraint_vars:
				new_constraints.append([new_constraint_vars, new_constraint_val]) # prune
			
			# fully realize constraints
			# continue while there are still newly formed constraints
			dream_constraints = copy.deepcopy(constraints)
			while new_constraints:
				constraint_vars, constraint_val = new_constraints.popleft()
				delete_set = set()
				for i, constraint in enumerate(dream_constraints):
					if constraint_val == 0:
						# resolved constraint, all safe
						constraint[0] -= constraint_vars
					elif len(constraint_vars) == constraint_val:
						# resolved constraint, all mines
						constraint[1] -= len(constraint[0].intersection(constraint_vars))
						constraint[0] -= constraint_vars
					else:
						# unresolved constraint
						if constraint[0].issuperset(constraint_vars):
							# new constraint is subset of old constraint
							constraint[0] -= constraint_vars
							constraint[1] -= constraint_val
						elif constraint[0].issubset(constraint_vars):
							# old constraint is subset of new constraint
							new_vars = constraint_vars.difference(constraint[0])
							new_val = constraint_val - constraint[1]
							new_constraints.append([new_vars, new_val])
							continue # skip remaining? must not add unaltered constraint

					if not constraint[0]:
						if constraint[1] != 0:
							raise InvalidConstraint
						delete_set.add(i) # empty constraint, remove

					# if constraint is resolved, add new variables to list
					if constraint[1] == 0:
						new_constraints.append(constraint)
						dream_safe_moves = dream_safe_moves.union(constraint[0])
						delete_set.add(i)
					elif len(constraint[0]) == constraint[1]:
						new_constraints.append(constraint)
						dream_known_mines = dream_known_mines.union(constraint[0])
						delete_set.add(i)

				for i in sorted(delete_set, reverse=True):
					del dream_constraints[i]

				if constraint_val == 0:
					for move in constraint_vars:
						if move in variables:
							dream_safe_moves.add(move)
				elif len(constraint_vars) == constraint_val:
					dream_known_mines = dream_known_mines.union(constraint_vars)
				elif constraint_vars:
					dream_constraints.append([constraint_vars, constraint_val])
			
			dream_variables = variables - dream_known_mines - dream_safe_moves
			if not dream_variables:
				win_probs[position] = 1
			else:
				# recursively call to calculate win prob
				var_win_prob = self._calculate_win_prob(dream_board, dream_constraints, dream_variables, dream_known_mines, dream_safe_moves)
				win_prob = np.mean([var_win_prob[pos] for pos in dream_variables])
				#pdb.set_trace()
				win_probs[position] = win_prob

		return win_probs


	def _constraint_dfs(self, constraint_list, win_sums, total, board_values, variables):
		if not constraint_list: # all constraints resolved
			total += 1
			adjacent_mines = compute_adjacent_mines(board_values)
			win_prob = self._calculate_win_prob(adjacent_mines, self.constraints, variables, set(), set())
			#pdb.set_trace()
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