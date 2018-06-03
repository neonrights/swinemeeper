import copy
import operator
import numpy as np

from collections import deque

from MinesweeperState import *
from MinesweeperSolver import *
from debugger import *


class InvalidConstraint(Exception):
	pass


class CSPSolver(MinesweeperSolver):
	def __init__(self, board, name='csp'):
		super(CSPSolver, self).__init__(board, name)


	def new_game(self, new_board):
		super(CSPSolver, self).new_game(new_board)
		self.nodes = 0
		self.guesses = 0
		self.safe_moves = set() # set of safe spaces for fast moves
		self.known_mines = set() # set of known mines

		# initialize constraint list
		all_variables = set(tuple(index) for index in np.ndindex(self.board.shape))
		self.constraints = [[all_variables, self.board.mine_count]]
		# add dictionary for faster constraint pruning
		self.var_dict = dict()
		for var in all_variables:
			self.var_dict[var] = [self.constraints[0]]

		# already made first move
		if self.board.move == 1:
			start = tuple(np.array(np.where(~self.board.covered))[:,0])
			self._add_constraint(start)


	def _add_constraint(self, position):
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
		if constraint_val < 0:
			raise InvalidConstraint

		new_constraints = deque()
		new_constraints.append([set([position]), 0]) # prune newly revealed space
		if constraint_vars:
			new_constraints.append([constraint_vars, constraint_val]) # prune
		
		# continue while there are still newly formed constraints
		while new_constraints:
			constraint_vars, constraint_val = new_constraints.popleft()
			delete_set = set()
			for i, constraint in enumerate(self.constraints):
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
					self.safe_moves = self.safe_moves.union(constraint[0])
					delete_set.add(i)
				elif len(constraint[0]) == constraint[1]:
					new_constraints.append(constraint)
					self.known_mines = self.known_mines.union(constraint[0])
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


	def _get_disjoint_sets(self, variables):
		# split unknown variables into disjoint sets of variables and constraints
		disjoint_sets = list()
		while variables:
			var = variables.pop()
			disjoint_vars, disjoint_constraints = set(), set()
			disjoint_vars.add(var)
			last_vars = set()
			while disjoint_vars != last_vars:
				last_vars = set(disjoint_vars)
				for i, constraint in enumerate(self.constraints):
					if disjoint_vars.intersection(constraint[0]):
						disjoint_vars = disjoint_vars.union(constraint[0])
						disjoint_constraints.add(i)

			assert disjoint_constraints, "variable does not belong to a constraint"
			disjoint_sets.append((disjoint_vars, [self.constraints[i] for i in disjoint_constraints]))
			variables -= disjoint_vars

		return disjoint_sets


	def _probabilistic_guess(self):
		probabilities = dict() # calculate probabilities

		# get unknown variables
		remaining_variables = set([(pos1, pos2) for pos1, pos2 in zip(*np.where(self.board.covered))])
		remaining_variables -= self.known_mines
		remaining_variables -= self.safe_moves
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
				# do initial ordering of variables by number of constraints they appear in
				sums, total = self._constraint_dfs(constraints, dict(), 0, list())
				for var in variables:
					probabilities[var] = float(sums[var]) / total

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
			total += 1
			for var, val in var_val_pairs:
				try:
					sums[var] += val
				except KeyError:
					sums[var] = val
			return sums, total

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
						return sums, total
					elif new_constraint[1] > 0 and not new_constraint[0]: # invalid assignment
						return sums, total
					elif len(constraint[0]) < constraint[1]:
						return sums, total
					elif not new_constraint[0]:
						delete_set.add(j)

				# delete empty constraints
				for j in sorted(delete_set, reverse=True):
					del new_constraint_list[j]
				
				# recurse
				new_var_val_pairs = list(var_val_pairs) + [(var, 0) for var in constraint[0]]
				sums, total = self._constraint_dfs(new_constraint_list, sums, total, new_var_val_pairs)
				return sums, total
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
						return sums, total
					elif new_constraint[1] > 0 and not new_constraint[0]: # invalid assignment
						return sums, total
					elif len(constraint[0]) < constraint[1]:
						return sums, total
					elif not new_constraint[0]:
						delete_set.add(j)

				# delete empty constraints
				for j in sorted(delete_set, reverse=True):
					del new_constraint_list[j]
				
				# recurse
				new_var_val_pairs = list(var_val_pairs) + [(var, 1) for var in constraint[0]]
				sums, total = self._constraint_dfs(new_constraint_list, sums, total, new_var_val_pairs)
				return sums, total
			
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
			new_var_val_pairs = list(var_val_pairs)
			new_var_val_pairs.append((chosen_var, chosen_val))
			sums, total = self._constraint_dfs(new_constraint_list, sums, total, new_var_val_pairs)

		return sums, total # backtrack, no valid options left


	def act(self):
		super(CSPSolver, self).act()

		# view constraint list, determine which space to choose
		if self.safe_moves:
			pos = self.safe_moves.pop()
		else:
			pos, _ = self._probabilistic_guess()
			self.guesses += 1

		val = self.board.reveal(pos)
		if val >= 0:
			self._add_constraint(pos)

		return val


@exception_debugger
def test_csp_constraint_prune():
	board = MinesweeperState((20,16), 100)
	test_solver = CSPSolver(board)
	assert test_solver.constraints[0][1] == 100
	assert len(test_solver.constraints[0][0]) == 20*16
	test_solver.save_state()

	# test constraint reducer
	board = MinesweeperState((20, 20), 40, start=(5,4))
	test_solver = CSPSolver(board)
	assert len(test_solver.constraints[0][0]) == (20**2 - 9), "improper pruning"
	test_solver.save_state()
	test_solver._probabilistic_guess = None
	while test_solver.safe_moves:
		test_solver.act()
		test_solver.save_state()

	# test disjoint sets

@exception_debugger
def test_csp_dfs():
	# test dfs
	board = MinesweeperState((5,5),5)
	test_solver = CSPSolver(board)
	assert len(test_solver.constraints) == 1
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
	test_csp_dfs()
	#test_csp_constraint_prune()
