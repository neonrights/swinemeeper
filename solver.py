import os
import sys
import random
import operator
import numpy as np

from collections import deque

from state import MinesweeperState
from debugger import exception_debugger


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
					assert constraint[1] == 0, "constraint has no variables but value is not 0"
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


	def _probabilistic_guess(self):
		probabilities = dict() # calculate probabilities

		# get unknown variables
		remaining_variables = set([tuple(var) for var in np.where(self.board.covered)])
		remaining_variables -= self.known_mines
		remaining_variables -= self.safe_moves

		# split unknown variables into disjoint sets of variables and constraints
		disjoint_sets = list()
		while remaining_variables:
			var = remaining_variables.pop()
			disjoint_vars, disjoint_constraints = set(), set()
			disjoint_vars.add(var)
			for constraint in self.constraints:
				if disjoint_vars.intersection(constraint[0]):
					disjoint_vars += constraint[0]

			assert disjoint_constraints, "variable does not belong to a constraint"
			disjoint_sets.append((disjoint_vars, disjoint_constraints))
			remaining_variables -= disjoint_vars

		for variables, constraints in disjoint_sets:
			# check if single constraint
			if len(constraints) == 1:
				assert self.constraints[0][1] > 0, "uncaught resolved constraint"
				# determine if crapshoot, if so, guess immediately
				prob = float(constraints[0][1]) / len(constrains[0][0])
				for var in constraint[0][0]:
					probabilities[var] = prob
			else:
				# do initial ordering of variables by number of constraints they appear in
				sums, total = self._constraint_dfs(constraints, dict(), 0, list())
				for var in variables:
					probabilities[var] = float(sums[var]) / total


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
		constraint_counts = dict()
		for i, constraint in enumerate(constraint_list):
			assert len(constraint[0]) >= constraint[1]
			if constraint[1] == 0:
				# all must be 0, set all as 0
				new_constraint_list = constraint_list[:]
				del new_constraint_list[i]
				
				# update constraints, look for conflicts
				delete_set = set()
				for j, new_constraint in enumerate(new_constraint_list):
					new_constraint[0] -= constraint[0]

					if new_constraint[1] < 0: # invalid assignment
						return sums, total
					elif new_constraint[1] > 0 and not new_constraint[0]: # invalid assignment
						return sums, total
					elif not new_constraint[0]:
						delete_set.add(i)

				# delete empty constraints
				for i in sorted(delete_set, reverse=True):
					del new_constraint_list[i]
				
				# recurse
				var_val_pairs += [(var, 0) for var in constraint[0]]
				sums, total = self._constraint_dfs(new_constraint_list, sums, total, var_val_pairs)
				var_val_pairs = var_val_pairs[:-len(constraint[0])]
				return sums, total
			elif len(constraint[0]) == constraint[1]:
				# all must be 1, set all as 1
				new_constraint_list = constraint_list[:]
				del new_constraint_list[i]
				
				# update constraints, look for conflicts
				delete_set = set()
				for j, new_constraint in enumerate(new_constraint_list):
					new_constraint[1] -= len(constraint[0].disjoint(new_constraint[0]))
					new_constraint[0] -= constraint[0]

					if new_constraint[1] < 0: # invalid assignment
						return sums, total
					elif new_constraint[1] > 0 and not new_constraint[0]: # invalid assignment
						return sums, total
					elif not new_constraint[0]:
						delete_set.add(i)

				# delete empty constraints
				for i in sorted(delete_set, reverse=True):
					del new_constraint_list[i]
				
				# recurse
				var_val_pairs += [(var, 1) for var in constraint[0]]
				sums, total = self._constraint_dfs(new_constraint_list, sums, total, var_val_pairs)
				var_val_pairs = var_val_pairs[:-len(constraint[0])]
				return sums, total
			
			for var in constraint[0]:
				try:
					constraint_counts[var] += 1
				except KeyError:
					constraint_counts[var] = 1

		chosen_var = sorted(constraint_counts.items(), key=operator.itemgetter(1), reverse=True)[0]
		for chosen_val in [0, 1]:
			# copy of constraints, update constraints based off of chosen value
			new_constraint_list = constraint_list[:]
			delete_set = set()
			for i, constraint in enumerate(new_constraint_list):
				if chosen_var in constraint[0]:
					constraint[0].remove(chosen_var)
					constraint[1] -= chosen_val

				if constraint[1] < 0: # invalid assignment
					continue
				elif constraint[1] > 0 and not constraint[0]: # invalid assignment
					continue
				elif not constraint[0]:
					delete_set.add(i)

			# delete empty constraints
			for i in sorted(delete_set, reverse=True):
				del new_constraint_list[i]

			# recurse with newly assigned value
			var_val_pairs.append((chosen_var, chosen_val))
			sums, total = self._constraint_dfs(new_constraint_list, sums, total, var_val_pairs)
			var_val_pairs.pop()

		return sums, total # backtrack, no valid options left


	def act(self):
		# view constraint list, determine which space to choose
		self.move += 1
		if self.safe_moves:
			pos = self.safe_moves.pop()
		else:
			raise NotImplementedError
			pos = self._probabilistic_guess()

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


if __name__ == '__main__':
	test_cps()
