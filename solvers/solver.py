import os
import sys
import copy
import random
import operator
import numpy as np

from collections import deque
from scipy.misc import comb

from state import *
from debugger import *


class InvalidConstraint(Exception):
	pass

class MinesweeperSolver(object):
	def __init__(self, board, name='solver'):
		self.name = name
		self.new_game(board)
		# other shared variables, like remaining mines

	def new_game(self, new_board):
		self.board = new_board


	def act(self):
		if self.board.is_goal() or self.board._loss:
			raise GameOver


	def save_state(self):
		if not os.path.exists("images"):
			os.mkdir("images")

		if not os.path.exists("images/%s" % self.name):
			os.mkdir("images/%s" % self.name)

		self.board.to_image("images/%s/board_%d.png" % (self.name, self.board.move))


class ExpectedReturnSolver(CCCSPSolver):
	# same as before, except it reorders guesses based on off of potential information gain
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
			# take all guesses with min probability
			# calculate expected information return
			# select guess with best return
			guess = min(probabilities.items(), key=operator.itemgetter(1))[0]

		return guess, probabilities

@exception_debugger
def test_csp_constraint_prune():
	test_solver = CSPSolver((20, 16), 100)
	assert test_solver.constraints[0][1] == 100
	assert len(test_solver.constraints[0][0]) == 20*16
	test_solver.save_state()

	# test constraint reducer
	test_solver = CSPSolver((20, 20), 40, start=(5,4))
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
	test_csp_dfs()
	test_cccsp_dfs()
