import os
import sys

from MinesweeperState import *


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
