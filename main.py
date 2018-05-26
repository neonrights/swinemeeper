import os
import time
import imageio

from solver import *
from state import *

# calculate performance
trials = 1000
render = False

total_time = 0.
total_wins = 1e-12
total_guesses = 0
solver = CCCSPSolver(MinesweeperState((16, 16), 40, start=(0,0), render=render))
for i in range(trials):
	board = MinesweeperState((16, 16), 40, start=(0,0), render=render)
	solver.new_game(board)
	if render:
		solver.save_state()

	start = time.time()
	while True:
		try:
			solver.act()
		except GameOver:
			break
		finally:
			if render:
				solver.save_state()

	end = time.time()

	print end - start, solver.guesses, board.is_goal()
	
	if board.is_goal():
		total_time += end - start
		total_wins += 1
		total_guesses += solver.guesses

print float(total_wins) / trials, total_time / total_wins, float(total_guesses / total_wins)
