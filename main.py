import os
import time
import imageio

from solver import *
from state import *

wins = 0
for i in range(1000):
	start = time.time()
	board = MinesweeperState((30, 16), 99, start=(0,0), render=False)
	solver = CCCSPSolver(board)
	solver.save_state()
	while solver.act() >= 0 and not solver.board.is_goal():
		solver.save_state()

	end = time.time()
	solver.save_state()
	print end - start
	if solver.board.is_goal():
		wins += 1

print(float(wins) / 1000)

"""
images = list()
i = 1
while os.path.isfile('images/board_%d.png' % i):
	images.append(imageio.imread('images/board_%d.png' % i))
	i += 1

imageio.mimsave('images/board.gif', images, fps=20)
"""