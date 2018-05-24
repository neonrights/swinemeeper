import os
import imageio

from solver import *


solver = CSPSolver((30, 16), 99, start=(0,0))
solver.save_state()
while solver.act() >= 0 and not solver.board.is_goal():
	solver.save_state()

solver.save_state()

images = list()
i = 1
while os.path.isfile('images/board_%d.png' % i):
	images.append(imageio.imread('images/board_%d.png' % i))
	i += 1

imageio.mimsave('images/board.gif', images)
