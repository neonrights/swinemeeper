import copy
from solvers import *

trials = 100
csp_nodes = list()
cccsp_nodes = list()
board_args = ((9,9), 10, start=True, render=False)

csp_board = MinesweeperState(*board_args)
cccsp_board = copy.deepcopy(board1)

csp_solver = CSPSolver(board1)
cccsp_solver = CCCSPSolver(board2)

print "started nodes generation trials"
with open("data/nodes.data.txt", 'a') as data_file:
	for i in range(trials):
		csp_board = MinesweeperState(*board_args)
		cccsp_board = copy.deepcopy(board1)

		while True:
			try:
				csp_solver.act()
			except GameOver:
				break

		while True:
			try:
				cccsp_solver.act()
			except GameOver:
				break

		csp_nodes.append(csp_solver.nodes)
		cccsp_nodes.append(cccsp_solver.nodes)
		data_file.write("%d\t%d\t%d\t%d\n" % (csp_solver.guesses, csp_solver.nodes, cccsp_solver.guesses, cccsp_solver.nodes))

