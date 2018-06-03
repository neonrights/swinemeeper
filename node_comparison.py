import copy
from solvers import *

trials = 1000
csp_nodes = list()
cccsp_nodes = list()
csp_guesses = list()
cccsp_guesses = list()

board_kwargs = {'shape': (5,5), 'mines': 4, 'start': True, 'render': False}

csp_board = MinesweeperState(**board_kwargs)
cccsp_board = copy.deepcopy(csp_board)

csp_solver = CSPSolver(csp_board)
cccsp_solver = CCCSPSolver(cccsp_board)

print "started nodes generation trials"
with open("data/nodes.data.txt", 'a') as data_file:
	for i in range(trials):
		csp_board = MinesweeperState(**board_kwargs)
		cccsp_board = copy.deepcopy(csp_board)

		csp_solver.new_game(csp_board)
		cccsp_solver.new_game(cccsp_board)

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
		csp_guesses.append(csp_solver.guesses)
		cccsp_guesses.append(cccsp_solver.guesses)
		data_file.write("%d\t%d\t%d\t%d\n" % (csp_solver.guesses, csp_solver.nodes, cccsp_solver.guesses, cccsp_solver.nodes))

print "%f\t%f\t%f\t%f" % (float(sum(csp_guesses)) / trials, float(sum(csp_nodes)) / trials, float(sum(cccsp_guesses)) / trials, floats(sum(cccsp_nodes)) / trials)