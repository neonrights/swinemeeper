import os
import time
import matplotlib.pyplot as plt

from multiprocessing import Pool
from solvers import *

# calculate performance
def run_trial(trial_name, solver, trials, *args, **kwargs):
	assert trials > 0
	with open("data/%s.data.txt" % trial_name, 'a') as data_file:
		times, guesses, wins = [], [], 0
		solver = solver(MinesweeperState(*args, **kwargs))
		for i in range(trials):
			board = MinesweeperState(*args, **kwargs)
			solver.new_game(board)
			if kwargs['render']:
				solver.save_state()

			start = time.time()
			while True:
				try:
					solver.act()
				except GameOver:
					break
				finally:
					if kwargs['render']:
						solver.save_state()
			
			end = time.time()

			if board.is_goal():
				times.append(end - start)
				guesses.append(solver.guesses)
				wins += 1

			data_file.write("%f\t%d\t%d\t%s\n" % (end - start, solver.guesses, solver.nodes, board.is_goal()))
	
	print("%s\t%d\t%f\t%f\t%f" % (trial_name, trials, float(sum(guesses)) / wins, float(sum(times)) / wins, float(wins) / trials))

	fig = plt.figure()
	plt.hist(guesses)
	plt.title("Histogram of Guesses per Game for %s" % trial_name)
	plt.xlabel("counts of guesses")
	plt.ylabel("counts of counts of guesses")
	plt.savefig("images/charts/%s Guesses.png" % trial_name)

	fig = plt.figure()
	plt.hist(times)
	plt.title("Histogram of Times per Game for %s" % trial_name)
	plt.xlabel("counts of times in bin")
	plt.ylabel("time")
	plt.savefig("images/charts/%s Times.png" % trial_name)


def wrapper(job):
	if job == 0:
		#run_trial("Beginner CCCSP", CCCSPSolver, 10000, (9, 9), 10, render=False)
		run_trial("Intermediate CCCSP", CCCSPSolver, 1000, (16, 16), 40, start=True, render=False)
		run_trial("Expert CCCSP", CCCSPSolver, 100, (16, 30), 99, start=True, render=False)
	elif job == 1:
		#run_trial("Corner Beginner CCCSP", CCCSPSolver, 10000, (9, 9), 10, start=(0,0), render=False)
		run_trial("Corner Intermediate CCCSP", CCCSPSolver, 1000, (16, 16), 40, start=(0,0), render=False)
		run_trial("Corner Expert CCCSP", CCCSPSolver, 100, (16, 30), 99, start=(0,0), render=False)
	elif job == 2:
		#run_trial("Edge Beginner CCCSP", CCCSPSolver, 10000, (9, 9), 10, start=(0,4), render=False)
		run_trial("Edge Intermediate CCCSP", CCCSPSolver, 1000, (16, 16), 40, start=(0,7), render=False)
		run_trial("Edge Expert CCCSP", CCCSPSolver, 100, (16, 30), 99, start=(0,14), render=False)
	elif job == 3:
		#run_trial("Center Beginner CCCSP", CCCSPSolver, 10000, (9, 9), 10, start=(4,4), render=False)
		run_trial("Center Intermediate CCCSP", CCCSPSolver, 1000, (16, 16), 40, start=(7,7), render=False)
		run_trial("Center Expert CCCSP", CCCSPSolver, 100, (16, 30), 99, start=(7,14), render=False)


if __name__ == '__main__':
	if not os.path.exists('images'):
		os.mkdir('images')

	if not os.path.exists('images/charts'):
		os.mkdir('images/charts')

	if not os.path.exists('data'):
		os.mkdir('data')

	print('name\ttrials\tguesses\ttime\twins')
	pool = Pool(4)
	pool.map(wrapper, range(4))
	pool.close()
	pool.join()

	"""
	run_trial("Beginner CCCSP", CCCSPSolver, 10000, (9, 9), 10, render=False)
	run_trial("Intermediate CCCSP", CCCSPSolver, 1000, (16, 16), 40, render=False)
	run_trial("Expert CCCSP", CCCSPSolver, 100, (16, 30), 99, render=False)

	run_trial("Corner Beginner CCCSP", CCCSPSolver, 10000, (9, 9), 10, start=(0,0), render=False)
	run_trial("Corner Intermediate CCCSP", CCCSPSolver, 1000, (16, 16), 40, start=(0,0), render=False)
	run_trial("Corner Expert CCCSP", CCCSPSolver, 100, (16, 30), 99, start=(0,0), render=False)

	run_trial("Edge Beginner CCCSP", CCCSPSolver, 10000, (9, 9), 10, start=(0,4), render=False)
	run_trial("Edge Intermediate CCCSP", CCCSPSolver, 1000, (16, 16), 40, start=(0,7), render=False)
	run_trial("Edge Expert CCCSP", CCCSPSolver, 100, (16, 30), 99, start=(0,14), render=False)

	run_trial("Center Beginner CCCSP", CCCSPSolver, 10000, (9, 9), 10, start=(4,4), render=False)
	run_trial("Center Intermediate CCCSP", CCCSPSolver, 1000, (16, 16), 40, start=(7,7), render=False)
	run_trial("Center Expert CCCSP", CCCSPSolver, 100, (16, 30), 99, start=(7,14), render=False)
	"""