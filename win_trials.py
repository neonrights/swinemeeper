import os
import time
import matplotlib.pyplot as plt

from multiprocessing import Pool
from solvers import *

# calculate performance
def run_trial(trial_name, solver_class, trials, *args, **kwargs):
	assert trials > 0
	times, guesses, wins = [], [], 0
	solver = solver_class(MinesweeperState(*args, **kwargs))
	
	with open("data/%s/%s.data.txt" % (solver.name, trial_name), 'a') as data_file:
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


def wrapper(job, solver_class=OverlapSolver):
	if job == 0:
		#run_trial("Beginner", solver_class, 10000, (9, 9), 10, render=False)
		run_trial("Intermediate", solver_class, 1000, (16, 16), 40, start=True, render=False)
		run_trial("Expert", solver_class, 100, (16, 30), 99, start=True, render=False)
	elif job == 1:
		#run_trial("Corner Beginner", solver_class, 10000, (9, 9), 10, start=(0,0), render=False)
		run_trial("Corner Intermediate", solver_class, 1000, (16, 16), 40, start=(0,0), render=False)
		run_trial("Corner Expert", solver_class, 100, (16, 30), 99, start=(0,0), render=False)
	elif job == 2:
		#run_trial("Edge Beginner", solver_class, 10000, (9, 9), 10, start=(0,4), render=False)
		run_trial("Edge Intermediate", solver_class, 1000, (16, 16), 40, start=(0,7), render=False)
		run_trial("Edge Expert", solver_class, 100, (16, 30), 99, start=(0,14), render=False)
	elif job == 3:
		#run_trial("Center Beginner", solver_class, 10000, (9, 9), 10, start=(4,4), render=False)
		run_trial("Center Intermediate", solver_class, 1000, (16, 16), 40, start=(7,7), render=False)
		run_trial("Center Expert", solver_class, 100, (16, 30), 99, start=(7,14), render=False)


if __name__ == '__main__':
	if not os.path.exists('images'):
		os.mkdir('images')

	if not os.path.exists('images/charts'):
		os.mkdir('images/charts')

	if not os.path.exists('data'):
		os.mkdir('data')

	solver_class = OverlapSolver
	name = OverlapSolver(MinesweeperState((5,5),1)).name
	if not os.path.exists("data/%s" % name):
		os.mkdir("data/%s" % name)
	
	print('name\ttrials\tguesses\ttime\twins')
	pool = Pool(4)
	pool.map(wrapper, range(4))
	pool.close()
	pool.join()

	"""
	run_trial("Beginner", solver_class, 10000, (9, 9), 10, render=False)
	run_trial("Intermediate", solver_class, 1000, (16, 16), 40, start=True, render=False)
	run_trial("Expert", solver_class, 100, (16, 30), 99, start=True, render=False)
	run_trial("Corner Beginner", solver_class, 10000, (9, 9), 10, start=(0,0), render=False)
	run_trial("Corner Intermediate", solver_class, 1000, (16, 16), 40, start=(0,0), render=False)
	run_trial("Corner Expert", solver_class, 100, (16, 30), 99, start=(0,0), render=False)
	run_trial("Edge Beginner", solver_class, 10000, (9, 9), 10, start=(0,4), render=False)
	run_trial("Edge Intermediate", solver_class, 1000, (16, 16), 40, start=(0,7), render=False)
	run_trial("Edge Expert", solver_class, 100, (16, 30), 99, start=(0,14), render=False)
	run_trial("Center Beginner", solver_class, 10000, (9, 9), 10, start=(4,4), render=False)
	run_trial("Center Intermediate", solver_class, 1000, (16, 16), 40, start=(7,7), render=False)
	run_trial("Center Expert", solver_class, 100, (16, 30), 99, start=(7,14), render=False)
	#"""