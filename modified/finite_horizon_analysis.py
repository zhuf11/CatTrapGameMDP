from CatGame import Game
import time
import matplotlib.pyplot as plt

g = Game(5)
num_states = [i for i in range(1, 26)]
max_depth = 2  # can be set to 1, 2, 3 too
times_list = []
for num_state in num_states:
	print("n", num_state)
	start_time = time.time()
	result = g.mdp(max_depth, num_state)
	elapsed_time = (time.time() - start_time) * 1000
	times_list.append(elapsed_time)
plt.plot(num_states, times_list)
title_str = "Time or max depth = " + str(max_depth)
plt.title(title_str)
plt.xlabel("num_states")
plt.ylabel("Time taken (ms)")
plt.show()
