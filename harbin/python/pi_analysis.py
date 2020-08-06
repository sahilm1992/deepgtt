# import pickle 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pickle

dbfile = open('pi2x', 'rb')
pi_to_x = pickle.load(dbfile) 
# pi_to_x = [set() for _ in range(10)]
# for i in range(10):
	# pi_to_x[i].add((i,i))
# pi_to_x is a list of sets, each set containing the x's that were mapped to one cluster
colors = cm.rainbow(np.linspace(0, 1, 1000))
for i, set_i in enumerate(pi_to_x):
	if (i%100 != 0):
		continue
	c=colors[i]
	for dests in set_i:
		plt.scatter(dests[1], dests[0], c=np.array([c]))
plt.savefig('pi_to_x.png')

file = open('next_segment_predictions', 'rb')
lis = pickle.load(file)



