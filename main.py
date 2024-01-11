import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ndtree import *
from pareto import *
from func import *
from voisinage import *
from paretolocalsearch import *

w=np.zeros(200,dtype=int)
v=np.zeros((200,6),dtype=int)
filename = "data/2KP200-TA-0.dat"
W=readFile(filename,w,v)

#Génération de m solutions aléatoires
m=10

costs = {}
solutions = {}

for i in range(m):
	w_total = 0
	current_solution = set()
	current_cost =  np.zeros(v.shape[1])
	arr = np.arange(w.shape[0])
	np.random.shuffle(arr)
	for j in arr:
		if w_total + w[j] <= W:
			current_solution.add(j)
			next_item = v[j]
			w_total += w[j]
			for k in range(v.shape[1]):
				current_cost[k] += next_item[k]
	costs[i] = current_cost
	solutions[i] = current_solution

PLS(costs, solutions, v, w, W)