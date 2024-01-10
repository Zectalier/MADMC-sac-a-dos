import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ndtree import *
from pareto import *
from func import *
from voisinage import *

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


visited = set()

tree2 = NDTree(None, pareto_dom_func = pareto_dominance_maximisation)

for k, val in costs.items():
	tree2.update_tree(val, solutions[k])

costs, solutions = tree2.get_all_costs_values()

iter_count = 0

all_voisins = {-1: 1} # Pour entrer dans la boucle
while len(all_voisins) > 0:
	print("Nombre de coûts à l'itération " + str(iter_count) + ": " + str(len(costs)))
	all_voisins = dict()
	all_solutions = dict()
	for k, val in costs.items():
		c, s, visited = voisinage_var(val, solutions[k], v, w, W, visited)
		all_voisins[k] = c
		all_solutions[k] = s
	
	voisin_count = 0
	for k, val in all_voisins.items():
		voisin_count += len(val)

	print("Nombre de voisins à l'itération " + str(iter_count) + ": " + str(voisin_count))
	for k, val in all_voisins.items():
		for i, c in val.items():
			tree2.update_tree(c, all_solutions[k][i])

	costs, solutions = tree2.get_all_costs_values()
	iter_count += 1

costs, solutions = tree2.get_all_costs_values()
print(costs)
print(solutions)