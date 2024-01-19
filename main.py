import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gurobipy as gp

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
m=1000

#Reshape v and w to only take first three objectives and first 20 items
v = v[:20]
v = v[:,:3]
w = w[:20]

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

# Create a new model
model = gp.Model("knapsack")

# Define the decision variables
x = {}
for i in range(num_items):
    x[i] = model.addVar(vtype=gp.GRB.BINARY, name=f"x_{i}")

# Set the objective function
model.setObjective(gp.quicksum(profit[i] * x[i] for i in range(num_items)), gp.GRB.MAXIMIZE)

# Add the capacity constraint
model.addConstr(gp.quicksum(weight[i] * x[i] for i in range(num_items)) <= capacity, "capacity")

# Optimize the model
model.optimize()

# Check if the optimization was successful
if model.status == gp.GRB.OPTIMAL:
    # Get the optimal solution
    solution = model.getAttr("x", x)
    for i in range(num_items):
        if solution[i] > 0.5:
            print(f"Item {i} is selected")
else:
    print("No feasible solution found.")