from voisinage import *
from ndtree import *
from ndtree_tuple import NDTree as NDTree_tuple
import matplotlib.pyplot as plt
from IPython import display

#Pareto local search algorithm
#Input: costs, solutions, v, w, W, costs and solutions are the initial population chosen for the algorithm, v and w are the data, W is the maximum weight
def PLS(costs, solutions, v, w, W):
    visited = set()
    tree = NDTree()
    for k, val in costs.items():
        tree.update_tree(val, solutions[k]) #Initialisation of the tree with pareto dominating solutions
    costs, solutions = tree.get_all_costs_values()
    for k, val in costs.items():
        visited.add(frozenset(val))
    empty = False
    while not empty:
        print(len(costs))
        id, cost = costs.popitem()
        sol = solutions.pop(id)
        to_test, to_test_sol, visited = voisinage_var(cost, sol, v, w, W, visited)
        if len(to_test) == 0:
            empty = True
        else:
            for k, val in to_test.items():
                tree.update_tree(val, to_test_sol[k])
        costs, solutions = tree.get_all_costs_values()
        #Display the pareto front for the first two objectives dynamically
        plt.scatter(np.vstack(np.array(list(costs.values()))[:,0]), np.vstack(np.array(list(costs.values()))[:,1]))
        display.clear_output(wait=True)
        display.display(plt.gcf())

        if len(costs) == 0:
            empty = True
        
        
    costs, solutions = tree.get_all_costs_values()
    return costs, solutions

def PLS_tuple(costs, v, w, W):
    visited = set()
    tree = NDTree_tuple()
    for k in costs:
        tree.update_tree(k) #Initialisation of the tree with pareto dominating solutions
    costs = tree.get_all_costs()
    for k in costs:
        visited.add(frozenset(k[1]))
    empty = False
    while not empty:
        print(len(costs))
        cost = costs.pop()
        to_test, visited = voisinage_tuple_var(cost[0], cost[1], v, w, W, visited)
        if len(to_test) == 0:
            empty = True
        else:
            for k in to_test:
                tree.update_tree(k)
        costs = tree.get_all_costs()
        #Display the pareto front for the first two objectives dynamically
        plt.clf()
        plt.scatter(np.vstack(np.array(list(costs.values()))[:,0]), np.vstack(np.array(list(costs.values()))[:,1]))
        display.clear_output(wait=True)
        display.display(plt.gcf())

        if len(costs) == 0:
            empty = True
        
        
    costs = tree.get_all_costs()
    return costs