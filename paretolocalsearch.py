from voisinage import *
from ndtree import *
import matplotlib.pyplot as plt
from IPython import display

#Pareto local search algorithm
#Entrée: costs, solutions, v, w, W. costs et solutions sont la population initiale choisie pour l'algorithme, v et w sont les données pour les objets (valeurs et poids) et W le poids maximal du sac
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