import sys
from gurobi import *
from ndtree import *
from voisinage import *

#Regret Based Local Search
def RBLS(costs, solutions, v, w, W, dm, agr_func, eps, logging = False):
    visited = set()
    tree = NDTree()
    for k, val in costs.items():
        tree.update_tree(val, solutions[k]) #Initialisation of the tree with pareto dominating solutions
    costs, solutions = tree.get_all_costs_values()
    for k, val in costs.items():
        visited.add(frozenset(val))
    empty = False

    X = np.array(list(costs.values()))
    id = np.array(list(costs.keys()))

    queries = 0 #Total number of queries
    to_visit, n_query = get_best_sol(X, dm, v.shape[1], agr_func, eps, logging)
    queries += n_query
    to_visit_sol = solutions[id[np.where(X == to_visit)[0]][0]]
    i = 0
    while not empty:
        i += 1
        if logging:
            print("\nIteration " + str(i))
            print("Current number of solutions: " + str(len(costs)))
        to_test, to_test_sol, visited = voisinage_var(to_visit, to_visit_sol, v, w, W, visited)
        if len(to_test) == 0:
            empty = True
        else:
            for k, val in to_test.items():
                tree.update_tree(val, to_test_sol[k])
        costs, solutions = tree.get_all_costs_values()

        X = np.array(list(costs.values()))
        id = np.array(list(costs.keys()))
        old_best = to_visit
        to_visit, n_query = get_best_sol(np.array(list(costs.values())), dm, v.shape[1], agr_func, eps, logging)
        if logging:
            print("\rNumber of queries: " + str(n_query))
        queries += n_query
        if (old_best==to_visit).all():
            empty = True
            break
        to_visit_sol = solutions[id[np.where(X == to_visit)[0]][0]]
    
    return tree.get_all_costs_values(), (to_visit, to_visit_sol), queries
    
def get_best_sol(X, dm, n_crit, agr_func, eps, logging = False):
    rbe = RBE(n_crit, agr_func, X)
    res = rbe.MMR(X)
    MMR_value = res[0]
    i = 0
    while MMR_value > eps:
        i += 1
        if logging:
            sys.stdout.write("\rQuery " + str(i))
            sys.stdout.flush()
        res = rbe.CSS_ask_query(X, dm)
        MMR_value = res[0]
    return res[1], i
