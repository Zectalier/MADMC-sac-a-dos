from gurobi import *
from ndtree import *
from voisinage import *

#Incremental Elicitation Local Search
def IELS(costs, solutions, v, w, W, agr_func, eps, do_display = False):
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

    dm = DM(v.shape[1], agr_func)

    to_visit = get_best_sol(X, dm, v.shape[1], agr_func, eps)
    to_visit_sol = solutions[id[np.where(X == to_visit)[0]][0]]

    while not empty:
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
        to_visit = get_best_sol(np.array(list(costs.values())), dm, v.shape[1], agr_func, eps)
        if (old_best==to_visit).all():
            empty = True
            break
        to_visit_sol = solutions[id[np.where(X == to_visit)[0]][0]]
    
    return tree.get_all_costs_values(), (to_visit, to_visit_sol)
    
def get_best_sol(X, dm, n_crit, agr_func, eps):
    rbe = RBE(n_crit, agr_func, X)
    res = rbe.MMR(X)
    MMR_value = res[0]
    while MMR_value > eps:
        res = rbe.CSS_ask_query(X, dm)
        MMR_value = res[0]
    return res[1]
