import numpy as np

#Donne les voisins de la solution sol, dont la valeur de la solution est cost, pour la liste d'objet obj et leur poids weight
def voisinage(cost, sol, obj, weight, max_weight,):
    inside = np.array(list(sol))
    outside = np.delete(np.array(list(range(len(obj)))), inside)

    w_sol = weight[np.array(list(sol))].sum()
    voisinage = dict()
    voisinage_cost = dict()
    k = 0
    for i in inside:
        for j in outside:
            if w_sol - weight[i] + weight[j] <= max_weight:
                new_sol = sol.copy()
                new_sol.remove(i)
                new_sol.add(j)
                voisinage[k] = set(new_sol)
                voisinage_cost[k] = cost - obj[i] + obj[j]
                k+=1
    return voisinage_cost, voisinage

#Variante du voisinage, qui ne retourne pas les voisins qui ont été déjà visités
def voisinage_var(cost, sol, obj, weight, max_weight, visited):
    inside = np.array(list(sol))
    outside = np.delete(np.array(list(range(len(obj)))), inside)

    w_sol = weight[np.array(list(sol))].sum()
    voisinage = dict()
    voisinage_cost = dict()
    k = 0
    for i in inside:
        for j in outside:
            if w_sol - weight[i] + weight[j] <= max_weight:
                new_sol = sol.copy()
                new_sol.remove(i)
                new_sol.add(j)
                if set(new_sol) not in visited:
                    voisinage[k] = set(new_sol)
                    voisinage_cost[k] = cost - obj[i] + obj[j]
                    visited.add(frozenset(new_sol))
                    k+=1
    return voisinage_cost, voisinage, visited
    
#Variante du voisinage, qui ne retourne pas les voisins qui ont été déjà visités
def voisinage_tuple_var(cost, sol, obj, weight, max_weight, visited):
    inside = np.array(list(sol))
    outside = np.delete(np.array(list(range(len(obj)))), inside)

    w_sol = weight[np.array(list(sol))].sum()
    voisinage = []
    k = 0
    for i in inside:
        for j in outside:
            if w_sol - weight[i] + weight[j] <= max_weight:
                new_sol = sol.copy()
                new_sol.remove(i)
                new_sol.add(j)
                if set(new_sol) not in visited:
                    voisin_cost = cost - obj[i] + obj[j]
                    voisinage.append((list(voisin_cost), set(new_sol)))
                    visited.add(frozenset(new_sol))
                    k+=1
    return voisinage, visited
    
