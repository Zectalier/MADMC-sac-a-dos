import numpy as np

def pareto_test(costs):
    # Test si chaque cout est pareto optimal ou non dans le cas d'une maximisation
    # True si opt, False sinon
    is_opt = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        is_opt[i] = np.all(np.any(costs[:i]<c, axis=1)) and np.all(np.any(costs[i+1:]<c, axis=1))
    return is_opt


def pareto_front(dict_vect):
    # prend en entrée un dictionnaire des vecteurs de coûts
    # supprime les vecteurs Pareto dominés
    # retourne le nouveau dictionnaire, ne contenant que des vecteurs non dominés pour la relation de Pareto

    items = [v for v in dict_vect.values()]
    keys = [k for k in dict_vect.keys()]
    to_test = np.array(items)

    # Test si chaque cout est dominé ou non par un autre point
    is_pareto = pareto_test(to_test)

    new_dict = {}
    for i in range(is_pareto.shape[0]):
        if is_pareto[i]:
            new_dict[keys[i]] = items[i]
        
    return new_dict

dico = {2:np.array([1,2,3,4,2,3]),0:np.array([1,2,3,4,2,8]),1:np.array([4,5,1,4,2,3]),3:np.array([1,4,3,4,2,3])}
print(pareto_front(dico))