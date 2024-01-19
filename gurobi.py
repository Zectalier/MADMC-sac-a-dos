import gurobipy as gp

#Regret-based Elicitation
class RBE():
    
    def __init__(self, nb_criteres, weights):
        self.model = gp.Model("RBE")
        self.n_crit = nb_criteres
        self.weights = []
        
    def init_model(self):
        weight_var = [self.model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=1, name="w"+str(w)) for w in range(self.n_crit)]
        self.model.addConstr(gp.quicksum(weight_var) == 1)
        self.weights.append(weight_var)
        
