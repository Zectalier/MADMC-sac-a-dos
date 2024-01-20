import gurobipy as gp
import numpy as np

#Regret-based Elicitation
class RBE():
    
    def __init__(self, nb_criteres, agr_func, X, log = False):
        self.model = gp.Model("RBE")
        self.logging = log
        self.model.Params.LogToConsole = 0 #Disable the console output
        self.n_crit = nb_criteres
        self.n_queries = 0
        self.agr_func = agr_func
        self.pairs = []
        self.init_model()
        self.init_MR = np.max([self.MR(x, X) for x in X]) #Set the initial MR, wich correspond to the initial MR without any queries

        if self.logging == True:
            self.model.write('model' + str(self.n_queries) + '.LP') #Write the model for the current state of the elicitation process

    def init_model(self):
        self.weight_var = [self.model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=1, name="w"+str(w)) for w in range(self.n_crit)]
        self.model.addConstr(gp.quicksum(self.weight_var) == 1)

    #Insert new query corresponding to a new pair (x, y) such as x is known to be preffered to y
    def insert_pair(self, pair):
        self.pairs.append(pair)
        self.n_queries += 1
        x = pair[0]
        y = pair[1]
        if self.agr_func == "WS":
            self.model.addConstr(gp.quicksum(self.weight_var[i]*(x[i]-y[i]) for i in range(self.n_crit)) >= 0)
    
    #The Minimax Regret (MMR) over X 
    def MMR(self, X):
        MR_res = np.min([self.MR_normalized(x, X) for x in X])
        return MR_res

    #The Max Regret (MR) of alternative x ∈ X
    def MR(self, x, X):
        return np.max([self.PMR(x, y) for y in X])
    
    #The Max Regret normalized, used for the incremental elicitation
    def MR_normalized(self, x, X):
        return np.max([self.PMR(x, y) for y in X])/self.init_MR
    
    #The Pairwise Max Regret (PMR) of alternatives x, y ∈ X, such as x is known to be preffered to y
    def PMR(self, x, y):
        if self.agr_func == "WS":
            self.model.setObjective(gp.quicksum(self.weight_var[i]*(y[i]-x[i]) for i in range(self.n_crit)), gp.GRB.MAXIMIZE)
        self.model.update()
        self.model.optimize()
        self.model.printStats()
        return self.model.objVal

    #Current Solution strategy (CSS), return xp and yp for the next query
    def CSS_get_compare(self, X):
        xp = np.argmin([self.MR(x, X) for x in X])
        yp = np.argmax([self.PMR(X[xp], y) for y in X])
        return X[xp], X[yp]
    
    #Ask the DM to compare two new alternatives
    def CSS_ask_query(self, X, DM):
        xp, yp = self.CSS_get_compare(X)
        if DM.compare(xp, yp):
            self.insert_pair([xp, yp])
        else:
            self.insert_pair([yp, xp])
        if self.logging == True:
            self.model.write('model' + str(self.n_queries) + '.LP') #Write the model for the current state of the elicitation process, will write the last pair from PMR compared
        return self.MMR(X)
        
class DM():
    
    def __init__(self, nb_criteres, agr_func):
        self.n_crit = nb_criteres
        self.agr_func = agr_func
        self.weights = self.get_random_weights()

    #Return a random vector of weights of size n_crit, summing to 1
    def get_random_weights(self):
        if self.agr_func == "WS":
            return np.random.dirichlet(np.ones(self.n_crit),size=1)[0]
    
    def compare(self, x, y):
        if self.agr_func == "WS":
            return np.sum(self.weights*x) >= np.sum(self.weights*y)