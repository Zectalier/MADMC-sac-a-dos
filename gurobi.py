from itertools import combinations
import random
import gurobipy as gp
import numpy as np

def get_all_combinations(p):
    all_subsets=[]
    for k in range(p+1):
        subset=list(combinations(range(p),k))
        all_subsets+=subset
    res = [list(l) for l in all_subsets if l != ()]
    return res

#Regret-based Elicitation, Contains the model for the incremental elicitation process
class RBE():
    
    def __init__(self, nb_criteres, agr_func, X, log = False):
        self.model = gp.Model("RBE")
        self.logging = log
        self.model.Params.LogToConsole = 0 #Disable the console output
        self.n_crit = nb_criteres
        self.n_queries = 0
        self.agr_func = agr_func
        self.pairs = []
        self.comb = get_all_combinations(self.n_crit) #For the Choquet integral
        self.init_model()
        self.init_MR = np.max([self.MR(x, X) for x in X]) #Set the initial MR, wich correspond to the initial MR without any queries

        """if self.logging:
            self.model.write('model' + str(self.n_queries) + '.LP') #Write the model for the current state of the elicitation process"""

    def init_model(self):
        if self.agr_func == "WS" or self.agr_func == "OWA":
            self.weight_var = [self.model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=1, name="w"+str(w)) for w in range(self.n_crit)]
            self.model.addConstr(gp.quicksum(self.weight_var) == 1)
        elif self.agr_func == "Choquet":
            #We add as many variables as there are subsets of criteria
            self.weight_var = [self.model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=1, name="w"+str(w)) for w in range(2**self.n_crit-1)]

    #Insert new query corresponding to a new pair (x, y) such as x is known to be preffered to y
    def insert_pair(self, pair):
        self.pairs.append(pair)
        self.n_queries += 1
        x = pair[0]
        y = pair[1]
        if self.agr_func == "WS":
            self.model.addConstr(gp.quicksum(self.weight_var[i]*(x[i]-y[i]) for i in range(self.n_crit)) >= 0)
        elif self.agr_func == "OWA":
            #sort x and y
            x_temp = np.sort(x)
            y_temp = np.sort(y)
            self.model.addConstr(gp.quicksum(self.weight_var[i]*(x_temp[i]-y_temp[i]) for i in range(self.n_crit)) >= 0)
        elif self.agr_func == "Choquet":
            #sort x and y
            x_temp = np.sort(x)
            y_temp = np.sort(y)
            weights_x = self.get_weight_choquet(x)
            weights_y = self.get_weight_choquet(y)
            self.model.addConstr(self.weight_var[weights_x[0]]*x_temp[0] - self.weight_var[weights_y[0]]*y_temp[0] >= 0)
            self.model.addConstr(gp.quicksum(self.weight_var[weights_x[i]]*(x_temp[i]-x_temp[i-1]) - self.weight_var[weights_y[i]]*(y_temp[i]-y_temp[i-1]) for i in range(1,self.n_crit)) >= 0 )
            self.add_convexity_constraint()
                
    
    #Return which criteria get which weight
    def get_weight_choquet(self, x):
        ind_x = np.argsort(x)
        temp = list(range(self.n_crit))
        weights = [self.comb.index(temp)]
        for i in range(self.n_crit - 1):
            temp.remove(ind_x[i])
            weights.append(self.comb.index(temp))
        return weights
    
    #Add constraints to the model to ensure convexity for the Choquet integral
    def add_convexity_constraint(self):
        added = set()
        for i in self.comb:
            for j in self.comb:
                #if i != j and the pair is not already in added
                if i != j and (frozenset(i),frozenset(j)) not in added and (frozenset(j),frozenset(i)) not in added:
                    added.add((frozenset(i),frozenset(j)))
                    A_U_B = list(set(i).union(set(j)))
                    A_I_B = list(set(i).intersection(set(j)))
                    if len(A_I_B) == 0:
                        self.model.addConstr(self.weight_var[self.comb.index(A_U_B)] >= self.weight_var[self.comb.index(i)] + self.weight_var[self.comb.index(j)])
                    else:
                        self.model.addConstr(self.weight_var[self.comb.index(A_U_B)] + self.weight_var[self.comb.index(A_I_B)] >= self.weight_var[self.comb.index(i)] + self.weight_var[self.comb.index(j)])

    #The Minimax Regret (MMR) over X 
    def MMR(self, X):
        all_mr = [self.MR_normalized(x, X) for x in X]
        ind = np.argmin(all_mr)
        return all_mr[ind], X[ind]

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
        elif self.agr_func == "OWA":
            #sort x and y
            x_temp = np.sort(x)
            y_temp = np.sort(y)
            self.model.setObjective(gp.quicksum(self.weight_var[i]*(y_temp[i]-x_temp[i]) for i in range(self.n_crit)), gp.GRB.MAXIMIZE)
        elif self.agr_func == "Choquet":
            #sort x and y
            x_temp = np.sort(x)
            y_temp = np.sort(y)
            weights_x = self.get_weight_choquet(x)
            weights_y = self.get_weight_choquet(y)
            first = self.weight_var[weights_y[0]]*y_temp[0] - self.weight_var[weights_x[0]]*x_temp[0]
            #add first to the objective
            self.model.setObjective(first + gp.quicksum(self.weight_var[weights_y[i]]*(y_temp[i]-y_temp[i-1]) - self.weight_var[weights_x[i]]*(x_temp[i]-x_temp[i-1]) for i in range(1,self.n_crit)), gp.GRB.MAXIMIZE)

        self.model.update()
        self.model.optimize()
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
        """if self.logging:
            self.model.write('model' + str(self.n_queries) + '.LP') #Write the model for the current state of the elicitation process, will write the last pair from PMR compared"""
        return self.MMR(X)
        
#Decision Maker, used to ask queries
class DM():
    
    def __init__(self, nb_criteres, agr_func):
        self.n_crit = nb_criteres
        self.agr_func = agr_func
        self.comb = get_all_combinations(self.n_crit)
        self.weights = self.get_random_weights()

    #Return a random vector of weights of size n_crit, summing to 1
    def get_random_weights(self):
        if self.agr_func == "WS":
            return np.random.dirichlet(np.ones(self.n_crit),size=1)[0]
        elif self.agr_func == "OWA": #OWA with decreasing weights
            weights = np.random.dirichlet(np.ones(self.n_crit),size=1)[0]
            weights = np.sort(weights)
            return weights[::-1]
        elif self.agr_func == "Choquet":
            #Generate a random convex Choquet integral
            weights_len = np.zeros(self.n_crit)
            weights = []
            for i in range(1, self.n_crit):
                list_i = [l for l in self.comb if len(l) == i]
                total = 0
                for _ in list_i:
                    r = random.random() + weights_len[i-1]
                    weights.append(r)
                    total += r
                weights_len[i-1] += total
                weights_len[i] += weights_len[i-1]

            last = [l for l in self.comb if len(l) == self.n_crit] #last element, which is v(N)
            for _ in last: 
                r = random.random() + weights_len[self.n_crit-1]
                weights.append(r)
            #Normalize weights
            weights = np.array(weights)
            return weights / weights.sum()

    
    def compare(self, x, y):
        if self.agr_func == "WS":
            return np.sum(self.weights*x) >= np.sum(self.weights*y)
        elif self.agr_func == "OWA":
            x_t = np.sort(x)
            y_t = np.sort(y)
            return np.sum(self.weights*x_t) >= np.sum(self.weights*y_t)
        elif self.agr_func == "Choquet":
            return self.get_choquet_value(x) >= self.get_choquet_value(y)
    
    def get_opt(self, X):
        if self.agr_func == "WS":
            res = [np.argmax([np.sum(self.weights*x) for x in X])]
            return X[np.argmax(res)], np.max(res)
        if self.agr_func == "OWA":
            res = [np.argmax([np.sum(self.weights*np.sort(x)) for x in X])]
            return X[np.argmax(res)], np.max(res)
        if self.agr_func == "Choquet":
            res = [self.get_choquet_value(x) for x in X]
            return X[np.argmax(res)], np.max(res)
    
    def get_value(self, x):
        if self.agr_func == "WS":
            return np.sum(self.weights*x)
        if self.agr_func == "OWA":
            return np.sum(self.weights*np.sort(x))
        if self.agr_func == "Choquet":
            return self.get_choquet_value(x)
        
    def get_choquet_value(self, x):
        ind_x = np.argsort(x)
        temp = list(range(self.n_crit))
        weights = [self.comb.index(temp)]
        for i in range(self.n_crit - 1):
            temp.remove(ind_x[i])
            weights.append(self.comb.index(temp))
        x_temp = np.sort(x)
        return np.sum([self.weights[weights[0]]*x_temp[0]] + [self.weights[weights[i]]*(x_temp[i]-x_temp[i-1]) for i in range(1,self.n_crit)])