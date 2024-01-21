import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gurobipy as gp

def readFile(filename,w,v):
	f = open(filename, "r")
	i=0
	for line in f:
		if line[0]=="i":
			data = line.split()
			w[i]=int(data[1])
			for j in range(v.shape[1]):
				v[i,j]=int(data[2+j])
			i=i+1
		else:
			if line[0]=="W":
				data = line.split()	
				W=int(data[1])
	f.close()
	return W

def readPoints(filename,p):
	f = open(filename, "r")
	nbPND = 0
	for line in f:
		nbPND += 1
	YN = np.zeros((nbPND,p))
	f = open(filename, "r")
	i=0
	for line in f:
		data = line.split()
		for j in range(p):
			YN[i][j]=int(data[j])
		i=i+1
	f.close()
	return YN

#Ordererd weight average fonction d'agrégation
def OWA(costs, weights):
    return np.sum(np.sort(costs)*weights)

#Weighted sum fonction d'agrégation
def WS(costs, weights):
    return np.sum(costs*weights)

#Pairwise max regret selon la fonction d'agrégation
def PMR(x, y, weights, Fp):
	res = []
	for agr in Fp:
		res.append(agr(x, weights) - agr(y, weights))
	return np.max(res)

#Get optimal solution for a weighted sum function using Gurobi
def get_opt_ws(v, w, W, weights): #v value of each object, w weight of each object, W max weight of the knapsack, weights weights of the Choquet integral
	model = gp.Model("ws")
	model.Params.LogToConsole = 0
	x = [model.addVar(vtype=gp.GRB.BINARY, name="x"+str(i)) for i in range(v.shape[0])]
	model.setObjective(gp.quicksum(weights[j]*v[i][j]*x[i] for i in range(len(x)) for j in range(len(weights))), gp.GRB.MAXIMIZE)
	model.addConstr(gp.quicksum(w[i]*x[i] for i in range(v.shape[0])) <= W)
	model.update()
	model.optimize()
	objets = []
	for v in model.getVars(): #Only var is x
		if v.X == 1:
			objets.append(v.varName[1:])
	model.write('modelws.LP')
	return (objets, model.objVal)

#Get optimal solution for a ordered weighted average function using Ogryczak formulation, weights sorted in descending order (monotonic)
def get_opt_owa(v, w, W, weights): 
	model = gp.Model("owa")
	model.Params.LogToConsole = 0
	w_p = [weights[i] - weights[i+1] for i in range(len(weights)-1)] + [weights[-1]]
	m = len(w_p)
	r = [model.addVar(vtype=gp.GRB.CONTINUOUS,name="r"+str(k)) for k in range(m)]
	d = [[model.addVar(vtype=gp.GRB.CONTINUOUS,lb=0,name="d"+str(i)+"_"+str(k)) for i in range(m)] for k in range(m)]
	x = [model.addVar(vtype=gp.GRB.BINARY, name="x"+str(i)) for i in range(v.shape[0])]
	model.setObjective(gp.quicksum((k+1)*w_p[k]*r[k] - gp.quicksum(w_p[k]*d[i][k] for i in range(m)) for k in range(m)), gp.GRB.MAXIMIZE)
	model.addConstr(gp.quicksum(w[i]*x[i] for i in range(v.shape[0])) <= W)
	for i in range(m):
		yi = gp.quicksum(v[:,i] * x)
		for k in range(m):
			model.addConstr(d[i][k] >= r[k] - yi)
	model.update()
	model.optimize()
	objets = []
	for v in model.getVars():
		if v.X == 1 and v.varName[0] == 'x':
			objets.append(v.varName[1:])
	model.write('modelowa.LP')
	return (objets, model.objVal)

#Get optimal solution for a Choquet integral function using Gurobi
def get_opt_choquet(v, w, W, weights): 
	#TODO
	return None
