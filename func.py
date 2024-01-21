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
	m = gp.Model("ws")
	m.Params.LogToConsole = 0
	x = m.addVars(v.shape[0], vtype=gp.GRB.BINARY, name="x")
	m.setObjective(gp.quicksum(v[i]*x[i] for i in range(v.shape[0])), gp.GRB.MAXIMIZE)
	m.addConstr(gp.quicksum(w[i]*x[i] for i in range(v.shape[0])) <= W)
	m.update()
	m.optimize()
	objets = []
	for v in m.getVars():
		if v.x == 1:
			objets.append(v.varName)
	return (objets, m.objVal)

#Get optimal solution for a ordered weighted average function using Ogryczak formulation
def get_opt_owa(v, w, W, weights): 
	#TODO
	return None

def get_opt_choquet(v, w, W, weights): 
	#TODO
	return None

