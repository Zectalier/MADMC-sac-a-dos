import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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