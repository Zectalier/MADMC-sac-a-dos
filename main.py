import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import argparse, sys

from incremental_local_search import *
from ndtree import *
from pareto import *
from func import *
from voisinage import *
from paretolocalsearch import *
from gurobi import *

N_CRITERES = 3
N_OBJETS = 50
AGR_FUNC = "WS"
EPS = 0.001

def main():
	global N_CRITERES, N_OBJETS, AGR_FUNC, EPS
	parser=argparse.ArgumentParser()
	parser.add_argument('-p', help='which process to run')
	parser.add_argument('-c', help='number of criteria')
	parser.add_argument('-o', help='number of objects')
	parser.add_argument('-f', help='aggregation function')
	parser.add_argument('-eps', help='epsilon for which the algorithm should stop')
	parser.add_argument('-a', help='do all functions and save results')
	parser.add_argument('-l', help='log the results')

	args=parser.parse_args()

	w=np.zeros(200,dtype=int)
	v=np.zeros((200,6),dtype=int)
	filename = "data/2KP200-TA-0.dat"
	W = readFile(filename,w,v)

	#Reshape v and w to only take first the required number of objects and criteria
	v = v[:N_OBJETS]
	v = v[:,:N_CRITERES]
	w = w[:N_OBJETS]

	#W equals the sum of the weights divided by 2
	W = int(w.sum()/2)

	#Generate m random solutions
	m=1000

	if args.c == None:
		N_CRITERES = 3
	elif int(args.c) > 6:
		print("Warning: Invalid number of criteria. Maximum is 6, using default value, 3.")
	elif int(args.c) < 2:
		print("Warning: Invalid number of criteria. Minimum is 2, using default value, 3.")
	else:
		N_CRITERES = int(args.c)

	if args.o == None:
		N_OBJETS = 50
	elif int(args.o) > 200:
		print("Warning: Invalid number of objects. Maximum is 200, using default value, 50.")
	elif int(args.o) < 2:
		print("Warning: Invalid number of objects. Minimum is 2, using default value, 50.")
	else:
		N_OBJETS = int(args.o)

	if args.eps == None:
		pass
	elif float(args.eps) > 1:
		print("Warning: Invalid epsilon. Maximum is 1, using default value, 0.05.")
	elif float(args.eps) < 0:
		print("Warning: Invalid epsilon. Minimum is 0, using default value, 0.05.")
	else:
		EPS = float(args.eps)

	match args.f:
		case "OWA":
			AGR_FUNC = "OWA"
		case "WS":
			AGR_FUNC = "WS"
		case "Choquet":
			AGR_FUNC = "Choquet"
		case _:
			print("Warning: Invalid or not specified aggregation function. Valid are \"OWA\", \"WS\" or \"Choquet\", using default value, \"WS\".")
			AGR_FUNC = "WS"

	if args.l == "True":
		log = True
	else:
		log = False

	costs = {}
	solutions = {}

	for i in range(m):
		w_total = 0
		current_solution = set()
		current_cost =  np.zeros(v.shape[1])
		arr = np.arange(w.shape[0])
		np.random.shuffle(arr)
		for j in arr:
			if w_total + w[j] <= W:
				current_solution.add(j)
				next_item = v[j]
				w_total += w[j]
				for k in range(v.shape[1]):
					current_cost[k] += next_item[k]
		costs[i] = current_cost

		solutions[i] = current_solution

	if args.a == "True":
		dm_ws = DM(N_CRITERES, "WS")
		dm_owa = DM(N_CRITERES, "OWA")
		dm_choq = DM(N_CRITERES, "Choquet")
		dms = np.array([str(dm_ws.weights), str(dm_owa.weights), str(dm_choq.weights)]).reshape(1,3)
		dms = pd.DataFrame(dms)
		dms.columns = ["WS", "OWA", "Choquet"]
		if log:
			#if directory does not exist, create it
			if not os.path.exists("./logs/process1/"):
				os.makedirs("./logs/process1/")
			if not os.path.exists("./logs/process2/"):
				os.makedirs("./logs/process2/")
			#Increment the log file name if it already exists
			filename = "./logs/process1/" + str(N_CRITERES) + "c_" + str(N_OBJETS) + "o_" + str(EPS) + "eps_" + "all" + "_DMs"
			i = 1
			while os.path.isfile(filename + "_" + str(i) + ".csv"):
				i += 1
			filename = filename + "_" + str(i) + ".csv"
			dms.to_csv(filename,index=False)

			filename = "./logs/process2/" + str(N_CRITERES) + "c_" + str(N_OBJETS) + "o_" + str(EPS) + "eps_" + "all" + "_DMs"
			filename = filename + "_" + str(i) + ".csv"
			dms.to_csv(filename,index=False)


		print("------------------\nStarting process 1\n------------------")
		procedure_1_all(costs, solutions, v, w, W, dm_ws, dm_owa, dm_choq, log)
		print("------------------\nStarting process 2\n------------------")
		procedure_2_all(costs, solutions, v, w, W, dm_ws, dm_owa, dm_choq, log)
	else:
		if args.p == "1" or args.p == None:
			procedure_1(costs, solutions, v, w, W, log)
		elif args.p == "2":
			procedure_2(costs, solutions, v, w, W, log)
		else:
			print("Warning: Invalid process. Valid are \"1\" or \"2\", using default value, \"1\".")
			procedure_1(costs, solutions, v, w, W, log)
	

def procedure_1(costs, solutions, v, w, W, logging = False):
	global N_CRITERES, N_OBJETS, AGR_FUNC, EPS
	print("Starting Local Search ... \n")
	time_start = time.time()
	
	pls = PLS(costs, solutions, v, w, W)

	id = np.array(list(pls[0].keys()))
	X = np.array(list(pls[0].values()))

	dm = DM(N_CRITERES, AGR_FUNC)
	rbe = RBE(N_CRITERES, AGR_FUNC, X, log=logging)
	print("\nWeights for the Decision Maker " + str(dm.weights) + "\n")
	
	opt_cost = np.zeros(N_CRITERES)
	if(AGR_FUNC == "WS"):
		obj, obj_value = get_opt_ws(v, w, W, dm.weights)
	elif(AGR_FUNC == "OWA"):
		obj, obj_value = get_opt_owa(v, w, W, dm.weights)
	elif(AGR_FUNC == "Choquet"):
		print("Warning: Optimal for Choquet integral not implemented yet, using optimal for the current solutions found instead.")
		obj, obj_value = dm.get_opt(X)
	for i in obj:
		opt_cost += v[int(i)]
	print("Optimal solution (using LP and gurobi) for the Decision Maker " + str(opt_cost) + "\n")

	res_init = rbe.MMR(X)
	MMR_value = res_init[0]
	print("Initial MMR value before any query: " + str(res_init[0]) + ", found for the solution " + str(res_init[1]) + "\n")

	if logging:
		l = np.array([rbe.n_queries, res_init[0], str(res_init[1]), str(opt_cost), str(dm.get_value(res_init[1])/obj_value),  time.time() - time_start]).reshape(1,6)
		log = pd.DataFrame(l)
		log.columns = ["Query", "MMR", "MMR Solution", "Actual Solution", "Error rate", "Time since start"]

	print("Starting Incremental Elicitation ... \n")
	#Incremental Elicitation
	while MMR_value > EPS:
		res = rbe.CSS_ask_query(X, dm)
		MMR_value = res[0]
		print("Query " + str(rbe.n_queries) + ": " + str(rbe.pairs[-1]))
		print("MMR value for the query: " + str(res[0]) + ", found for the solution " + str(res[1]) + "\n")

		if logging:
			l = np.array([rbe.n_queries, res[0], str(res[1]), str(opt_cost), str(dm.get_value(res[1])/obj_value), time.time() - time_start])
			log.loc[len(log.index)] = l

	obj = pls[1][id[np.where(X == res[1])[0]][0]]
	print("End of Incremental Elicitation, MMR final value:" + str(MMR_value) + "\n")
	print("Solution opt found: " + str(res[1]) + " with objects: " + str(obj) + "\n")
	print("Actual opt Solution: " + str(dm.get_opt(X)))
	if logging:
		#if directory does not exist, create it
		if not os.path.exists("./logs/process1/"):
			os.makedirs("./logs/process1/")
		#Increment the log file name if it already exists
		filename = "./logs/process1/" + str(N_CRITERES) + "c_" + str(N_OBJETS) + "o_" + str(EPS) + "eps_" + str(AGR_FUNC) + "_single_log"
		i = 1
		while os.path.isfile(filename + "_" + str(i) + ".csv"):
			i += 1
		filename = filename + "_" + str(i) + ".csv"
		log.to_csv(filename,index=False)
	
	print("End")


def procedure_2(costs, solutions, v, w, W, logging = False):
	global N_CRITERES, N_OBJETS, AGR_FUNC, EPS

	dm = DM(N_CRITERES, AGR_FUNC)

	opt_cost = np.zeros(N_CRITERES)
	if(AGR_FUNC == "WS"):
		obj, obj_value = get_opt_ws(v, w, W, dm.weights)
	elif(AGR_FUNC == "OWA"):
		obj, obj_value = get_opt_owa(v, w, W, dm.weights)
	elif(AGR_FUNC == "Choquet"):
		print("Warning: Optimal for Choquet integral not implemented yet.")
		obj = np.zeros(N_OBJETS)
		obj_value = np.zeros(N_CRITERES)
	for i in obj:
		opt_cost += v[int(i)]
	print("Optimal solution (using LP and gurobi) for the Decision Maker " + str(opt_cost) + "\n")

	print("Starting Incremental Local Search ... \n")
	time_start = time.time()
	X, res, total_queries = RBLS(costs, solutions, v, w, W, dm, AGR_FUNC, EPS)
	time_end = time.time() - time_start

	print("End of Incremental Local Search, Solution found:" + str(res[0]) + " with objects: " + str(res[1]))
	print("Actual opt Solution: " + str(opt_cost) + "\n")
	print("Total number of queries done: " + str(total_queries) + "\n")

	if logging:
		if not os.path.exists("./logs/process2/"):
			os.makedirs("./logs/process2/")
		#Increment the log file name if it already exists
		filename = "./logs/process2/" + str(N_CRITERES) + "c_" + str(N_OBJETS) + "o_" + str(EPS) + "eps_" + str(AGR_FUNC) + "_single_log"
		i = 1
		while os.path.isfile(filename + "_" + str(i) + ".csv"):
			i += 1
		filename = filename + "_" + str(i) + ".csv"
		l = np.array([total_queries, str(res[0]), str(opt_cost), str(dm.get_value(res[0])/obj_value), time_end]).reshape(1,5)
		log = pd.DataFrame(l)
		log.columns = ["Number Queries", "Approx Solution", "Actual Solution", "Error rate", "Time"]
		log.to_csv(filename,index=False)

def procedure_1_all(costs, solutions, v, w, W, dm_ws, dm_owa, dm_choq, logging = False): #Used for comparing the different aggregation functions, so they can use the same results from the PLS
	time_start = time.time()
	pls = PLS(costs, solutions, v, w, W)
	pls_time = time.time() - time_start
	if logging:
		if not os.path.exists("./logs/process1/"):
			os.makedirs("./logs/process1/")
		#Increment the log file name if it already exists
		filename = "./logs/process1/" + str(N_CRITERES) + "c_" + str(N_OBJETS) + "o_" + str(EPS) + "eps_" + "all" + "_timePLS"
		i = 1
		while os.path.isfile(filename + "_" + str(i) + ".csv"):
			i += 1
		filename = filename + "_" + str(i) + ".csv"
		timelog = pd.DataFrame(np.array([pls_time]))
		timelog.to_csv(filename,index=False)

	id = np.array(list(pls[0].keys()))
	X = np.array(list(pls[0].values()))

	#WS
	AGR_FUNC = "WS"

	rbe = RBE(N_CRITERES, AGR_FUNC, X, log=logging)
	opt_cost = np.zeros(N_CRITERES)
	obj, obj_value = get_opt_ws(v, w, W, dm_ws.weights)
	for i in obj:
		opt_cost += v[int(i)]
	ws_start = time.time() #We start timer here, so it's not affected by the time it takes to get the optimal solution
	res_init = rbe.MMR(X)
	MMR_value = res_init[0]
	if logging:
		l = np.array([rbe.n_queries, res_init[0], str(res_init[1]), str(opt_cost), str(dm_ws.get_value(res_init[1])/obj_value), time.time() - ws_start]).reshape(1,6)
		log = pd.DataFrame(l)
		log.columns = ["Query", "MMR", "MMR Solution", "Actual Solution", "Error rate", "Time since start"]
	
	while MMR_value > EPS:
		res = rbe.CSS_ask_query(X, dm_ws)
		MMR_value = res[0]
		if logging:
			l = np.array([rbe.n_queries, res[0], str(res[1]), str(opt_cost), str(dm_ws.get_value(res[1])/obj_value), time.time() - ws_start])
			log.loc[len(log.index)] = l

	if logging:
		#Increment the log file name if it already exists
		filename = "./logs/process1/" + str(N_CRITERES) + "c_" + str(N_OBJETS) + "o_" + str(EPS) + "eps_" + str(AGR_FUNC) + "_all_log"
		i = 1
		while os.path.isfile(filename + "_" + str(i) + ".csv"):
			i += 1
		filename = filename + "_" + str(i) + ".csv"
		log.to_csv(filename,index=False)
	
	#OWA
	AGR_FUNC = "OWA"

	rbe = RBE(N_CRITERES, AGR_FUNC, X, log=logging)
	opt_cost = np.zeros(N_CRITERES)
	obj, obj_value = get_opt_owa(v, w, W, dm_owa.weights)
	for i in obj:
		opt_cost += v[int(i)]
	owa_start = time.time()
	res_init = rbe.MMR(X)
	MMR_value = res_init[0]
	if logging:
		l = np.array([rbe.n_queries, res_init[0], str(res_init[1]), str(opt_cost), str(dm_owa.get_value(res_init[1])/obj_value), time.time() - owa_start]).reshape(1,6)
		log = pd.DataFrame(l)
		log.columns = ["Query", "MMR", "MMR Solution", "Actual Solution", "Error rate", "Time since start"]
	
	while MMR_value > EPS:
		res = rbe.CSS_ask_query(X, dm_owa)
		MMR_value = res[0]
		if logging:
			l = np.array([rbe.n_queries, res[0], str(res[1]), str(opt_cost), str(dm_owa.get_value(res[1])/obj_value), time.time() - owa_start])
			log.loc[len(log.index)] = l

	if logging:
		#Increment the log file name if it already exists
		filename = "./logs/process1/" + str(N_CRITERES) + "c_" + str(N_OBJETS) + "o_" + str(EPS) + "eps_" + str(AGR_FUNC) + "_all_log"
		i = 1
		while os.path.isfile(filename + "_" + str(i) + ".csv"):
			i += 1
		filename = filename + "_" + str(i) + ".csv"
		log.to_csv(filename,index=False)

	#Choquet
	AGR_FUNC = "Choquet"

	rbe = RBE(N_CRITERES, AGR_FUNC, X, log=logging)
	print("Warning: Optimal for Choquet integral not implemented yet, using optimal for the current solutions found instead.")
	opt_cost, obj_value = dm_choq.get_opt(X)
	choquet_start = time.time()
	res_init = rbe.MMR(X)
	MMR_value = res_init[0]
	if logging:
		l = np.array([rbe.n_queries, res_init[0], str(res_init[1]), str(opt_cost), str(dm_choq.get_value(res_init[1])/obj_value), time.time() - choquet_start]).reshape(1,6)
		log = pd.DataFrame(l)
		log.columns = ["Query", "MMR", "MMR Solution", "Actual Solution", "Error rate", "Time since start"]
	
	while MMR_value > EPS:
		res = rbe.CSS_ask_query(X, dm_choq)
		MMR_value = res[0]
		if logging:
			l = np.array([rbe.n_queries, res[0], str(res[1]), str(opt_cost), str(dm_choq.get_value(res_init[1])/obj_value), time.time() - choquet_start])
			log.loc[len(log.index)] = l

	if logging:
		#Increment the log file name if it already exists
		filename = "./logs/process1/" + str(N_CRITERES) + "c_" + str(N_OBJETS) + "o_" + str(EPS) + "eps_" + str(AGR_FUNC) + "_all_log"
		i = 1
		while os.path.isfile(filename + "_" + str(i) + ".csv"):
			i += 1
		filename = filename + "_" + str(i) + ".csv"
		log.to_csv(filename,index=False)

def procedure_2_all(costs, solutions, v, w, W, dm_ws, dm_owa, dm_choq, logging = False):
	global N_CRITERES, N_OBJETS, AGR_FUNC, EPS

	AGR_FUNC = "WS"
	opt_cost = np.zeros(N_CRITERES)
	obj, obj_value = get_opt_ws(v, w, W, dm_ws.weights)
	for i in obj:
		opt_cost += v[int(i)]
	time_start = time.time()
	X, res, total_queries = RBLS(costs, solutions, v, w, W, dm_ws, AGR_FUNC, EPS, False)
	time_end = time.time() - time_start

	if logging:
		if not os.path.exists("./logs/process2/"):
			os.makedirs("./logs/process2/")
		#Increment the log file name if it already exists
		filename = "./logs/process2/" + str(N_CRITERES) + "c_" + str(N_OBJETS) + "o_" + str(EPS) + "eps_" + str(AGR_FUNC) + "_all_log"
		i = 1
		while os.path.isfile(filename + "_" + str(i) + ".csv"):
			i += 1
		filename = filename + "_" + str(i) + ".csv"
		l = np.array([total_queries, str(res[0]), str(opt_cost),str(dm_ws.get_value(res[0])/obj_value), time_end]).reshape(1,5)
		log = pd.DataFrame(l)
		log.columns = ["Number Queries", "Approx Solution", "Actual Solution", "Error rate", "Time"]
		log.to_csv(filename,index=False)

	AGR_FUNC = "OWA"
	opt_cost = np.zeros(N_CRITERES)
	obj, obj_val = get_opt_ws(v, w, W, dm_owa.weights)
	for i in obj:
		opt_cost += v[int(i)]
	time_start = time.time()
	X, res, total_queries = RBLS(costs, solutions, v, w, W, dm_owa, AGR_FUNC, EPS, False)
	time_end = time.time() - time_start

	if logging:
		if not os.path.exists("./logs/process2/"):
			os.makedirs("./logs/process2/")
		#Increment the log file name if it already exists
		filename = "./logs/process2/" + str(N_CRITERES) + "c_" + str(N_OBJETS) + "o_" + str(EPS) + "eps_" + str(AGR_FUNC) + "_all_log"
		i = 1
		while os.path.isfile(filename + "_" + str(i) + ".csv"):
			i += 1
		filename = filename + "_" + str(i) + ".csv"
		l = np.array([total_queries, str(res[0]), str(opt_cost),str(dm_owa.get_value(res[0])/obj_value), time_end]).reshape(1,5)
		log = pd.DataFrame(l)
		log.columns = ["Number Queries", "Approx Solution", "Actual Solution", "Error rate", "Time"]
		log.to_csv(filename,index=False)
	
	AGR_FUNC = "Choquet"
	opt_cost = np.zeros(N_CRITERES)
	print("Warning: Optimal for Choquet integral not implemented yet.")
	obj = np.zeros(N_OBJETS)
	obj_value = np.zeros(N_CRITERES)
	for i in obj:
		opt_cost += v[int(i)]
	time_start = time.time()
	X, res, total_queries = RBLS(costs, solutions, v, w, W, dm_choq, AGR_FUNC, EPS, False)
	time_end = time.time() - time_start

	if logging:
		if not os.path.exists("./logs/process2/"):
			os.makedirs("./logs/process2/")
		#Increment the log file name if it already exists
		filename = "./logs/process2/" + str(N_CRITERES) + "c_" + str(N_OBJETS) + "o_" + str(EPS) + "eps_" + str(AGR_FUNC) + "_all_log"
		i = 1
		while os.path.isfile(filename + "_" + str(i) + ".csv"):
			i += 1
		filename = filename + "_" + str(i) + ".csv"
		l = np.array([total_queries, str(res[0]), str(opt_cost), str(dm_choq.get_value(res[0])/obj_value), time_end]).reshape(1,5)
		log = pd.DataFrame(l)
		log.columns = ["Number Queries", "Approx Solution", "Actual Solution", "Error rate", "Time"]
		log.to_csv(filename,index=False)


if __name__ == "__main__":
	main()