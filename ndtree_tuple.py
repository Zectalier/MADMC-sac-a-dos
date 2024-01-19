import random
import numpy as np
from pareto import *

DEFAULT_SIZE = 6
DEFAULT_NUMBER_OF_CHILDREN = 2
IS_MAXIMISATION = True

#Retourne true si u domine v, false sinon
def pareto_dominance_maximisation(u, v):
    return np.all(u >= v) and np.any(u > v)

class NDTree:

    class Node:
        def __init__(self, parent = None, children = [], l_points = list(), nadir = None, ideal = None):
            self.parent = parent
            self.children = children
            
            self.l_points = l_points #l_points is the L set of points from the paper, here it's a list of points with [cost, solution]
            
            self.nadir = nadir
            self.ideal = ideal

        #return the union of the points of the node and its children
        def get_S_points(self):
            s_points = self.l_points.copy()
            s_points = self.rec_get_S_points(s_points)
            return s_points
        
        def rec_get_S_points(self, points):
            if len(self.children) == 0:
                return self.l_points
            else:
                for child in self.children:
                    points = points + child.rec_get_S_points(points)
            return points
        
        def __str__(self, level=0):
            if level == 0:
                print(str(self.get_S_points())+"\n")
            ret = "\t"*level+"ideal:"+str(self.ideal)+",nadir:"+str(self.nadir)+"\n"
            for child in self.children:
                ret += child.__str__(level+1)
            return ret
        
        def delete_node(self):
            if len(self.children) != 0:
                for child in self.children:
                    child.delete_node()
            self.parent.children.remove(self)
            self.l_points = None
            self.nadir = None
            self.ideal = None
            self.children = None
            del self
        
    def __init__(self, root = None, max_size = DEFAULT_SIZE, nb_children = DEFAULT_NUMBER_OF_CHILDREN, pareto_dom_func = pareto_dominance_maximisation, is_maximisation = True):
        self.root = root
        self.max_size = max_size
        self.nb_children = nb_children

        self.pareto_dom_func = pareto_dom_func

        self.IS_MAXIMISATION = is_maximisation

    def delete_sub_tree(self, N):
        if N.parent is None:
            self.root = None
        else:
            N.delete_node()
    
    def get_root(self):
        return self.root
    
    def get_all_costs(self):
        return self.root.get_S_points()
    
    #Return True if u covers v, False otherwise
    def coverage(self, u, v):
        a_u = np.array(u)
        a_v = np.array(v)
        if np.array_equal(a_u, a_v) or self.pareto_dom_func(a_u, a_v):
            return True
        return False
    
    #Update the NDTree, new_cost must be a np.array
    def update_tree(self, new_cost):
        if self.root == None:
            self.root = self.Node(l_points = [new_cost], nadir = np.copy(new_cost[0]), ideal = np.copy(new_cost[0]))
            return True
        else:
            N = self.root
            if self.update_node(N, new_cost):
                self.insert(N, new_cost)
                return True
        return False

    def update_node(self, N, new_cost):
        if self.coverage(N.nadir, new_cost[0]):
            return False
        elif self.coverage(new_cost[0], N.ideal):
            self.delete_sub_tree(N)
            return True
        elif self.pareto_dom_func(N.ideal, new_cost[0]) or self.pareto_dom_func(new_cost[0], N.nadir):
            if len(N.children) == 0: #is leaf node
                to_remove = []
                for z in N.l_points:
                    if self.pareto_dom_func(z[0], new_cost[0]):
                        return False
                    elif self.coverage(new_cost[0], z[0]):
                        to_remove.append(z)
                for r in to_remove:
                    N.l_points.remove(r)
            else:
                for child in N.children:
                    if not(self.update_node(child, new_cost)):
                        return False
                    else:
                        if child in N.children: #Check if child is still in N.children, can be removed during coverage check
                            #if child is empty, remove it
                            if len(child.get_S_points()) == 0:
                                N.children.remove(child)
                #if there is only one child remaining, remove n and use the child as new n
                if len(N.children) == 1:
                    child = N.children[0]
                    N.l_points = child.l_points
                    N.nadir = child.nadir
                    N.ideal = child.ideal
                    N.children = child.children
                    for c in child.children:
                        c.parent = N
                    del child
        else:
            #skip this node
            pass
        return True
    
    def update_ideal_nadir(self, N, new_cost):
        changed = False
        #Check if any component of new_cost is better than the corresponding component of N.nadir and N.ideal
        if self.IS_MAXIMISATION:
            for i in range(len(new_cost[0])):
                if new_cost[0][i] < N.nadir[i]:
                    N.nadir[i] = new_cost[0][i]
                    changed = True
                if new_cost[0][i] > N.ideal[i]:
                    N.ideal[i] = new_cost[0][i]
                    changed = True
        else:
            for i in range(len(new_cost[0])):
                if new_cost[0][i] > N.nadir[i]:
                    N.nadir[i] = new_cost[0][i]
                    changed = True
                if new_cost[0][i] < N.ideal[i]:
                    N.ideal[i] = new_cost[0][i]
                    changed = True
        if changed:
            if N.parent is not None:
                self.update_ideal_nadir(N.parent, new_cost)
        return
    
    #Find the point that is the furthest apart from all other points in N.l_points
    def find_furthest_point(self, N):
        max_dist = 0
        max_point = None
        for z1 in N.l_points:
            a_z1 = np.array(z1[0])
            for z2 in N.l_points:
                a_z2 = np.array(z2[0])
                dist = np.linalg.norm(a_z1 - a_z2)
                if dist > max_dist:
                    max_dist = dist
                    max_point = z1
        return max_point
    
    #Find the child of N that is closest to z.
    def find_closest_child(self, N, z):
        min_dist = np.inf
        min_child = None
        a_z = np.array(z[0])
        for child in N.children:
            middle = (child.nadir + child.ideal) / 2
            dist = np.linalg.norm(a_z - middle)
            if dist < min_dist:
                min_dist = dist
                min_child = child
        return min_child
    
    def split(self, N):
        #Find the point that is the furthest apart from all other points in N.l_points
        max_point = self.find_furthest_point(N)
        
        #Create a new child with l_points = max_point
        child_node = self.Node(parent = N, children=[], l_points = [max_point], nadir = np.copy(max_point[0]), ideal = np.copy(max_point[0]))
        N.children.append(child_node)
        N.l_points.remove(max_point)
        self.update_ideal_nadir(child_node, max_point)

        while len(N.children) < self.nb_children and len(N.l_points) > 0: #while N has less than nb_children and there are still points in l_points
            #Find the point that is the furthest apart from all other points in N.l_points
            max_point = self.find_furthest_point(N)

            #Create a new child with an l_points = max_point
            new_child = self.Node(parent = N, children=[],  l_points = [max_point], nadir = np.copy(max_point[0]), ideal = np.copy(max_point[0]))
            N.children.append(new_child)
            N.l_points.remove(max_point)
            self.update_ideal_nadir(new_child, max_point)
            
        while len(N.l_points) > 0: #while there are still points in l_points
            z = N.l_points.pop()
            #Find the child of N that is closest to z.
            min_child = self.find_closest_child(N, z)
            min_child.l_points.append(z)
            self.update_ideal_nadir(min_child, z)
        return
    
    def insert(self, N, new_cost):
        if self.root is None:
            self.root = self.Node(l_points = [new_cost], children = [], nadir = np.copy(new_cost[0]), ideal = np.copy(new_cost[0]))
            return
        else:
            if len(N.children) == 0: #if N is leaf node
                N.l_points.append(new_cost)
                self.update_ideal_nadir(N, new_cost)
                if len(N.l_points) > self.max_size:
                    self.split(N)
            else:
                min_child = self.find_closest_child(N, new_cost)
                self.insert(min_child, new_cost)
            return
        
if __name__ == "__main__":
    #TESTS
    #List random de points de test (6 dimensions)
    random.seed(0)
    list_test = []
    for i in range(50):
        list_test.append((np.random.randint(0, 10, 6).tolist(), np.random.randint(1, 50, 10).tolist()))
    #Test de l'insertion dans l'arbre
    tree = NDTree(pareto_dom_func = pareto_dominance_maximisation)
    for cost in list_test:
        tree.update_tree(cost)
    print(list_test)
    print(str(tree.root))
    tree.update_tree(([1, 1, 1, 1, 1, 1], np.random.randint(0, 50, 10).tolist()))
    print(str(tree.root))
    tree.update_tree(([20, 9, 9, 9, 9, 9],  np.random.randint(0, 50, 10).tolist()))
    print(str(tree.root))

    #Verification que le pareto front est bien le même
    print(len(tree.root.get_S_points()))
