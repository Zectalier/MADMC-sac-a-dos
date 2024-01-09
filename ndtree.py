import numpy as np

DEFAULT_SIZE = 6
DEFAULT_NUMBER_OF_CHILDREN = 2
IS_MAXIMISATION = True

#Retourne true si u domine v, false sinon
def pareto_dominance_maximisation(u, v):
    return np.all(u >= v) and np.any(u > v)

class NDTree:

    class Node:
        def __init__(self, parent = None, children = [], l_points = dict(), solutions = dict(), nadir = None, ideal = None):
            self.parent = parent
            self.children = children
            
            self.l_points = l_points #l_points is the L set of points from the paper, here it's a dict
            self.solutions = solutions #solutions is the solution associated for each point in costs
            
            self.nadir = nadir
            self.ideal = ideal

        #return the union of the points of the node and its children
        def get_S_points(self):
            s_points = self.l_points.copy()
            s_points = self.rec_get_S_points( s_points)
            return s_points
        
        def rec_get_S_points(self, points):
            if len(self.children) == 0:
                return self.l_points
            else:
                for child in self.children:
                    print(child.l_points)
                    points.update(child.rec_get_points(points))
            return points
        
    def __init__(self, root = None, max_size = 6, nb_children = 2, id = 0, pareto_dom_func = None, is_maximisation = True):
        self.root = root
        self.max_size = max_size
        self.nb_children = nb_children
        self.id = id

        self.pareto_dom_func = pareto_dom_func

        self.IS_MAXIMISATION = is_maximisation

    def delete_sub_tree(self, N):
        l_points = N.l_points
        to_remove = l_points.keys()
        if N.parent is None:
            return 
        else:
            self.remove_points_parent(N, to_remove)
            N.parent.nodes.remove(N)

    #Remove each point to_remove of l_points for the node N and all its parents
    def remove_points_all_parent(self, N, to_remove):
        if N.parent is None:
            return
        else:
            self.remove_points(N, to_remove)
            self.remove_points_parent(N.parent, to_remove)
        
    #Remove each point of l_points from the list of key to remove
    def remove_points(self, N, to_remove):
        for key in to_remove:
            del N.l_points[key]
            del N.solutions[key]
        return
    
    #Return True if u covers v, False otherwise
    def coverage(self, u, v):
        if np.array_equal(u, v) or self.pareto_dom_func(u, v):
            return True
        return False
    
    #Update the NDTree, new_cost must be a np.array
    def update_tree(self, new_cost, new_solution):
        if self.root == None:
            self.root = self.Node(l_points = {self.id: new_cost}, solutions = {self.id: new_solution}, nadir = np.copy(new_cost), ideal = np.copy(new_cost))
            self.id += 1
            return True
        else:
            N = self.root
            if self.update_node(N, new_cost):
                self.insert(N, new_cost, new_solution)
                return True
        return False

    def update_node(self, N, new_cost):
        if self.coverage(N.nadir, new_cost):
            return False
        elif self.coverage(new_cost, N.ideal):
            self.delete_sub_tree(N)
            return True
        elif self.pareto_dom_func(N.ideal, new_cost) or self.pareto_dom_func(new_cost, N.nadir):
            if len(N.children) == 0: #is leaf node
                for key, z in N.l_points.items():
                    if self.pareto_dom_func(z, new_cost):
                        return False
                    elif self.coverage(new_cost, z):
                        self.remove_points_all_parent(N, [key])
            else:
                for child in N.children:
                    if not(self.update_node(child, new_cost)):
                        return False
                    else:
                        #if child is empty, remove it
                        if len(child.l_points) == 0:
                            N.children.remove(child)
                #if there is only one child remaining, remove n and use the child as new n
                if len(N.children) == 1:
                    child = N.children[0]
                    N.l_points = child.l_points
                    N.solutions = child.solutions
                    N.nadir = child.nadir
                    N.ideal = child.ideal
                    N.children = child.children
        else:
            #skip this node
            pass
        return True
    
    def update_ideal_nadir(self, N, new_cost):
        changed = False
        #Check if any component of new_cost is better than the corresponding component of N.nadir and N.ideal
        if self.IS_MAXIMISATION:
            for i in range(len(new_cost)):
                if new_cost[i] > N.nadir[i]:
                    N.nadir[i] = new_cost[i]
                    changed = True
                if new_cost[i] < N.ideal[i]:
                    N.ideal[i] = new_cost[i]
                    changed = True
        else:
            for i in range(len(new_cost)):
                if new_cost[i] < N.nadir[i]:
                    N.nadir[i] = new_cost[i]
                    changed = True
                if new_cost[i] > N.ideal[i]:
                    N.ideal[i] = new_cost[i]
                    changed = True
        if changed:
            if N.parent is not None:
                self.update_ideal_nadir(N.parent, new_cost)
        return
    
    #Find the point that is the furthest apart from all other points in N.l_points
    def find_furthest_point(self, N):
        max_dist = 0
        max_point = None
        for key1, z1 in N.l_points.items():
            for key2, z2 in N.l_points.items():
                dist = np.linalg.norm(z1 - z2)
                if dist > max_dist:
                    max_dist = dist
                    max_point = z1
                    max_key = key1
        return max_point, max_key
    
    #Find the child of N that is closest to z.
    def find_closest_child(self, N, z):
        min_dist = np.inf
        min_child = None
        for child in N.children:
            middle = (child.nadir + child.ideal) / 2
            dist = np.linalg.norm(z - middle)
            if dist < min_dist:
                min_dist = dist
                min_child = child
        return min_child
    
    def split(self, N):
        #Find the point that is the furthest apart from all other points in N.l_points
        max_point, max_key = self.find_furthest_point(N)
        
        #Create a new child with l_points = max_point
        child_node = self.Node(parent = N, l_points = {max_key: max_point}, solutions = {max_key: N.solutions[max_key]}, nadir = np.copy(max_point), ideal = np.copy(max_point))
        N.children.append(child_node)
        lpoint = N.l_points.pop(max_key)
        solution = N.solutions.pop(max_key)
        self.update_ideal_nadir(child_node, max_point)

        while len(N.children) < self.nb_children and len(N.l_points) > 0: #while N has less than nb_children and there are still points in l_points
            #Find the point that is the furthest apart from all other points in N.l_points
            max_point, max_key = self.find_furthest_point(N)

            #Create a new child with an l_points = max_point
            new_child = self.Node(parent = N, l_points = {max_key: max_point}, solutions = {max_key: N.solutions[max_key]}, nadir = np.copy(max_point), ideal = np.copy(max_point))
            N.children.append(new_child)
            N.l_points.pop(max_key)
            N.solutions.pop(max_key)
            self.update_ideal_nadir(new_child, max_point)
            
        while len(N.l_points) > 0: #while there are still points in l_points
            z = N.l_points.popitem() #return a tuple (key, value)
            N.solutions.pop(z[0])
            #Find the child of N that is closest to z.
            min_child = self.find_closest_child(N, z[1])
            min_child.l_points[z[0]] = z[1]
            self.update_ideal_nadir(min_child, z[1])
        return
    
    def insert(self, N, new_cost, new_solution):
        if len(N.children) == 0: #if N is leaf node
            N.l_points[self.id] = new_cost
            N.solutions[self.id] = new_solution
            self.id += 1
            self.update_ideal_nadir(N, new_cost)
            if len(N.l_points) > self.max_size:
                self.split(N)
        else:
            min_child = self.find_closest_child(N, new_cost)
            self.insert(min_child, new_cost, new_solution)
        return