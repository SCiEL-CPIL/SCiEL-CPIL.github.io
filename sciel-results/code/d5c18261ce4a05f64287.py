import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Nodes are 2..256
    n = 256
    vertices = list(range(2, n + 1))
    num_vertices = len(vertices)
    
    # Weights: 1/x
    weights = {v: 1.0 / v for v in vertices}
    
    # Build compatibility graph (complement of conflict graph)
    # Edge exists if lcm(i, j) > 256
    # We represent the graph as a list of sets of neighbors
    # To save memory and speed, we use a list of lists or bitsets.
    # Given N=255, a list of sets is fine.
    neighbors = {v: set() for v in vertices}
    
    # Precompute adjacency
    # To speed up, we can iterate i < j
    for i_idx in range(num_vertices):
        i = vertices[i_idx]
        for j_idx in range(i_idx + 1, num_vertices):
            j = vertices[j_idx]
            if lcm(i, j) > 256:
                neighbors[i].add(j)
                neighbors[j].add(i)
                
    # Sort vertices by weight descending for better pruning
    # The order in Bron-Kerbosch affects performance.
    # We will maintain P and X as sets of vertices.
    
    best_sum = 0.0
    best_set = []
    
    # Helper to check time
    def time_check():
        return time.time() > deadline
    
    # Bron-Kerbosch with pivot and weighted pruning
    # R: current clique (list)
    # P: candidates (set)
    # X: excluded (set)
    def bron_kerbosch(R, P, X):
        nonlocal best_sum, best_set
        
        if time_check():
            return
            
        if not P and not X:
            # Maximal clique found
            s = sum(weights[v] for v in R)
            if s > best_sum:
                best_sum = s
                best_set = list(R)
            return
        
        # Pruning: Upper bound on sum
        # Sum of weights in R + Sum of weights in P
        # This is a valid upper bound because we can't pick more than all of P
        current_sum = sum(weights[v] for v in R)
        if current_sum + sum(weights[v] for v in P) <= best_sum:
            return
            
        # Pivot selection
        # Choose u in P U X with maximum degree in P
        union_px = P | X
        if union_px:
            # Find u with max len(neighbors[u] & P)
            u = max(union_px, key=lambda x: len(neighbors[x] & P))
            candidates = P - neighbors[u]
        else:
            candidates = P
            
        for v in list(candidates):
            if time_check():
                return
            R.append(v)
            # New P = P & neighbors[v]
            new_P = P.intersection(neighbors[v])
            # New X = X & neighbors[v]
            new_X = X.intersection(neighbors[v])
            
            bron_kerbosch(R, new_P, new_X)
            
            R.pop()
            P.remove(v)
            X.add(v)
            
    # Initial call
    # All vertices are candidates
    initial_P = set(vertices)
    initial_X = set()
    bron_kerbosch([], initial_P, initial_X)
    
    if not best_set:
        # Fallback if something went wrong or deadline hit before finding anything
        # The greedy from the seed is a safe fallback
        selected = []
        for x in range(2, 257):
            if all(lcm(x, y) > 256 for y in selected):
                selected.append(x)
        return selected
        
    return best_set
