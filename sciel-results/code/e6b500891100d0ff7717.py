# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import permutations

# Global deadline guard
_START_TIME = None
_TIME_LIMIT = 115.0  # Safety margin from 120s

def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > _TIME_LIMIT

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    _ensure_start_time()
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(current_node)
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # Check time before doing expensive work
    if _time_exceeded():
        # Fallback to nearest neighbor
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    # For small numbers of unvisited nodes, find the exact best next node
    # by evaluating all permutations of the remaining nodes.
    # 8! = 40,320 is acceptable for a single call.
    if n_unvisited <= 8:
        best_cost = float('inf')
        best_first_node = -1
        
        # Generate permutations
        for p in permutations(unvisited_nodes):
            # Path: current -> p[0] -> p[1] -> ... -> p[-1] -> destination
            cost = distance_matrix[current_node, p[0]]
            for i in range(len(p) - 1):
                cost += distance_matrix[p[i], p[i+1]]
            cost += distance_matrix[p[-1], destination_node]
            
            if cost < best_cost:
                best_cost = cost
                best_first_node = p[0]
                
        return int(best_first_node)

    # For larger sets, use a vectorized greedy completion estimation.
    # For each candidate u in U, estimate the cost of a greedy path from u
    # through the remaining nodes in U\{u} to destination_node.
    
    U = unvisited_nodes
    m = len(U)
    
    # Distance from current to all candidates
    d_curr = distance_matrix[current_node, U]
    
    # We will compute a completion cost for each candidate u.
    # completion_cost[u] = estimated cost to go from U[u] to destination_node
    # visiting all other nodes in U exactly once, using a greedy nearest neighbor strategy.
    
    # Initialize remaining nodes mask for each candidate
    # This is a bit tricky to fully vectorize without loops, but we can use a loop
    # over candidates (m <= 50) where each iteration is O(m^2) vectorized.
    # Total O(m^3) but with very small constants due to vectorization.
    
    completion_costs = np.zeros(m)
    
    # Pre-extract the sub-matrix of distances between unvisited nodes
    # sub[i, j] = dist(U[i], U[j])
    sub = distance_matrix[np.ix_(U, U)]
    
    for i in range(m):
        # Candidate is U[i]
        # We need to simulate greedy path from U[i] to destination, visiting U[j] for j != i
        
        # Remaining nodes are all U except U[i]
        # Mask for remaining nodes
        mask = np.ones(m, dtype=bool)
        mask[i] = False
        
        # Current node in simulation is U[i]
        current_sim = i
        cost_sim = 0.0
        
        # We need to visit m-1 nodes.
        # We will do this in a loop, but we can vectorize the "find nearest" step.
        
        # Actually, a full greedy simulation loop is O(m^2) per candidate i.
        # With m=50, that's 2500 operations per candidate, 50 candidates -> 125k ops.
        # This is very fast in Python if we use NumPy for the min finding.
        
        # Let's implement the greedy simulation for candidate i.
        # We maintain a set of unvisited indices in U (excluding i).
        
        # To make it fast, we can just loop m-1 times.
        # In each step, find the nearest unvisited node to current_sim.
        
        # We'll use a boolean array to track visited in the simulation.
        visited_sim = np.zeros(m, dtype=bool)
        visited_sim[i] = True # Start node is 'visited' in the sense that we are at it
        
        curr_idx = i
        
        for _ in range(m - 1):
            # Get distances from curr_idx to all nodes
            dists = sub[curr_idx, :]
            # Mask out visited nodes
            dists[visited_sim] = np.inf
            
            # Find nearest
            next_idx = np.argmin(dists)
            
            # If no unvisited nodes found (should not happen if m-1 > 0)
            if dists[next_idx] == np.inf:
                break
                
            cost_sim += dists[next_idx]
            visited_sim[next_idx] = True
            curr_idx = next_idx
        
        # Finally, add distance from last node to destination
        # Note: destination_node is not in U, so we use the original distance_matrix
        cost_sim += distance_matrix[curr_idx, destination_node]
        
        completion_costs[i] = cost_sim
    
    # Total score for candidate i is d_curr[i] + completion_costs[i]
    scores = d_curr + completion_costs
    
    best_idx = np.argmin(scores)
    return int(U[best_idx])

# EVOLVE-BLOCK-END
