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
    if n_unvisited <= 7:
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

    # For larger sets, use a multi-factor scoring heuristic
    U = unvisited_nodes
    m = len(U)
    
    # Distance from current to all candidates
    d_curr = distance_matrix[current_node, U]
    
    # Distance from each candidate to destination (exit cost)
    d_dest = distance_matrix[U, destination_node]
    
    # Distance matrix between all candidates
    sub = distance_matrix[np.ix_(U, U)].copy()
    np.fill_diagonal(sub, np.inf)
    
    # 2-step lookahead: min distance from u to any other v in U
    min_next = np.min(sub, axis=1)
    
    # Vectorized nearest-neighbor completion cost estimate:
    # For each candidate u, estimate the cost to complete the tour from u
    # by greedily visiting nearest unvisited nodes until reaching destination.
    # This is O(m^2) per candidate but vectorized across candidates.
    
    # Build a completion cost estimate for each candidate
    # For candidate u: cost = d_curr[u] + estimated_completion_cost(u)
    # where estimated_completion_cost approximates the path u -> ... -> destination
    
    # Quick estimate: for each u, sum of distances along a greedy NN path
    # through remaining nodes to destination
    completion_costs = np.zeros(m)
    
    # Use a simplified vectorized approach:
    # For each candidate, estimate remaining path cost as:
    # (sum of min outgoing edges from each node in U\{u}) / (m-1) * some_factor
    # This is too rough. Instead, use a 3-step lookahead for better accuracy.
    
    # 3-step lookahead: for each u, find min over v,w in U\{u} of
    # d(u,v) + d(v,w) where w != u,v
    # This gives a better estimate of the "local" path cost
    
    # For efficiency, use a combination:
    # score = d_curr + alpha * min_next + beta * d_dest + gamma * (min_next of next)
    
    # Let's use a weighted combination that balances:
    # 1. Immediate cost (d_curr)
    # 2. Local connectivity (min_next)  
    # 3. Exit cost (d_dest)
    
    # Adaptive weights based on position in tour
    # Early in tour: weight immediate cost more
    # Late in tour: weight exit cost more
    progress = (len(unvisited_nodes) - n_unvisited) / 50.0  # rough progress estimate
    
    # Weights that shift from immediate to exit cost
    w_curr = 1.0
    w_next = 0.3 * (1.0 - progress)  # less important as we get closer to end
    w_dest = 0.5 + 0.5 * progress  # more important near end
    
    scores = w_curr * d_curr + w_next * min_next + w_dest * d_dest
    
    best_idx = np.argmin(scores)
    return int(U[best_idx])

# EVOLVE-BLOCK-END
