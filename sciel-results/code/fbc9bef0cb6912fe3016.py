# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline guard
_START_TIME = None
_DEADLINE = 115.0  # 5s margin before 120s

def _init_deadline():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_exceeded():
    global _START_TIME
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > _DEADLINE

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    _init_deadline()
    
    if len(unvisited_nodes) == 0:
        return depot

    # Convert to numpy array for fast indexing
    unvisited = np.array(unvisited_nodes, dtype=np.int64)
    
    # Filter by capacity
    demands_arr = np.array(demands, dtype=np.float64)
    feasible_mask = demands_arr[unvisited] <= rest_capacity
    
    if not np.any(feasible_mask):
        return depot
        
    feasible_nodes = unvisited[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot

    # Calculate distances
    dists_from_current = distance_matrix[current_node, feasible_nodes]
    dists_from_depot = distance_matrix[depot, feasible_nodes]
    
    # Classic savings with proximity tie-break:
    # score(j) = d(i,j) - d(depot,j) + lambda * d(i,j) / (d(depot,j) + 1.0)
    # The last term is a proximity penalty that breaks ties among candidates
    # with similar savings, preferring nodes closer to the current position.
    
    lambda_prox = 0.05
    scores = dists_from_current - dists_from_depot + lambda_prox * dists_from_current / (dists_from_depot + 1.0)
    
    # Pick the one with minimum score
    min_idx = np.argmin(scores)
    next_node = feasible_nodes[min_idx]
    
    return int(next_node)
# EVOLVE-BLOCK-END
