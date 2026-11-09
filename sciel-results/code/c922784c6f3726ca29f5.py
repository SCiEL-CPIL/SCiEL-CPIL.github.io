"""Heuristic for capacitated vehicle routing: lookahead-based route completion."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state for timing and caching
_start_time = None
_dm_cache = {}

def _get_dm(distance_matrix):
    """Get or create numpy view of distance matrix."""
    key = id(distance_matrix)
    if key not in _dm_cache:
        _dm_cache[key] = np.asarray(distance_matrix, dtype=np.float64)
        global _start_time
        if _start_time is None:
            _start_time = time.time()
    return _dm_cache[key]

def _check_deadline():
    """Return True if we are close to the time limit."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return (time.time() - _start_time) > 140

def _simulate_route_completion(current_node, depot, remaining_unvisited, rest_capacity, demands, dm, max_steps=10):
    """
    Simulate completing the route from current_node using a greedy nearest-feasible strategy.
    Returns the estimated total additional distance (excluding already-traveled path).
    """
    if len(remaining_unvisited) == 0:
        return dm[current_node, depot]
    
    # Quick deadline check
    if _check_deadline():
        # Fallback: just return distance to depot
        return dm[current_node, depot]
    
    total_dist = 0.0
    current = current_node
    remaining = set(remaining_unvisited)
    cap = rest_capacity
    steps = 0
    
    while remaining and steps < max_steps:
        # Find feasible nodes
        feasible = [n for n in remaining if demands[n] <= cap]
        if not feasible:
            break
        
        # Pick nearest feasible node
        best_node = min(feasible, key=lambda n: dm[current, n])
        total_dist += dm[current, best_node]
        cap -= demands[best_node]
        remaining.discard(best_node)
        current = best_node
        steps += 1
    
    # Add return to depot
    total_dist += dm[current, depot]
    return total_dist

def _select_by_lookahead(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    For each feasible candidate, simulate route completion and pick the one
    that minimizes the total estimated route length.
    """
    unvisited = np.asarray(unvisited_nodes)
    dm = _get_dm(distance_matrix)
    
    # Filter by capacity feasibility
    feasible_mask = demands[unvisited] <= rest_capacity
    if not np.any(feasible_mask):
        return int(depot)
    
    feasible_nodes = unvisited[feasible_mask]
    n_feasible = len(feasible_nodes)
    
    if n_feasible == 1:
        return int(feasible_nodes[0])
    
    # For each candidate, estimate the total route length if we go there first
    best_score = float('inf')
    best_node = feasible_nodes[0]
    
    for candidate in feasible_nodes:
        # Cost to reach candidate
        cost_to_candidate = dm[current_node, candidate]
        
        # Remaining capacity after visiting candidate
        new_cap = rest_capacity - demands[candidate]
        
        # Remaining unvisited nodes (excluding candidate)
        remaining_mask = ~np.isin(unvisited, [candidate])
        remaining_unvisited = unvisited[remaining_mask]
        
        # Simulate route completion from candidate
        completion_cost = _simulate_route_completion(
            candidate, depot, remaining_unvisited, new_cap, demands, dm, max_steps=8
        )
        
        total_estimated = cost_to_candidate + completion_cost
        
        if total_estimated < best_score:
            best_score = total_estimated
            best_node = candidate
    
    return int(best_node)

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Select the next node to visit or return to depot (0).
    
    Parameters:
    - current_node: current node index
    - depot: depot index (0)
    - unvisited_nodes: array/list of unvisited customer indices
    - rest_capacity: remaining vehicle capacity
    - demands: array of demand for each node
    - distance_matrix: 2D array of distances
    
    Returns:
    - int: next node to visit, or 0 (depot) to close the route
    """
    if len(unvisited_nodes) == 0:
        return int(depot)
    
    unvisited = np.asarray(unvisited_nodes)
    
    # Check deadline
    if _check_deadline():
        # Fallback to simple nearest feasible
        feasible_mask = demands[unvisited] <= rest_capacity
        if not np.any(feasible_mask):
            return int(depot)
        feasible_nodes = unvisited[feasible_mask]
        if len(feasible_nodes) == 1:
            return int(feasible_nodes[0])
        dm = _get_dm(distance_matrix)
        dists = dm[current_node, feasible_nodes]
        return int(feasible_nodes[np.argmin(dists)])
    
    # Use lookahead-based selection
    return _select_by_lookahead(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix)

# EVOLVE-BLOCK-END
