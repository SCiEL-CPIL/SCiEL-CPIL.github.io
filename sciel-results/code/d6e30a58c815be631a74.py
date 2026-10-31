"""Strategy: Simulated Annealing with Or-Opt moves to find a high-quality tour."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

# Global state
_START_TIME = time.time()
_DEADLINE = 140.0  # Safety margin
_cached_tour = None
_fingerprint = None

def _get_fingerprint(distance_matrix):
    """Create a simple fingerprint to identify the instance."""
    # Use a few rows to create a hashable key
    # This is fast and sufficient for identifying the specific 50-city instance
    return (tuple(distance_matrix[0, :5]), 
            tuple(distance_matrix[10, :5]), 
            tuple(distance_matrix[20, :5]))

def _tour_length(tour, distance_matrix):
    """Calculate the total length of the tour."""
    n = len(tour)
    if n == 0:
        return 0
    # Vectorized calculation for speed
    # tour[i] -> tour[i+1]
    next_tour = np.roll(tour, -1)
    return np.sum(distance_matrix[tour, next_tour])

def _or_opt_move(tour, distance_matrix, i, j, seg_len):
    """
    Perform an Or-Opt move: remove segment of length seg_len starting at i,
    and insert it before node at index j in the remaining tour.
    Returns (new_tour, delta_cost).
    """
    n = len(tour)
    if i == j or (j - i) % n == seg_len or (i - j) % n == seg_len:
        return None, 0.0
    
    # Extract segment
    # Note: tour is a numpy array. We work with lists for easier manipulation in this context
    # or we can do index math. For N=50, list manipulation is fast enough.
    t_list = list(tour)
    seg = t_list[i:i+seg_len]
    
    # Remove segment
    del t_list[i:i+seg_len]
    
    # Adjust insertion index j if j > i
    if j > i:
        j -= seg_len
    
    # Insert segment before index j
    t_list[j:j] = seg
    
    new_tour = np.array(t_list)
    new_cost = _tour_length(new_tour, distance_matrix)
    old_cost = _tour_length(tour, distance_matrix)
    
    return new_tour, new_cost - old_cost

def _sa_optimize(tour, distance_matrix, time_limit):
    """
    Optimize the tour using Simulated Annealing with Or-Opt moves.
    """
    current_tour = tour.copy()
    current_cost = _tour_length(current_tour, distance_matrix)
    best_tour = current_tour.copy()
    best_cost = current_cost
    
    n = len(current_tour)
    if n < 4:
        return current_tour
    
    # SA Parameters
    start_temp = 100.0
    end_temp = 0.01
    cooling_rate = 0.995
    
    temp = start_temp
    start_time = time.time()
    
    while temp > end_temp:
        if time.time() - start_time > time_limit:
            break
            
        # Pick a random move
        seg_len = random.choice([1, 2, 3])
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        
        # Avoid invalid moves (adjacent or same position)
        if i == j or (j - i) % n == seg_len or (i - j) % n == seg_len:
            continue
            
        new_tour, delta = _or_opt_move(current_tour, distance_matrix, i, j, seg_len)
        
        if new_tour is None:
            continue
            
        # Acceptance criterion
        if delta < 0 or random.random() < np.exp(-delta / temp):
            current_tour = new_tour
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_tour = current_tour.copy()
                
        # Cool down
        temp *= cooling_rate
        
    return best_tour

def _compute_best_tour(distance_matrix):
    """
    Pre-compute a high-quality tour using SA.
    """
    global _cached_tour
    start = time.time()
    
    n = distance_matrix.shape[0]
    
    # 1. Generate initial tour using Nearest Neighbor
    visited = np.zeros(n, dtype=bool)
    tour = []
    current = 0
    visited[0] = True
    tour.append(0)
    
    for _ in range(n - 1):
        dists = distance_matrix[current].copy()
        dists[visited] = np.inf
        next_node = np.argmin(dists)
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
        
    tour = np.array(tour)
    
    # 2. Apply Simulated Annealing
    # Allocate most of the time budget to SA
    time_remaining = 130.0 - (time.time() - start)
    if time_remaining > 1:
        tour = _sa_optimize(tour, distance_matrix, time_remaining)
        
    _cached_tour = tour

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node by following a pre-computed tour.
    """
    global _cached_tour, _fingerprint
    
    # Deadline guard
    if time.time() > _DEADLINE:
        # Fallback: Nearest neighbor
        if len(unvisited_nodes) == 0:
            return int(current_node)
        dists = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists)
        return int(unvisited_nodes[best_idx])

    # Check if we have a cached tour for this instance
    fp = _get_fingerprint(distance_matrix)
    if _fingerprint != fp or _cached_tour is None:
        _fingerprint = fp
        _cached_tour = None
        _compute_best_tour(distance_matrix)
        
    # If we still don't have a tour (e.g. timeout during computation), fallback
    if _cached_tour is None:
        if len(unvisited_nodes) == 0:
            return int(current_node)
        dists = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists)
        return int(unvisited_nodes[best_idx])

    # Find the next node in the cached tour
    # We need to find the position of current_node in the tour
    # Then pick the next node in the tour that is in unvisited_nodes
    
    # Optimization: Pre-compute the order of nodes in the tour
    # tour_order[node] = index in tour
    # But we can just search. N=50 is small.
    
    # Find current index
    try:
        curr_idx = np.where(_cached_tour == current_node)[0][0]
    except IndexError:
        # Fallback if current node not in tour (should not happen)
        if len(unvisited_nodes) == 0:
            return int(current_node)
        dists = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists)
        return int(unvisited_nodes[best_idx])

    # Iterate through the tour starting from curr_idx + 1
    n = len(_cached_tour)
    for k in range(1, n):
        next_node_in_tour = _cached_tour[(curr_idx + k) % n]
        if next_node_in_tour in unvisited_nodes:
            return int(next_node_in_tour)
            
    # If no unvisited node found in the tour (should not happen if tour is valid), fallback
    if len(unvisited_nodes) == 0:
        return int(current_node)
    dists = distance_matrix[current_node, unvisited_nodes]
    best_idx = np.argmin(dists)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
