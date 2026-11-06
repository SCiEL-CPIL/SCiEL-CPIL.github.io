"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track the tour being built
_current_tour = []
_start_time = None
_TIME_LIMIT = 145.0  # Safety margin for 150s kill

def _reset_state():
    global _current_tour, _start_time
    _current_tour = []
    _start_time = time.time()

def _apply_2opt(tour, distance_matrix, max_passes=2):
    """
    Apply 2-opt improvement to a list of nodes representing an open path.
    The cost function includes the closing edge back to the start.
    """
    n = len(tour)
    if n < 4:
        return tour
    
    # Copy tour to avoid modifying original
    t = tour[:]
    improved = True
    passes = 0
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n - 1):
            for j in range(i + 2, n):
                # Skip the case where we would reverse the entire path
                # which is equivalent to the same cycle in reverse
                if i == 0 and j == n - 1:
                    continue
                    
                # Calculate delta cost
                # Current edges: (t[i], t[i+1]) and (t[j], t[j+1] or t[0] if j==n-1)
                # New edges: (t[i], t[j]) and (t[i+1], t[j+1] or t[0] if j==n-1)
                
                if j == n - 1:
                    # Closing edge involved
                    old_cost = distance_matrix[t[i], t[i+1]] + distance_matrix[t[j], t[0]]
                    new_cost = distance_matrix[t[i], t[j]] + distance_matrix[t[i+1], t[0]]
                else:
                    old_cost = distance_matrix[t[i], t[i+1]] + distance_matrix[t[j], t[j+1]]
                    new_cost = distance_matrix[t[i], t[j]] + distance_matrix[t[i+1], t[j+1]]
                
                if new_cost < old_cost - 1e-8:
                    # Reverse segment t[i+1] to t[j]
                    t[i+1:j+1] = t[i+1:j+1][::-1]
                    improved = True
                    # Break inner loop and restart outer for correctness/simplicity
                    # This is slower but ensures we don't miss improvements in the same pass
                    break
            
            if improved:
                break
        
    return t

def _estimate_tour_length(tour, distance_matrix):
    """
    Calculate the total length of a closed tour.
    """
    n = len(tour)
    if n <= 1:
        return 0.0
    
    total = 0.0
    # Sum consecutive edges
    for i in range(n - 1):
        total += distance_matrix[tour[i], tour[i+1]]
    # Closing edge
    total += distance_matrix[tour[-1], tour[0]]
    return total

def _select_with_lookahead(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node by evaluating each candidate with a fast greedy completion
    and a limited 2-opt refinement.
    """
    best_candidate = -1
    best_score = float('inf')
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Pre-compute distances from current_node to all unvisited nodes
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # We will evaluate candidates. To save time, we can limit the number of candidates
    # if the tour is already large (simulation cost grows).
    # However, for N=50, evaluating all is usually fine if simulation is fast.
    
    # Strategy: 
    # 1. For each candidate, simulate the rest of the tour using Nearest Neighbor.
    # 2. Apply 2-opt to the simulated full tour.
    # 3. Pick the candidate with the lowest estimated final tour length.
    
    # Optimization: 
    # Instead of creating a new list for remaining nodes for each candidate, 
    # we can iterate and use a mask or just slice. 
    # Since n_unvisited is small (~50), list operations are acceptable.
    
    for idx, candidate in enumerate(unvisited_nodes):
        # Simulate the rest of the tour using Nearest Neighbor
        sim_tour = _current_tour + [candidate]
        
        # Remaining nodes are unvisited_nodes excluding candidate
        # We need to efficiently get the remaining nodes.
        # np.delete is O(N) which is fine for N=50.
        remaining_arr = np.delete(unvisited_nodes, idx)
        
        curr = candidate
        while len(remaining_arr) > 0:
            # Find nearest neighbor in remaining_arr
            dists = distance_matrix[curr, remaining_arr]
            next_idx = np.argmin(dists)
            next_node = remaining_arr[next_idx]
            sim_tour.append(int(next_node))
            curr = next_node
            # Remove next_node from remaining_arr
            remaining_arr = np.delete(remaining_arr, next_idx)
        
        # Evaluate the simulated tour with 2-opt
        # Check time
        if time.time() - _start_time > _TIME_LIMIT - 5:
            # Fallback to simple distance or skip 2-opt
            # If time is tight, just use the simulated NN length without 2-opt
            # Or even just the distance from current to candidate
            score = _estimate_tour_length(sim_tour, distance_matrix)
        else:
            # Apply 2-opt
            # Limit passes to keep it fast
            optimized_tour = _apply_2opt(sim_tour, distance_matrix, max_passes=2)
            score = _estimate_tour_length(optimized_tour, distance_matrix)
        
        if score < best_score:
            best_score = score
            best_candidate = candidate
            
    return int(best_candidate)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _current_tour, _start_time
    
    # Initialize on first call
    if _start_time is None:
        _start_time = time.time()
        _current_tour = [current_node]
    
    # Time check
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        # Fallback to simple nearest neighbor to finish quickly
        if len(unvisited_nodes) == 0:
            return -1
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    if len(unvisited_nodes) == 0:
        return -1
    
    if len(unvisited_nodes) == 1:
        next_node = int(unvisited_nodes[0])
        _current_tour.append(next_node)
        return next_node
    
    # Use lookahead strategy
    next_node = _select_with_lookahead(current_node, destination_node, unvisited_nodes, distance_matrix)
    
    # Update global state
    _current_tour.append(next_node)
    
    return next_node

# EVOLVE-BLOCK-END
