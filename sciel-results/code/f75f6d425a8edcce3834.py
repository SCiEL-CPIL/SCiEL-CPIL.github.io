"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for pre-computed tour
_precomputed_tour = None
_start_time = None
_MAX_TIME = 100.0  # Safety margin before 120s/150s limit

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _precomputed_tour, _start_time
    
    # Initialize start time
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard: if we are running out of time, fallback to fast greedy
    if time.time() - _start_time > _MAX_TIME:
        if len(unvisited_nodes) == 0:
            return int(destination_node)
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    # If we haven't computed the tour yet, and we are at the beginning of the tour
    # (i.e., most nodes are still unvisited)
    if _precomputed_tour is None:
        if len(unvisited_nodes) > 40:  # Threshold for "start of tour"
            _precomputed_tour = _compute_optimized_tour(current_node, unvisited_nodes, distance_matrix)
            
    # If we have a pre-computed tour, follow it
    if _precomputed_tour is not None:
        # Find current_node in the tour
        try:
            idx = np.where(_precomputed_tour == current_node)[0][0]
            # Get the next node in the cyclic tour
            next_node = _precomputed_tour[(idx + 1) % len(_precomputed_tour)]
            
            # The next node must be in the unvisited set (unless it's the destination/start and we are closing the loop)
            # However, the interface implies we pick from unvisited_nodes.
            # If next_node is the start/destination, it is NOT in unvisited_nodes.
            # In that case, we should return destination_node? 
            # But usually, the loop stops when unvisited is empty.
            # If next_node is in unvisited_nodes, return it.
            if next_node in unvisited_nodes:
                return int(next_node)
            else:
                # If the next node in the optimized tour is not in unvisited (e.g., it's the start node),
                # and we are forced to pick from unvisited, this is a conflict.
                # But logically, if we are following a valid tour, the next node should be unvisited 
                # until the very last step.
                # If we are at the last step, next_node is the start. 
                # If the evaluator expects us to return the start to close the tour, 
                # but the signature says "picks the next city... from unvisited", 
                # it might not call us when unvisited is empty.
                # If unvisited is not empty, but next_node is not in it, something is wrong.
                # Fallback to nearest neighbor in this edge case.
                return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
        except (IndexError, ValueError):
            # Fallback if current_node not found in tour
            return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    # Fallback: Nearest Neighbor
    return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

def _compute_optimized_tour(start_node, unvisited_nodes, distance_matrix):
    """
    Computes a high-quality tour using Multi-Start 2-Opt.
    Returns the tour as a numpy array of node indices.
    """
    all_nodes = np.concatenate(([start_node], unvisited_nodes))
    n = len(all_nodes)
    
    best_tour = None
    best_len = float('inf')
    
    # Multi-start with random permutations
    restarts = 50  # 50 restarts is fast for N=50
    for i in range(restarts):
        # Random permutation
        tour = np.random.permutation(all_nodes)
        
        # 2-Opt improvement
        tour, length = _two_opt(tour, distance_matrix)
        
        if length < best_len:
            best_len = length
            best_tour = tour.copy()
            
    return best_tour

def _two_opt(tour, distance_matrix):
    """
    Performs 2-opt optimization on a tour.
    Returns the optimized tour and its length.
    """
    n = len(tour)
    improved = True
    # Limit iterations to prevent infinite loops in case of floating point issues
    max_iters = 1000
    iter_count = 0
    
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Skip if adjacent
                if i + 1 == j:
                    continue
                
                # Get nodes
                a = tour[i]
                b = tour[i + 1]
                c = tour[j]
                d = tour[(j + 1) % n]  # Wrap around for the last edge
                
                # Calculate delta
                # Current edges: (a,b) and (c,d)
                # New edges: (a,c) and (b,d)
                current_len = distance_matrix[a, b] + distance_matrix[c, d]
                new_len = distance_matrix[a, c] + distance_matrix[b, d]
                
                if new_len < current_len:
                    # Reverse the segment from i+1 to j
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    improved = True
    
    # Calculate final length
    total_len = 0.0
    for k in range(n):
        total_len += distance_matrix[tour[k], tour[(k+1)%n]]
        
    return tour, total_len
# EVOLVE-BLOCK-END
