"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level deadline guard
_START_TIME = time.time()
DEADLINE_MARGIN = 15.0  # seconds of safety margin before 150s limit
MAX_TIME_PER_INSTANCE = 10.0 # Max seconds to spend computing a tour for one instance

# Cache for pre-computed tours keyed by instance fingerprint
_tour_cache = {}

def _elapsed():
    return time.time() - _START_TIME

def _deadline_exceeded():
    return _elapsed() > 150.0 - DEADLINE_MARGIN

def _get_instance_fingerprint(distance_matrix):
    """
    Create a unique fingerprint for the distance matrix to cache tours.
    """
    flat_matrix = np.round(distance_matrix, decimals=4).flatten()
    return hash(flat_matrix.tobytes())

def _tour_length(tour, distance_matrix):
    """Calculate the total length of the closed tour."""
    n = len(tour)
    if n == 0:
        return 0
    next_tour = np.roll(tour, -1)
    return np.sum(distance_matrix[tour, next_tour])

def _sa_tour(distance_matrix, max_time=MAX_TIME_PER_INSTANCE):
    """
    Construct a high-quality tour using Simulated Annealing with 2-Opt moves.
    """
    n = distance_matrix.shape[0]
    start_time = time.time()
    
    # Initial tour: Random permutation
    tour = np.random.permutation(n)
    current_length = _tour_length(tour, distance_matrix)
    best_tour = tour.copy()
    best_length = current_length
    
    # Annealing parameters
    # For n=50, typical optimal length is around 10-20 (unit square).
    # We scale temperature to the initial tour length to be robust.
    T0 = current_length * 0.1 
    T_min = 1e-5
    # Number of iterations to reach T_min
    # We don't know how many iterations we can afford, so we estimate.
    # Let's run for a fixed number of iterations or until time runs out.
    # We'll use a geometric cooling schedule.
    # alpha = (T_min / T0)^(1/iterations)
    
    # Estimate max iterations possible in 1 second to tune alpha
    # But we have 10s budget. Let's just use a fixed number of iterations 
    # and check time frequently.
    # 50 cities -> 2-opt moves are O(1) cost update if we precompute, but here we recompute or use delta.
    # With numpy, delta calculation for 2-opt is fast.
    
    # Let's aim for ~20000 iterations which should be fast enough.
    max_iter = 20000
    alpha = (T_min / T0) ** (1.0 / max_iter)
    
    T = T0
    rng = np.random.RandomState(42) # Fixed seed for reproducibility within instance? 
    # Actually, random starts are better. Let's use global random.
    
    for i in range(max_iter):
        if time.time() - start_time > max_time:
            break
        
        # Generate a random 2-opt move
        # Pick two distinct indices i, j from 1 to n-1
        # (Index 0 is fixed to break symmetry, but since we permute, we can pick any)
        # Standard 2-opt: reverse segment between i and j.
        # Let's pick i < j.
        i = rng.randint(1, n)
        j = rng.randint(1, n)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        
        # Calculate delta
        # Edges involved: (i-1, i), (j, j+1) removed
        # New edges: (i-1, j), (i, j+1) added
        # Note: indices in tour array.
        a = tour[i-1]
        b = tour[i]
        c = tour[j]
        d = tour[(j+1) % n]
        
        delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
        
        if delta < 0:
            # Always accept
            tour[i:j+1] = tour[i:j+1][::-1]
            current_length += delta
            if current_length < best_length:
                best_length = current_length
                best_tour = tour.copy()
        else:
            # Accept with probability exp(-delta / T)
            if T > T_min:
                prob = np.exp(-delta / T)
                if rng.random_sample() < prob:
                    tour[i:j+1] = tour[i:j+1][::-1]
                    current_length += delta
        
        T *= alpha
        
    return best_tour

def _get_tour(distance_matrix):
    """
    Retrieve or compute the best tour for the given instance.
    """
    fingerprint = _get_instance_fingerprint(distance_matrix)
    
    if fingerprint in _tour_cache:
        return _tour_cache[fingerprint]
    
    # Check deadline
    if _deadline_exceeded():
        # Fallback: simple NN tour
        n = distance_matrix.shape[0]
        unvisited = set(range(n))
        tour = [0]
        unvisited.remove(0)
        current = 0
        while unvisited:
            dists = distance_matrix[current, list(unvisited)]
            min_idx = np.argmin(dists)
            next_node = list(unvisited)[min_idx]
            tour.append(next_node)
            unvisited.remove(next_node)
            current = next_node
        return tour
    
    # Compute tour using Simulated Annealing
    tour = _sa_tour(distance_matrix)
    
    # Cache it
    _tour_cache[fingerprint] = tour
    
    return tour

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick the next node in a TSP tour construction.
    Uses a pre-computed high-quality tour (Simulated Annealing) and follows it.
    """
    # Deadline guard: if we're running out of time, fall back to nearest neighbor
    if _deadline_exceeded():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Get or compute the pre-computed tour
    best_tour = _get_tour(distance_matrix)
    
    # Find current node in the tour
    try:
        current_idx = best_tour.index(current_node)
    except ValueError:
        # If current node is not in the tour (shouldn't happen), fallback to NN
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # The next node in the tour
    next_node_in_tour = best_tour[(current_idx + 1) % len(best_tour)]
    
    # Check if this node is in the unvisited set
    if next_node_in_tour in unvisited_nodes:
        return int(next_node_in_tour)
    
    # Fallback: Find the closest unvisited node to current_node
    return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
# EVOLVE-BLOCK-END
