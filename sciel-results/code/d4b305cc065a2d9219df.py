# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 145.0  # Safety margin before 150s kill

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _start_time
    
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        # Fallback to nearest neighbor
        if len(unvisited_nodes) == 0:
            return -1
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    if n_unvisited == 2:
        # Pick the one that gives shorter total: current->A->dest vs current->B->dest
        a, b = unvisited_nodes
        cost_a = distance_matrix[current_node, a] + distance_matrix[a, destination_node]
        cost_b = distance_matrix[current_node, b] + distance_matrix[b, destination_node]
        return int(a if cost_a <= cost_b else b)
    
    # For each candidate, estimate the remaining tour cost using a greedy nearest-neighbor completion
    # This is O(n^3) per step but n is small (<=50), so it's fast enough.
    # We vectorize the candidate selection loop as much as possible, but the inner greedy loop
    # is sequential. We can parallelize the candidate evaluations if we use a list comprehension
    # or just a simple loop since n is small.
    
    best_cost = np.inf
    best_node = unvisited_nodes[0]
    
    # Pre-extract submatrix for unvisited nodes to speed up lookups?
    # distance_matrix is already a numpy array, indexing is O(1).
    
    for cand in unvisited_nodes:
        # Simulate greedy completion from cand
        curr = cand
        remaining = np.delete(unvisited_nodes, np.where(unvisited_nodes == cand)[0])
        
        total_cost = distance_matrix[current_node, cand]
        
        while len(remaining) > 0:
            # Find nearest neighbor in remaining
            dists = distance_matrix[curr, remaining]
            min_idx = np.argmin(dists)
            next_node = remaining[min_idx]
            total_cost += dists[min_idx]
            
            # Remove next_node from remaining
            remaining = np.delete(remaining, min_idx)
            curr = next_node
            
            # Early termination if cost already exceeds best
            if total_cost >= best_cost:
                break
        
        # Add cost to close the tour back to destination
        if len(remaining) == 0:
            total_cost += distance_matrix[curr, destination_node]
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_node = cand
            
    return int(best_node)
# EVOLVE-BLOCK-END
