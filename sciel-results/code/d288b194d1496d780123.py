"""
Multi-Start NN + 2-Opt + Or-Opt guided TSP constructive heuristic.

Strategy:
  1. Detect new instance.
  2. Pre-compute a high-quality reference tour using:
     a. Nearest Neighbor initialization.
     b. 2-Opt local search.
     c. Or-Opt local search.
  3. During construction, always try to visit the successor of the current node 
     in the reference tour.
  4. If the successor is already visited (due to divergence), fall back to 
     the nearest unvisited node.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_LIMIT = 115.0  # Safety margin before 120s limit

# Per-instance state
_REF_TOUR = None
_INSTANCE_COUNTER = 0


def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT


def _nn_tour(distance_matrix, start_node):
    """Generate a tour using Nearest Neighbor heuristic."""
    n = distance_matrix.shape[0]
    visited = [start_node]
    unvisited = set(range(n))
    unvisited.remove(start_node)
    current = start_node
    
    while unvisited:
        # Find nearest unvisited node
        dists = distance_matrix[current, list(unvisited)]
        nearest_idx = np.argmin(dists)
        next_node = list(unvisited)[nearest_idx]
        visited.append(next_node)
        unvisited.remove(next_node)
        current = next_node
        
    return visited


def _two_opt(tour, distance_matrix):
    """Improve a tour using 2-Opt local search."""
    n = len(tour)
    improved = True
    best_tour = tour[:]
    best_len = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
    
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Skip if i and j are adjacent in the cycle
                if i == 0 or j == n - 1:
                    continue
                # Calculate improvement
                # Current edges: (i-1, i) and (j, j+1)
                # New edges: (i-1, j) and (i, j+1)
                d_old = distance_matrix[tour[i-1], tour[i]] + distance_matrix[tour[j], tour[(j+1)%n]]
                d_new = distance_matrix[tour[i-1], tour[j]] + distance_matrix[tour[i], tour[(j+1)%n]]
                
                if d_new < d_old - 1e-6:
                    # Reverse segment i..j
                    tour[i:j+1] = tour[i:j+1][::-1]
                    best_len = d_new + (best_len - d_old)
                    improved = True
                    break
            if improved:
                break
                
    return tour, best_len


def _or_opt(tour, distance_matrix):
    """Improve a tour using Or-Opt local search (move 1, 2, or 3 nodes)."""
    n = len(tour)
    improved = True
    best_len = sum(distance_matrix[tour[i], tour[(i+1)%n]] for i in range(n))
    
    while improved:
        improved = False
        for length in [1, 2, 3]:
            if n <= length + 2:
                continue
            for i in range(1, n - length):
                # Segment to move: tour[i : i+length]
                # Insert after position j
                for j in range(i + length + 1, n):
                    # Avoid inserting adjacent to original position
                    if j == i + length or (j == 0 and i == 1):
                        continue
                        
                    # Calculate improvement
                    # Remove segment from i
                    # Insert after j
                    
                    # Indices in the cycle
                    prev_i = (i - 1) % n
                    next_i = (i + length) % n
                    prev_j = (j - 1) % n
                    next_j = (j) % n # j is the node after which we insert? No, j is the index in the array.
                    
                    # Let's be precise.
                    # We move nodes at indices i, i+1, ..., i+length-1
                    # to be after node at index j.
                    
                    # Current connections:
                    # (prev_i, i), (i+length-1, next_i)
                    # (prev_j, j), (j, next_j)
                    
                    # New connections:
                    # (prev_i, next_i)
                    # (prev_j, i), (i+length-1, next_j)
                    
                    d_old = distance_matrix[tour[prev_i], tour[i]] + \
                            distance_matrix[tour[i+length-1], tour[next_i]] + \
                            distance_matrix[tour[prev_j], tour[j]] + \
                            distance_matrix[tour[j], tour[next_j]]
                            
                    d_new = distance_matrix[tour[prev_i], tour[next_i]] + \
                            distance_matrix[tour[prev_j], tour[i]] + \
                            distance_matrix[tour[i+length-1], tour[next_j]]
                            
                    # Note: The edge (prev_j, j) and (j, next_j) are broken and replaced.
                    # The edge (prev_i, i) and (i+length-1, next_i) are broken and replaced.
                    # We need to be careful not to double count or miss edges.
                    
                    # Actually, simpler check:
                    # Old cost part: d(prev_i, i) + d(i+len-1, next_i) + d(prev_j, j) + d(j, next_j)
                    # New cost part: d(prev_i, next_i) + d(prev_j, i) + d(i+len-1, next_j)
                    # The rest of the tour remains unchanged.
                    
                    if d_new < d_old - 1e-6:
                        # Perform move
                        segment = tour[i:i+length]
                        del tour[i:i+length]
                        # Adjust j index if deletion happened before j
                        if i < j:
                            j -= length
                        # Insert after j
                        tour[j+1:j+1] = segment
                        best_len = d_new + (best_len - d_old)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
                
    return tour, best_len


def _build_reference_tour(distance_matrix, start_node):
    """Build a high-quality reference tour."""
    if _is_deadline_passed():
        return _nn_tour(distance_matrix, start_node)
        
    # 1. NN Tour
    tour = _nn_tour(distance_matrix, start_node)
    
    # 2. 2-Opt
    tour, _ = _two_opt(tour, distance_matrix)
    
    # 3. Or-Opt
    tour, _ = _or_opt(tour, distance_matrix)
    
    return tour


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    Uses a pre-computed reference tour to guide the selection.
    """
    global _REF_TOUR, _INSTANCE_COUNTER
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    n_total = distance_matrix.shape[0]
    
    # Detect new instance: if we are at the start (only current_node visited)
    # unvisited_nodes will have size n_total - 1
    is_new_instance = (n_unvisited == n_total - 1)
    
    if is_new_instance:
        _INSTANCE_COUNTER += 1
        # Time budget for reference tour construction
        # 2-Opt and Or-Opt on 50 nodes are very fast (< 1ms usually)
        _REF_TOUR = _build_reference_tour(distance_matrix, current_node)
    
    if _REF_TOUR is None:
        # Fallback: Nearest Neighbor
        unvisited_arr = np.asarray(unvisited_nodes)
        dists = distance_matrix[current_node, unvisited_arr]
        best_idx = int(np.argmin(dists))
        return int(unvisited_arr[best_idx])
    
    # Try to follow the reference tour
    # Find current_node in _REF_TOUR
    try:
        current_idx = _REF_TOUR.index(current_node)
    except ValueError:
        # Should not happen if logic is correct, but fallback
        unvisited_arr = np.asarray(unvisited_nodes)
        dists = distance_matrix[current_node, unvisited_arr]
        best_idx = int(np.argmin(dists))
        return int(unvisited_arr[best_idx])
        
    next_in_ref = _REF_TOUR[(current_idx + 1) % n_total]
    
    # Check if next_in_ref is in unvisited_nodes
    if next_in_ref in unvisited_nodes:
        return next_in_ref
        
    # If not, fall back to nearest unvisited node
    # This can happen if the constructive path has diverged from the reference tour
    # or if we are in a state where the reference successor was already visited
    unvisited_arr = np.asarray(unvisited_nodes)
    dists = distance_matrix[current_node, unvisited_arr]
    best_idx = int(np.argmin(dists))
    return int(unvisited_arr[best_idx])

# EVOLVE-BLOCK-END
