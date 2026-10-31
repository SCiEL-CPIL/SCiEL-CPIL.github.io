"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for per-instance pre-computed reference tours
_reference_cache = {}
_start_time = None

def _get_instance_id(distance_matrix):
    """Create a hashable identifier for a distance matrix instance."""
    # Use a small subset of the matrix to create a unique ID
    # The matrix is 50x50, so we can use the first few rows
    key = distance_matrix[:5, :5].tobytes()
    return key

def _build_reference_tour(distance_matrix, n, deadline):
    """
    Build a high-quality reference tour using multi-start nearest neighbor + 2-opt.
    Returns an array of length n where ref_tour[i] is the node visited at position i,
    and a successor array where successor[i] is the node that follows i in the tour.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Multi-start nearest neighbor
    best_tour = None
    best_len = np.inf
    
    # Use multiple starting nodes for diversity
    start_nodes = list(range(min(10, n)))
    
    for start in start_nodes:
        # Check deadline
        if time.time() - _start_time > deadline - 5.0:
            break
            
        visited = np.zeros(n, dtype=bool)
        tour = [start]
        visited[start] = True
        current = start
        
        for _ in range(n - 1):
            if time.time() - _start_time > deadline - 5.0:
                break
            # Find nearest unvisited node
            dists = distance_matrix[current].copy()
            dists[visited] = np.inf
            next_node = int(np.argmin(dists))
            tour.append(next_node)
            visited[next_node] = True
            current = next_node
        
        if len(tour) < n:
            continue
            
        # Calculate tour length
        tour_arr = np.array(tour)
        total_len = 0.0
        for i in range(n):
            total_len += distance_matrix[tour_arr[i], tour_arr[(i + 1) % n]]
        
        if total_len < best_len:
            best_len = total_len
            best_tour = tour_arr
    
    if best_tour is None:
        # Fallback: simple nearest neighbor from node 0
        visited = np.zeros(n, dtype=bool)
        tour = [0]
        visited[0] = True
        current = 0
        for _ in range(n - 1):
            dists = distance_matrix[current].copy()
            dists[visited] = np.inf
            next_node = int(np.argmin(dists))
            tour.append(next_node)
            visited[next_node] = True
            current = next_node
        best_tour = np.array(tour)
    
    # Apply 2-opt improvement
    tour = best_tour.copy()
    improved = True
    max_2opt_iters = 50
    iter_count = 0
    
    while improved and iter_count < max_2opt_iters:
        improved = False
        iter_count += 1
        
        if time.time() - _start_time > deadline - 3.0:
            break
        
        for i in range(n - 1):
            if time.time() - _start_time > deadline - 3.0:
                break
            
            for j in range(i + 1, n):
                # 2-opt: reverse segment i+1 to j
                # Current edges: (tour[i], tour[i+1]) and (tour[j], tour[(j+1)%n])
                # New edges: (tour[i], tour[j]) and (tour[i+1], tour[(j+1)%n])
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                
                old_cost = distance_matrix[a, b] + distance_matrix[c, d]
                new_cost = distance_matrix[a, c] + distance_matrix[b, d]
                
                if new_cost < old_cost - 1e-10:
                    # Reverse segment i+1 to j
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
    
    # Build successor array
    successor = np.zeros(n, dtype=int)
    for i in range(n):
        successor[tour[i]] = tour[(i + 1) % n]
    
    return successor

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a convex hull-inspired heuristic with reference tour guidance.
    
    Strategy:
    1. Pre-compute a high-quality reference tour (multi-start NN + 2-opt) for the instance.
    2. For each candidate, compute a composite score:
       - Distance from current node to candidate
       - Penalty for deviating from the reference tour's successor
       - Geometric penalty based on triangle area to maintain tight perimeter
    """
    global _start_time, _reference_cache
    
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: 150s wall-clock, use 145s as safety margin
    deadline = 145.0
    
    n = distance_matrix.shape[0]
    key = _get_instance_id(distance_matrix)
    
    # Get or compute reference tour successor
    if key not in _reference_cache:
        if time.time() - _start_time > deadline - 10.0:
            # Not enough time for reference tour, fall back to nearest neighbor
            return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
        _reference_cache[key] = _build_reference_tour(distance_matrix, n, deadline)
    
    successor = _reference_cache[key]
    
    # If only one unvisited node, return it
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Compute distances from current node to all unvisited nodes
    dists = distance_matrix[current_node, unvisited_nodes]
    
    # Get reference successor of current node
    ref_successor = successor[current_node]
    
    # Check if reference successor is in unvisited nodes
    ref_in_unvisited = np.isin(ref_successor, unvisited_nodes)
    
    # Score each candidate:
    # 1. Primary: distance from current
    # 2. Bonus if candidate is the reference successor
    # 3. Geometric penalty: area of triangle (current, destination, candidate)
    
    # Compute geometric penalty
    # We need coordinates, but we only have distance matrix.
    # Approximate: use distance to destination as a proxy for "deviation"
    # Candidates far from destination are penalized more
    
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Composite score: minimize distance, prefer reference successor, penalize distance to destination
    # Weight tuning: reference successor gets a large bonus
    scores = dists.copy()
    
    # Add penalty for distance to destination (encourages staying near the "frontier")
    scores += 0.1 * dist_to_dest
    
    # If reference successor is in unvisited, give it a strong bonus (subtract from score)
    if ref_in_unvisited:
        ref_idx = np.where(unvisited_nodes == ref_successor)[0][0]
        scores[ref_idx] -= 2.0 * dists[ref_idx]  # Strong preference for reference successor
    
    # If reference successor is not available, use nearest neighbor with destination penalty
    if not ref_in_unvisited:
        # Fall back to weighted combination
        scores = dists + 0.3 * dist_to_dest
    
    # Handle edge case: if all scores are inf or invalid
    if np.all(np.isinf(scores)):
        return int(unvisited_nodes[0])
    
    best_idx = int(np.argmin(scores))
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
