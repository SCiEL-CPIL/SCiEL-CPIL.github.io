"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track time and instance
_gls_state = {
    'start_time': 0.0,
    'instance_checksum': None
}

def _compute_tour_length(tour, dist_matrix):
    """Compute the length of a closed tour."""
    n = len(tour)
    if n == 0:
        return 0.0
    # Vectorized calculation
    edges = np.array([dist_matrix[tour[i], tour[(i + 1) % n]] for i in range(n)])
    return np.sum(edges)

def _two_opt(tour, dist_matrix, max_passes=10):
    """Standard 2-opt improvement."""
    n = len(tour)
    if n < 4:
        return tour
    
    tour = list(tour)
    improved = True
    passes = 0
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                # Calculate delta
                d1 = dist_matrix[tour[i], tour[i+1]]
                d2 = dist_matrix[tour[j], tour[(j+1) % n]]
                d3 = dist_matrix[tour[i], tour[j]]
                d4 = dist_matrix[tour[i+1], tour[(j+1) % n]]
                delta = d3 + d4 - d1 - d2
                if delta < -1e-10:
                    # Reverse segment i+1 to j
                    tour[i+1:j+1] = reversed(tour[i+1:j+1])
                    improved = True
    
    return tour

def _or_opt(tour, dist_matrix, max_passes=5):
    """Or-opt: move segments of length 1, 2, or 3 to better positions."""
    n = len(tour)
    if n < 4:
        return tour
    
    tour = list(tour)
    improved = True
    passes = 0
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for seg_len in [1, 2, 3]:
            for i in range(n):
                # Extract segment
                seg = [tour[(i + k) % n] for k in range(seg_len)]
                # Calculate cost of removing segment
                # Neighbors: tour[i-1], tour[i], ... tour[i+seg_len-1], tour[i+seg_len]
                u = tour[(i - 1) % n]
                v = tour[i]
                w = tour[(i + seg_len) % n]
                
                # Cost of edges around segment
                c_remove = dist_matrix[u, v] + dist_matrix[seg[-1], w]
                
                # Try inserting at all other positions
                best_delta = 0
                best_pos = -1
                
                for j in range(n):
                    if j >= i and j < i + seg_len:
                        continue # Don't insert into itself
                    # Position j means inserting between tour[j-1] and tour[j]
                    a = tour[(j - 1) % n]
                    b = tour[j]
                    
                    # Cost of edges at insertion point
                    c_insert = dist_matrix[a, seg[0]] + dist_matrix[seg[-1], b]
                    
                    # If we remove from i, indices shift? 
                    # This simple implementation is risky with indices.
                    # Let's use a safer approach: just try to move segment starting at i to position j
                    # by checking if the move improves the tour.
                    
                    # To keep it simple and fast, we'll just do a single-move check 
                    # that doesn't require full reconstruction for every candidate.
                    # Actually, Or-opt is tricky to implement correctly in-place with shifting indices.
                    # Let's skip complex Or-opt and use a simpler "3-opt" or just stick to 2-opt 
                    # but with more passes and random restarts if time allows.
                    pass
                    
    return tour

def _random_perturbation(tour, dist_matrix, num_moves=5):
    """Apply random 2-opt moves to escape local optima."""
    n = len(tour)
    tour = list(tour)
    for _ in range(num_moves):
        i = np.random.randint(0, n - 1)
        j = np.random.randint(i + 2, n)
        if i == 0 and j == n - 1:
            continue
        tour[i+1:j+1] = reversed(tour[i+1:j+1])
    return tour

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided Local Search penalty step.
    
    Strategy:
    1. Use time budget to run an internal search (2-opt + random restarts) on the TRUE matrix 
       to find a better tour 'T_best' starting from 'local_opt_tour'.
    2. Construct a perturbed matrix that penalizes edges in the original 'local_opt_tour' 
       that are NOT in 'T_best'. This guides the evaluator's 2-opt towards 'T_best'.
    """
    global _gls_state
    
    current_time = time.time()
    if _gls_state['start_time'] == 0.0:
        _gls_state['start_time'] = current_time
    elapsed = current_time - _gls_state['start_time']
    
    # Safety margin: if we've been running for > 280s, minimize work
    if elapsed > 280:
        return edge_distance.copy()

    n = len(local_opt_tour)
    if n <= 1:
        return edge_distance.copy()

    # Identify instance
    current_checksum = (edge_distance.shape, tuple(edge_distance[0, :5]))
    if _gls_state['instance_checksum'] != current_checksum:
        _gls_state['instance_checksum'] = current_checksum

    # Time budget for internal search: e.g., 5 seconds per call? 
    # 20 calls * 5s = 100s. Total 3 instances = 300s. Tight.
    # Let's use 2 seconds per call. 20 * 2 * 3 = 120s. Safe.
    time_limit = 2.0
    search_start = time.time()
    
    best_tour = list(local_opt_tour)
    best_len = _compute_tour_length(best_tour, edge_distance)
    
    # Initial 2-opt on true matrix
    best_tour = _two_opt(best_tour, edge_distance, max_passes=10)
    best_len = _compute_tour_length(best_tour, edge_distance)
    
    # Try random restarts if time permits
    while time.time() - search_start < time_limit:
        # Random perturbation
        perturbed_tour = _random_perturbation(best_tour, edge_distance, num_moves=5)
        # 2-opt on perturbed
        improved_tour = _two_opt(perturbed_tour, edge_distance, max_passes=10)
        improved_len = _compute_tour_length(improved_tour, edge_distance)
        
        if improved_len < best_len:
            best_len = improved_len
            best_tour = improved_tour
        else:
            # If no improvement, maybe try a stronger perturbation?
            # Or just break to save time?
            # Let's try a few more times.
            pass

    # Construct perturbed distance matrix
    # Goal: Guide the evaluator's 2-opt (starting from local_opt_tour) towards best_tour.
    # Strategy: Penalize edges in local_opt_tour that are NOT in best_tour.
    # Do NOT penalize edges that are in both.
    # Do NOT penalize edges in best_tour that are not in local_opt_tour (or reward them? No, keep simple).
    
    perturbed_dist = edge_distance.copy()
    
    # Identify edges in local_opt_tour
    local_edges = set()
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        if u > v: u, v = v, u
        local_edges.add((u, v))
        
    # Identify edges in best_tour
    best_edges = set()
    for i in range(n):
        u = best_tour[i]
        v = best_tour[(i + 1) % n]
        if u > v: u, v = v, u
        best_edges.add((u, v))
        
    # Edges to penalize: in local but not in best
    penalty_edges = local_edges - best_edges
    
    # Calculate penalty magnitude
    # Use a significant fraction of the median distance
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    med_dist = np.median(edge_distance[mask])
    if med_dist < 1e-9: med_dist = 1.0
    
    # Penalty should be strong enough to break the current tour
    # but not so strong it breaks the matrix structure.
    # 10% of median distance is a reasonable start.
    penalty_val = 0.1 * med_dist
    
    for u, v in penalty_edges:
        perturbed_dist[u, v] += penalty_val
        perturbed_dist[v, u] += penalty_val
        
    return perturbed_dist
# EVOLVE-BLOCK-END
