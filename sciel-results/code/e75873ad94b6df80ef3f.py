"""
TSP Constructive Heuristic: Reference Tour Guidance with 2-Opt.

Strategy:
1. Detect new instance.
2. Generate a high-quality reference tour using Multi-Start Nearest Neighbor + 2-Opt.
3. During construction, select the next node by minimizing a weighted sum of:
   - Distance from current node to candidate.
   - Minimum distance from candidate to any other unvisited node (2-step lookahead).
   - Distance from candidate to its successor in the reference tour (structural guide).
"""
import numpy as np
import time

_start_time = time.time()
_LIMIT = 115.0  # Safety margin

# Global state for reference tour
_ref_next_map = {}
_instance_id = -1
_last_n = 0

def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT

def _nn_tour(dist, start):
    """Generate a tour using Nearest Neighbor heuristic."""
    n = dist.shape[0]
    unvisited = set(range(n))
    unvisited.remove(start)
    tour = [start]
    current = start
    while unvisited:
        # Find nearest unvisited node
        nearest = min(unvisited, key=lambda x: dist[current, x])
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    return tour

def _two_opt(dist, tour):
    """Improve tour using 2-Opt local search."""
    n = len(tour)
    improved = True
    # Limit iterations to prevent excessive time on difficult instances, though N=50 is small
    max_iters = 100
    iter_count = 0
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                # Check if reversing segment (i+1, j) improves the tour
                delta = dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d]
                if delta < 1e-10:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
    return tour

def _build_reference_tour(dist):
    """Build a reference tour using multi-start NN + 2-Opt."""
    n = dist.shape[0]
    best_tour = None
    best_len = float('inf')
    
    # Try multiple starting nodes to find a good initial tour
    starts = [0, 1, 2, 3, 4, 5]
    for s in starts:
        if _is_deadline_passed():
            break
        t = _nn_tour(dist, s)
        t = _two_opt(dist, t)
        # Calculate tour length
        length = sum(dist[t[i], t[(i + 1) % n]] for i in range(n))
        if length < best_len:
            best_len = length
            best_tour = t
            
    return best_tour

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _ref_next_map, _instance_id, _last_n
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    n_total = distance_matrix.shape[0]
    
    # Detect new instance
    # A new instance starts when the number of unvisited nodes is n_total - 1 (or n_total depending on convention)
    # Usually, the first call has n-1 unvisited nodes (start node is 'current').
    is_new_instance = (n_unvisited >= n_total - 1)
    
    if is_new_instance:
        _instance_id += 1
        _last_n = n_total
        _ref_next_map = {}
        
        # Build reference tour if time permits
        if not _is_deadline_passed():
            ref_tour = _build_reference_tour(distance_matrix)
            if ref_tour is not None:
                # Precompute next node in reference tour
                for i in range(len(ref_tour)):
                    u = ref_tour[i]
                    v = ref_tour[(i + 1) % len(ref_tour)]
                    _ref_next_map[u] = v
        else:
            _ref_next_map = None # Signal to use fallback
    
    # Prepare unvisited nodes as array for vectorization
    unvisited_arr = np.array(unvisited_nodes)
    
    # Fallback: Pure Nearest Neighbor if reference is not available
    if _ref_next_map is None:
        dists_cur = distance_matrix[current_node, unvisited_arr]
        best_idx = np.argmin(dists_cur)
        return int(unvisited_arr[best_idx])
    
    # Compute scores
    # 1. Distance from current to candidate
    dists_cur = distance_matrix[current_node, unvisited_arr]
    
    # 2. 2-step lookahead: min distance from candidate to any other unvisited node
    if n_unvisited > 2:
        sub_dist = distance_matrix[np.ix_(unvisited_arr, unvisited_arr)]
        np.fill_diagonal(sub_dist, np.inf)
        min_next = np.min(sub_dist, axis=1)
    else:
        min_next = np.zeros(n_unvisited)
        
    # 3. Reference tour guidance: distance from candidate to its successor in ref tour
    ref_costs = np.array([distance_matrix[c, _ref_next_map[c]] for c in unvisited_arr])
    
    # Combine scores
    # Weights:
    # alpha: weight for 2-step lookahead
    # beta: weight for reference tour alignment
    alpha = 0.5
    beta = 0.5
    
    scores = dists_cur + alpha * min_next + beta * ref_costs
    
    best_idx = np.argmin(scores)
    return int(unvisited_arr[best_idx])
