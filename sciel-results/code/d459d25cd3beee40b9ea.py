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

def _fast_2opt(tour, distance_matrix, max_time=0.5):
    """
    Fast 2-opt with time limit. tour is a list of node indices.
    Treats tour as a cycle.
    """
    n = len(tour)
    if n < 4:
        return tour
    
    t = tour[:]
    start = time.time()
    
    # Precompute as numpy array for faster indexing
    t_arr = np.array(t)
    
    improved = True
    while improved:
        if time.time() - start > max_time:
            break
        improved = False
        
        for i in range(n - 1):
            if time.time() - start > max_time:
                break
            
            # Vectorize the inner loop
            # For each j from i+2 to n-1
            # old_cost[i,j] = d[t[i], t[i+1]] + d[t[j], t[(j+1)%n]]
            # new_cost[i,j] = d[t[i], t[j]] + d[t[i+1], t[(j+1)%n]]
            
            i1 = i + 1
            i2 = i + 2
            
            if i2 >= n:
                continue
                
            # j ranges from i+2 to n-1
            js = np.arange(i2, n)
            # For j = n-1, next is 0; otherwise j+1
            next_js = np.where(js == n - 1, 0, js + 1)
            
            # Vectorized cost computation
            old_costs = distance_matrix[t[i], t[i+1]] + distance_matrix[t[js], t[next_js]]
            new_costs = distance_matrix[t[i], t[js]] + distance_matrix[t[i+1], t[next_js]]
            
            # Find best improvement
            diffs = new_costs - old_costs
            min_idx = np.argmin(diffs)
            
            if diffs[min_idx] < -1e-8:
                j = js[min_idx]
                # Reverse segment t[i+1] to t[j]
                t[i+1:j+1] = t[i+1:j+1][::-1]
                t_arr = np.array(t)
                improved = True
                break  # Restart outer loop after improvement
    
    return t

def _estimate_remaining_cost(current_node, unvisited_nodes, distance_matrix):
    """
    Estimate the cost to complete the tour from current_node through all unvisited_nodes and back.
    Uses a fast nearest neighbor heuristic.
    """
    if len(unvisited_nodes) == 0:
        return 0.0
    
    remaining = list(unvisited_nodes)
    curr = current_node
    total = 0.0
    
    while remaining:
        rem_arr = np.array(remaining)
        dists = distance_matrix[curr, rem_arr]
        min_idx = np.argmin(dists)
        next_node = remaining[min_idx]
        total += dists[min_idx]
        curr = next_node
        remaining.pop(min_idx)
    
    # Add cost to return to destination (which is the first node in _current_tour)
    # Actually, the tour must close back to the start
    start_node = _current_tour[0] if _current_tour else current_node
    total += distance_matrix[curr, start_node]
    
    return total

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
    
    # If only one node left, must pick it.
    if len(unvisited_nodes) == 1:
        next_node = int(unvisited_nodes[0])
        _current_tour.append(next_node)
        return next_node
    
    # If two nodes left, pick the one that gives shorter closing
    if len(unvisited_nodes) == 2:
        a, b = unvisited_nodes[0], unvisited_nodes[1]
        # Option 1: a then b, then back to start
        cost1 = distance_matrix[current_node, a] + distance_matrix[a, b] + distance_matrix[b, _current_tour[0]]
        # Option 2: b then a, then back to start
        cost2 = distance_matrix[current_node, b] + distance_matrix[b, a] + distance_matrix[a, _current_tour[0]]
        if cost1 <= cost2:
            next_node = int(a)
        else:
            next_node = int(b)
        _current_tour.append(next_node)
        return next_node
    
    # Strategy: Regret-based selection with fast estimation
    # For each candidate, compute:
    # 1. Immediate cost: d(current, candidate)
    # 2. Estimated remaining cost after picking candidate
    # Regret = estimated cost if we DON'T pick the best candidate now
    
    # Precompute immediate costs
    imm_costs = distance_matrix[current_node, unvisited_nodes]
    
    # For each candidate, estimate the total tour cost
    # To speed up, use a vectorized nearest neighbor completion
    n_cands = len(unvisited_nodes)
    
    # Estimate remaining cost for each candidate
    # This is expensive, so we'll do it efficiently
    est_costs = np.zeros(n_cands)
    
    for idx in range(n_cands):
        candidate = unvisited_nodes[idx]
        # Remaining nodes after picking candidate
        mask = np.ones(n_cands, dtype=bool)
        mask[idx] = False
        remaining = unvisited_nodes[mask]
        
        if len(remaining) == 0:
            est_costs[idx] = distance_matrix[candidate, _current_tour[0]]
        else:
            # Fast NN completion
            rem_list = list(remaining)
            curr = candidate
            rem_cost = 0.0
            while rem_list:
                rem_arr = np.array(rem_list)
                dists = distance_matrix[curr, rem_arr]
                min_idx = np.argmin(dists)
                rem_cost += dists[min_idx]
                curr = rem_list[min_idx]
                rem_list.pop(min_idx)
            rem_cost += distance_matrix[curr, _current_tour[0]]
            est_costs[idx] = rem_cost
    
    # Total estimated cost for each candidate
    total_costs = imm_costs + est_costs
    
    # Also apply a fast 2-opt on the current tour to improve it
    # But we can't modify the tour mid-selection... 
    # Instead, use the estimated costs to pick the best candidate
    
    # Pick the candidate with minimum total estimated cost
    best_idx = np.argmin(total_costs)
    best_candidate = int(unvisited_nodes[best_idx])
    
    # Update global state
    _current_tour.append(best_candidate)
    
    # If we have time, apply a quick 2-opt on the current partial tour
    # This doesn't affect the return value but improves the tour for future decisions
    if time.time() - _start_time < _TIME_LIMIT - 10 and len(_current_tour) >= 4:
        # Apply 2-opt to the current tour (as a path, not cycle yet)
        # But 2-opt on an open path is different... 
        # Skip for now to keep it simple and fast
        pass
    
    return best_candidate

# EVOLVE-BLOCK-END
