"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None


def _get_time_remaining():
    """Return seconds remaining before the 150s limit (with safety margin)."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    return max(0.0, 145.0 - elapsed)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Time guard: if we're running low on time, fall back to simple nearest neighbor
    if _get_time_remaining() < 0.5:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    candidates = unvisited_nodes
    n = len(candidates)
    
    if n == 0:
        return destination_node
    
    if n == 1:
        return int(candidates[0])
    
    # 2-Step Lookahead Heuristic
    # Score(c) = d(current, c) + min_{v != c} [ d(c, v) + min_{u != c, v} d(v, u) ]
    # Special handling for small n to account for the return to destination.
    
    best_score = float('inf')
    best_node = candidates[0]
    
    # Pre-compute distances from current to all candidates
    dist_from_current = distance_matrix[current_node, candidates]
    
    # Extract submatrix of distances between candidates
    # dist_sub[i, j] = distance_matrix[candidates[i], candidates[j]]
    dist_sub = distance_matrix[np.ix_(candidates, candidates)]
    
    # Distance from each candidate to destination
    dist_to_dest = distance_matrix[candidates, destination_node]
    
    for i, c in enumerate(candidates):
        cost_in = dist_from_current[i]
        
        if n == 2:
            # If 2 nodes left: c and v.
            # Path: current -> c -> v -> destination
            # Cost = d(cur, c) + d(c, v) + d(v, dest)
            j = 1 - i
            v = candidates[j]
            cost_out = dist_sub[i, j] + dist_to_dest[j]
            total_score = cost_in + cost_out
        elif n == 3:
            # If 3 nodes left: c, v, u.
            # Path: current -> c -> (v or u) -> (the other) -> destination
            # We need to minimize the cost of the remaining 2-node path.
            # Remaining nodes: candidates except i.
            # Let remaining indices be R.
            # We need min_{v in R} [ d(c, v) + min_{u in R \ {v}} (d(v, u) + d(u, dest)) ]
            # Since R has size 2, let R = {v, u}.
            # Option 1: c -> v -> u -> dest: d(c, v) + d(v, u) + d(u, dest)
            # Option 2: c -> u -> v -> dest: d(c, u) + d(u, v) + d(v, dest)
            # We pick the min of these two.
            
            # Get indices of remaining nodes
            rem_indices = [k for k in range(n) if k != i]
            v_idx = rem_indices[0]
            u_idx = rem_indices[1]
            
            # Path 1: c -> v -> u -> dest
            path1_cost = dist_sub[i, v_idx] + dist_sub[v_idx, u_idx] + dist_to_dest[u_idx]
            # Path 2: c -> u -> v -> dest
            path2_cost = dist_sub[i, u_idx] + dist_sub[u_idx, v_idx] + dist_to_dest[v_idx]
            
            cost_out = min(path1_cost, path2_cost)
            total_score = cost_in + cost_out
        else:
            # n > 3
            # For each v != c:
            #   cost_v = d(c, v) + min_{u != c, v} d(v, u)
            #   Note: We assume the next step from v is to its nearest neighbor u.
            #   We do NOT include d(u, dest) here because there are still >1 nodes left after u.
            #   The "min_{u} d(v, u)" is a greedy approximation for the rest of the tour from v.
            
            min_cost_from_v = float('inf')
            
            for j in range(n):
                if j == i:
                    continue
                # d(c, v)
                d_cv = dist_sub[i, j]
                
                # min_{u != c, v} d(v, u)
                min_vu = float('inf')
                for k in range(n):
                    if k == i or k == j:
                        continue
                    d_vu = dist_sub[j, k]
                    if d_vu < min_vu:
                        min_vu = d_vu
                
                cost_v = d_cv + min_vu
                if cost_v < min_cost_from_v:
                    min_cost_from_v = cost_v
            
            cost_out = min_cost_from_v
            total_score = cost_in + cost_out
            
        if total_score < best_score:
            best_score = total_score
            best_node = c
            
    return int(best_node)
# EVOLVE-BLOCK-END
