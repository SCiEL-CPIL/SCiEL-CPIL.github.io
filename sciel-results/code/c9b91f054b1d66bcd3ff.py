# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_LIMIT = 148.0  # Safety margin before 150s kill


def _fast_mst_weight(nodes, distance_matrix):
    """Compute MST weight for a set of nodes using Prim's algorithm with numpy vectorization."""
    n = len(nodes)
    if n <= 1:
        return 0.0
    
    # Build local distance submatrix
    local_dist = distance_matrix[np.ix_(nodes, nodes)]
    
    in_mst = np.zeros(n, dtype=bool)
    min_dist = np.full(n, np.inf)
    min_dist[0] = 0.0
    total_weight = 0.0
    
    for _ in range(n):
        # Find node with minimum distance not in MST
        masked = min_dist.copy()
        masked[in_mst] = np.inf
        best_idx = np.argmin(masked)
        
        if min_dist[best_idx] == np.inf:
            break
            
        in_mst[best_idx] = True
        total_weight += min_dist[best_idx]
        
        # Update distances using vectorized operation
        new_dists = local_dist[best_idx]
        update_mask = ~in_mst
        min_dist[update_mask] = np.minimum(min_dist[update_mask], new_dists[update_mask])
        
    return total_weight


def _mst_residual_estimate(nodes, distance_matrix):
    """
    Estimate residual tour cost for remaining nodes using MST-based heuristic.
    For a set of n nodes, the tour cost through them is roughly 2*MST_weight / n * (n-1)
    or more accurately, we use the fact that a TSP tour is at most 2*MST.
    We estimate the average edge cost in the residual tour.
    """
    n = len(nodes)
    if n <= 1:
        return 0.0
    if n == 2:
        return distance_matrix[nodes[0], nodes[1]]
    
    mst_w = _fast_mst_weight(nodes, distance_matrix)
    # The residual tour will visit n nodes, adding n-1 edges (plus the closing edge)
    # Average edge contribution per node is roughly mst_w / (n-1)
    return mst_w / max(n - 1, 1)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    TSP constructive heuristic using MST-residual-guided selection.
    
    Strategy: For each candidate next node, estimate the total remaining tour cost
    as: dist(current, candidate) + residual_estimate(remaining_after_candidate)
    where residual_estimate uses MST weight of the remaining unvisited nodes to
    approximate the cost of completing the tour from those nodes back to destination.
    
    This prioritizes moves that preserve low-cost connectivity for the remaining sub-tour.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _LIMIT - 0.5:
        # Fallback to nearest neighbor
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    if n_unvisited == 2:
        c1 = int(unvisited_nodes[0])
        c2 = int(unvisited_nodes[1])
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c1, c2] + distance_matrix[c2, destination_node]
        cost2 = distance_matrix[current_node, c2] + distance_matrix[c2, c1] + distance_matrix[c1, destination_node]
        return c1 if cost1 <= cost2 else c2
    
    # Compute immediate distances
    dist_to_candidates = distance_matrix[current_node, unvisited_nodes]
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    dist_current_to_dest = distance_matrix[current_node, destination_node]
    
    # For larger sets, use MST-based residual estimation
    # This is O(n^2) per MST computation, and we do it for each candidate, so O(n^3) total
    # For n up to ~30-40, this should be fast enough with numpy
    
    if n_unvisited <= 35 and elapsed < _LIMIT - 2.0:
        # Compute MST residual for each candidate
        scores = np.zeros(n_unvisited)
        base_mst = _fast_mst_weight(unvisited_nodes, distance_matrix)
        
        for i in range(n_unvisited):
            if time.time() - _start_time > _LIMIT - 1.0:
                # Fallback: use simple scoring for remaining
                break
            
            c = unvisited_nodes[i]
            remaining = np.concatenate([unvisited_nodes[:i], unvisited_nodes[i+1:]])
            
            # Immediate cost
            immediate = dist_to_candidates[i]
            
            # Residual estimate: cost to complete tour from 'remaining' nodes back to destination
            if len(remaining) == 0:
                residual = dist_to_dest[i]
            elif len(remaining) == 1:
                residual = distance_matrix[c, remaining[0]] + distance_matrix[remaining[0], destination_node]
            else:
                # Use MST-based estimate for the residual sub-tour
                # The sub-tour starts at 'c', visits all 'remaining', ends at 'destination'
                # Approximate: dist(c, nearest in remaining) + MST(remaining) + dist(farthest?, dest)
                # Better: MST of (remaining + [c, destination]) minus the two longest edges
                # Simplest: MST(remaining) + min edge from c to remaining + min edge from remaining to dest
                mst_rem = _fast_mst_weight(remaining, distance_matrix)
                min_from_c = np.min(distance_matrix[c, remaining])
                min_to_dest = np.min(dist_to_dest[i+1:] if i < n_unvisited-1 else dist_to_dest[:i])
                residual = min_from_c + mst_rem + min_to_dest
            
            scores[i] = immediate + residual
        
        # Check if we completed all scores
        if np.any(np.isnan(scores)) or (time.time() - _start_time > _LIMIT - 1.0 and n_unvisited > 10):
            # Fallback to simpler scoring
            pass
        else:
            best_idx = np.argmin(scores)
            return int(unvisited_nodes[best_idx])
    
    # Fallback: use detour + distance heuristic (fast)
    detours = dist_to_candidates + dist_to_dest - dist_current_to_dest
    
    # Normalize
    max_detour = np.max(detours)
    if max_detour > 1e-10:
        normalized_detours = detours / max_detour
    else:
        normalized_detours = detours
    
    max_dist = np.max(dist_to_candidates)
    if max_dist > 1e-10:
        normalized_local = dist_to_candidates / max_dist
    else:
        normalized_local = dist_to_candidates
    
    # Weighted combination
    alpha = 0.6  # local distance weight
    beta = 0.4   # detour weight
    
    scores = alpha * normalized_local + beta * normalized_detours
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
