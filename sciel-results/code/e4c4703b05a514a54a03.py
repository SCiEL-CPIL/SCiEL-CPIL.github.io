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

def _mst_cost(nodes, dm):
    """
    Compute the cost of the Minimum Spanning Tree on the given nodes using the distance matrix.
    Uses Prim's algorithm for efficiency.
    """
    n = len(nodes)
    if n <= 1:
        return 0.0
    
    # Extract submatrix
    sub = dm[np.ix_(nodes, nodes)]
    
    # Prim's algorithm
    in_mst = np.zeros(n, dtype=bool)
    min_dist = np.full(n, np.inf)
    min_dist[0] = 0.0
    
    total_cost = 0.0
    for _ in range(n):
        # Find node with minimum distance
        in_mst_mask = ~in_mst
        if not np.any(in_mst_mask):
            break
        min_dist[in_mst] = np.inf
        u = np.argmin(min_dist)
        if min_dist[u] == np.inf:
            break
        in_mst[u] = True
        total_cost += min_dist[u]
        
        # Update distances
        for v in range(n):
            if not in_mst[v] and sub[u, v] < min_dist[v]:
                min_dist[v] = sub[u, v]
    
    return total_cost

def _estimate_completion_cost(start, dest, rem, dm, use_mst=True):
    """
    Estimate the cost of completing a tour from 'start' through all nodes in 'rem' to 'dest'.
    Uses a combination of MST lower bound and greedy heuristic.
    """
    n = len(rem)
    if n == 0:
        return dm[start, dest]
    if n == 1:
        return dm[start, rem[0]] + dm[rem[0], dest]
    if n == 2:
        n0, n1 = rem[0], rem[1]
        cost0 = dm[start, n0] + dm[n0, n1] + dm[n1, dest]
        cost1 = dm[start, n1] + dm[n1, n0] + dm[n0, dest]
        return min(cost0, cost1)
    
    # For larger n, use a hybrid approach:
    # 1. Compute MST cost of the remaining nodes (global connectivity)
    # 2. Add connection costs from start and to dest
    
    if use_mst:
        mst = _mst_cost(rem, dm)
        
        # Find the best way to connect 'start' to the MST and 'dest' to the MST
        # Approximation: find min distance from start to any node in rem
        # and min distance from dest to any node in rem
        d_start = np.min(dm[start, rem])
        d_dest = np.min(dm[dest, rem])
        
        # The MST spans all rem nodes. We need to add start and dest as leaves.
        # A good lower bound is mst + d_start + d_dest
        # But this might be too optimistic. Let's use a slightly more conservative estimate.
        # Actually, for a path from start to dest visiting all rem, 
        # the cost is at least mst(rem) + min_edge_from_start + min_edge_to_dest
        # But we also need to ensure it's a single path, not a tree.
        
        # Let's use a greedy completion for a more accurate estimate,
        # but weighted by the MST structure to avoid subtours.
        
        # Simple greedy with 2-step lookahead
        curr = start
        rem_list = list(rem)
        cost = 0.0
        
        while rem_list:
            m = len(rem_list)
            if m == 1:
                cost += dm[curr, rem_list[0]] + dm[rem_list[0], dest]
                break
            elif m == 2:
                n0, n1 = rem_list[0], rem_list[1]
                cost0 = dm[curr, n0] + dm[n0, n1] + dm[n1, dest]
                cost1 = dm[curr, n1] + dm[n1, n0] + dm[n0, dest]
                cost += min(cost0, cost1)
                break
            else:
                # 2-step lookahead
                d_curr = dm[curr, rem_list]
                rem_arr = np.array(rem_list)
                rem_dist = dm[np.ix_(rem_arr, rem_arr)]
                np.fill_diagonal(rem_dist, np.inf)
                min_next = np.min(rem_dist, axis=1)
                scores = d_curr + min_next
                best_idx = np.argmin(scores)
                next_node = rem_list[best_idx]
                cost += d_curr[best_idx]
                curr = next_node
                rem_list.pop(best_idx)
        
        # Blend greedy estimate with MST-based lower bound
        # The MST bound is a lower bound, greedy is an upper bound.
        # We want to be more careful about connectivity, so let's weight the MST component.
        mst_bound = mst + d_start + d_dest
        # Use a convex combination: 0.5 * greedy + 0.5 * mst_bound
        # This penalizes solutions that create disconnected structures
        return 0.5 * cost + 0.5 * mst_bound
    else:
        # Pure greedy
        curr = start
        rem_list = list(rem)
        cost = 0.0
        while rem_list:
            m = len(rem_list)
            if m == 1:
                cost += dm[curr, rem_list[0]] + dm[rem_list[0], dest]
                break
            elif m == 2:
                n0, n1 = rem_list[0], rem_list[1]
                cost0 = dm[curr, n0] + dm[n0, n1] + dm[n1, dest]
                cost1 = dm[curr, n1] + dm[n1, n0] + dm[n0, dest]
                cost += min(cost0, cost1)
                break
            else:
                d_curr = dm[curr, rem_list]
                rem_arr = np.array(rem_list)
                rem_dist = dm[np.ix_(rem_arr, rem_arr)]
                np.fill_diagonal(rem_dist, np.inf)
                min_next = np.min(rem_dist, axis=1)
                scores = d_curr + min_next
                best_idx = np.argmin(scores)
                next_node = rem_list[best_idx]
                cost += d_curr[best_idx]
                curr = next_node
                rem_list.pop(best_idx)
        return cost

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a global connectivity-aware heuristic.
    
    For each candidate c, we estimate the cost of completing the tour from c
    to destination_node using a hybrid MST + greedy strategy.
    We select the candidate that minimizes the total estimated tour length.
    """
    # Time guard
    remaining = _get_time_remaining()
    if remaining < 3.0:
        # Fallback to simple nearest neighbor
        if len(unvisited_nodes) == 0:
            return int(destination_node)
        if len(unvisited_nodes) == 1:
            return int(unvisited_nodes[0])
        cands = unvisited_nodes
        scores = distance_matrix[current_node, cands] + distance_matrix[cands, destination_node]
        return int(cands[np.argmin(scores)])
    
    candidates = unvisited_nodes
    n = len(candidates)
    
    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(candidates[0])
    
    # For small n, do exact evaluation
    if n == 2:
        c0 = candidates[0]
        c1 = candidates[1]
        cost0 = distance_matrix[current_node, c0] + distance_matrix[c0, c1] + distance_matrix[c1, destination_node]
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c1, c0] + distance_matrix[c0, destination_node]
        return int(c0) if cost0 < cost1 else int(c1)
    
    # Adaptive strategy based on time remaining and number of candidates
    # Use MST-aware scoring for better global connectivity
    
    # For very large n, limit candidates to top-k by simple score to save time
    top_k = min(n, 20)
    if n > top_k:
        simple_scores = distance_matrix[current_node, candidates] + distance_matrix[candidates, destination_node]
        top_indices = np.argsort(simple_scores)[:top_k]
        candidates = candidates[top_indices]
        n = len(candidates)
    
    best_candidate = -1
    best_score = float('inf')
    
    for c in candidates:
        # Estimate cost of completing tour from c
        rem = np.array([x for x in unvisited_nodes if x != c], dtype=int)
        total_cost = distance_matrix[current_node, c] + _estimate_completion_cost(c, destination_node, rem, distance_matrix, use_mst=True)
        
        if total_cost < best_score:
            best_score = total_cost
            best_candidate = c
        
        # Check time
        if _get_time_remaining() < 5.0:
            break
    
    return int(best_candidate)
