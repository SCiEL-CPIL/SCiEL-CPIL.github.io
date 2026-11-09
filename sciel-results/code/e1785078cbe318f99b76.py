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

    # Pre-compute distance submatrix between candidates: dist_sub[i, j] = d(candidates[i], candidates[j])
    dist_sub = distance_matrix[np.ix_(candidates, candidates)]
    # Distance from each candidate to destination
    dist_to_dest = distance_matrix[candidates, destination_node]
    # Distance from current to each candidate
    dist_from_current = distance_matrix[current_node, candidates]

    best_score = float('inf')
    best_node = candidates[0]

    if n == 2:
        # Two nodes c, v. Path: current -> c -> v -> destination
        for i in range(n):
            j = 1 - i
            total = dist_from_current[i] + dist_sub[i, j] + dist_to_dest[j]
            if total < best_score:
                best_score = total
                best_node = candidates[i]
        return int(best_node)

    if n == 3:
        # Three nodes. For candidate c, remaining two v, u.
        # Path: current -> c -> {v or u} -> {other} -> destination
        for i in range(n):
            rem = [k for k in range(n) if k != i]
            a, b = rem[0], rem[1]
            path1 = dist_sub[i, a] + dist_sub[a, b] + dist_to_dest[b]
            path2 = dist_sub[i, b] + dist_sub[b, a] + dist_to_dest[a]
            total = dist_from_current[i] + min(path1, path2)
            if total < best_score:
                best_score = total
                best_node = candidates[i]
        return int(best_node)

    # n > 3: 2-step lookahead with destination-awareness.
    # For each candidate c (index i):
    #   cost_in  = d(current, c)
    #   cost_2step = min over v != c of [ d(c, v) + min( d(v, w) for w != c,v, d(v, dest) ) ]
    #   dest_term = w * d(c, destination)
    # This looks ahead two edges, approximating the degree-2 completion cost.
    
    if n <= 6:
        w = 0.5
    elif n <= 12:
        w = 0.3
    elif n <= 25:
        w = 0.15
    else:
        w = 0.05

    # Precompute for each candidate the min distance to any other candidate (excluding self)
    # and the second-min for the 2-step lookahead
    min_dist_from = np.empty(n)
    min_idx_from = np.empty(n, dtype=int)
    second_min_dist_from = np.empty(n)
    
    for i in range(n):
        row = dist_sub[i].copy()
        row[i] = float('inf')
        # Sort to get min and second min
        sorted_idx = np.argsort(row)
        min_dist_from[i] = row[sorted_idx[0]]
        min_idx_from[i] = sorted_idx[0]
        second_min_dist_from[i] = row[sorted_idx[1]] if n > 2 else float('inf')
    
    # For 2-step: for candidate i, we go to some v, then from v to either:
    #   - the nearest neighbor of v (excluding i and v itself), or
    #   - the destination (if v is the last node)
    # We approximate by: min over v != i of [ d(i,v) + min( second_min_dist_from[v], dist_to_dest[v] ) ]
    # But we need to be careful: if v's nearest neighbor is i, we should use second_min
    
    for i in range(n):
        # 2-step cost: min over v != i of [ d_sub[i,v] + best_out_from_v_excluding_i ]
        # best_out_from_v_excluding_i = min( d(v, w) for w != v, w != i, d(v, dest) )
        # Approximate: for each v, precompute min distance to any other node, and if that min is i, use second_min
        two_step_costs = np.empty(n)
        for v in range(n):
            if v == i:
                two_step_costs[v] = float('inf')
                continue
            # Best way to leave v, excluding going back to i
            # If v's nearest neighbor is i, use second_min; otherwise use min
            if min_idx_from[v] == i:
                leave_cost = second_min_dist_from[v]
            else:
                leave_cost = min_dist_from[v]
            # Also consider going to destination from v
            leave_cost = min(leave_cost, dist_to_dest[v])
            two_step_costs[v] = dist_sub[i, v] + leave_cost
        
        cost_2step = float(np.min(two_step_costs))
        total = dist_from_current[i] + cost_2step + w * dist_to_dest[i]
        
        if total < best_score:
            best_score = total
            best_node = candidates[i]

    return int(best_node)
# EVOLVE-BLOCK-END
