# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = _T0 + 145.0  # safety margin before 150s kill


def _time_ok():
    return (time.time() - _T0) < _DEADLINE


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    n = len(unvisited_nodes)
    if n == 0:
        return int(depot)

    unvisited_arr = np.asarray(unvisited_nodes, dtype=np.intp)
    k = len(unvisited_arr)

    # Feasibility: demand must fit in remaining capacity
    feas_mask = demands[unvisited_arr] <= rest_capacity
    if not np.any(feas_mask):
        return int(depot)

    feas = unvisited_arr[feas_mask]
    m = len(feas)

    # If only one feasible, take it
    if m == 1:
        return int(feas[0])

    # Distances from current node to each feasible candidate
    d_cur = distance_matrix[current_node, feas]

    # Savings heuristic: for candidate j, the "savings" of visiting j next
    # is approximated by: d(current, j) + min_{other unvisited o} d(j, o) - d(current, o)
    # But a simpler proxy: reward candidates that are close to current AND close to
    # other unvisited nodes (i.e., they form a natural continuation).
    # We compute for each feasible j:
    #   score_j = d_cur[j] - alpha * (mean distance from j to other unvisited nodes)
    # Lower score = better (closer to current, and also close to others => compact)
    
    # Compute mean distance from each feasible node to all other unvisited nodes
    # d_matrix among unvisited: shape (k, k)
    d_unvisited = distance_matrix[np.ix_(unvisited_arr, unvisited_arr)]
    
    # For each feasible index in the 'feas' array, get its row in d_unvisited
    # Map feas back to indices in unvisited_arr
    # Create an index map: for each element in feas, find its position in unvisited_arr
    # Since feas is a subset of unvisited_arr, we can use np.searchsorted if sorted,
    # but easier: use a dictionary or just compute directly.
    
    # Alternative: compute mean distance from each feasible node to all unvisited
    # d_unvisited is (k, k) where k = len(unvisited_arr)
    # For each feasible node at position p in unvisited_arr, mean dist to others =
    #   (sum of row p) / (k - 1)  [excluding self]
    
    # Get positions of feas in unvisited_arr
    # Since feas is a subarray of unvisited_arr, we can find positions by:
    pos_map = {}
    for i, u in enumerate(unvisited_arr):
        pos_map[u] = i
    
    feas_positions = np.array([pos_map[f] for f in feas], dtype=np.intp)
    
    # Row sums for feasible positions
    row_sums = d_unvisited[feas_positions].sum(axis=1)
    # Mean distance to other unvisited (excluding self, which is 0)
    mean_d_to_others = row_sums / max(k - 1, 1)
    
    # Score: d_cur[j] - alpha * mean_d_to_others[j]
    # We want to minimize distance to current, but also prefer nodes that are
    # well-connected to others (lower mean distance to others means more compact)
    alpha = 0.4
    scores = d_cur - alpha * mean_d_to_others
    
    # Pick the candidate with minimum score
    best_idx = np.argmin(scores)
    return int(feas[best_idx])
# EVOLVE-BLOCK-END
