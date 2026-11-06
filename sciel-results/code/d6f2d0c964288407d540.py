# EVOLVE-BLOCK-START
import numpy as np
import time

# Cached per-instance data so repeated calls are cheap.
_cache = {}
_start_time = None


def _get_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time


def _init(current_node, destination_node, unvisited_nodes, distance_matrix):
    key = (id(distance_matrix), distance_matrix.shape[0])
    c = _cache.get(key)
    if c is None or c[0] is not distance_matrix:
        c = (distance_matrix, np.ascontiguousarray(distance_matrix, dtype=np.float64))
        _cache[key] = c
    return c[1]


def _held_karp_suffix(current_node, unvisited_nodes, distance_matrix, dest_node, max_n=10):
    """
    Compute the optimal suffix path starting from current_node, visiting all
    unvisited_nodes, and ending at dest_node, using Held-Karp DP.
    Returns the first node in the optimal path (int).
    If n > max_n, falls back to nearest neighbor.
    """
    U = np.asarray(unvisited_nodes)
    n = U.size
    if n == 0:
        return int(current_node)
    if n == 1:
        return int(U[0])
    
    c = int(current_node)
    dest = int(dest_node)
    
    if n > max_n:
        # Too many nodes for exact DP, fall back to greedy
        d = distance_matrix[c, U]
        return int(U[int(np.argmin(d))])
    
    # For each candidate first node, compute the optimal cost of the remaining path
    # This is O(n * 2^(n-1) * n^2)
    # For n=10, this is 10 * 512 * 100 = 512,000 ops. Very fast.
    
    best_cost = 1e10
    best_first = 0
    
    for first in range(n):
        rem_indices = [i for i in range(n) if i != first]
        m = len(rem_indices)
        
        if m == 0:
            cost = distance_matrix[c, U[first]] + distance_matrix[U[first], dest]
            if cost < best_cost:
                best_cost = cost
                best_first = first
            continue
        
        # DP for remaining nodes
        rem_mask_size = 1 << m
        # dp[mask * m + i] = min cost to start at U[first], visit set 'mask' of rem_indices, end at rem_indices[i]
        dp = np.full(rem_mask_size * m, 1e10, dtype=np.float64)
        
        # Base case: go from U[first] to U[rem_indices[i]]
        for i in range(m):
            dp[(1 << i) * m + i] = distance_matrix[U[first], U[rem_indices[i]]]
        
        # Iterate over masks
        for mask in range(1, rem_mask_size):
            base_idx = mask * m
            for i in range(m):
                if not (mask & (1 << i)):
                    continue
                cost_i = dp[base_idx + i]
                if cost_i >= 1e10:
                    continue
                remaining = (rem_mask_size - 1) ^ mask
                for j in range(m):
                    if remaining & (1 << j):
                        new_mask = mask | (1 << j)
                        new_cost = cost_i + distance_matrix[U[rem_indices[i]], U[rem_indices[j]]]
                        idx = new_mask * m + j
                        if new_cost < dp[idx]:
                            dp[idx] = new_cost
        
        full_rem_mask = rem_mask_size - 1
        min_final = 1e10
        for i in range(m):
            val = dp[full_rem_mask * m + i]
            if val < 1e10:
                total = val + distance_matrix[U[rem_indices[i]], dest]
                if total < min_final:
                    min_final = total
        
        total_cost = distance_matrix[c, U[first]] + min_final
        if total_cost < best_cost:
            best_cost = total_cost
            best_first = first
    
    return int(U[best_first])


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Deadline guard: the evaluator kills after 150 s. If we are near the limit,
    # fall back to the cheap nearest-neighbour choice.
    try:
        elapsed = time.time() - _get_start_time()
        if elapsed > 140.0:
            d = distance_matrix[current_node, unvisited_nodes]
            return int(unvisited_nodes[int(np.argmin(d))])
    except Exception:
        pass

    try:
        dm = _init(current_node, destination_node, unvisited_nodes, distance_matrix)
        U = np.asarray(unvisited_nodes)
        n = U.size
        if n == 0:
            return int(current_node)
        if n == 1:
            return int(U[0])

        c = int(current_node)
        dest = int(destination_node)

        # Use Held-Karp for small n (up to 10 to ensure speed)
        if n <= 10:
            return _held_karp_suffix(c, U, dm, dest, max_n=10)
        
        # For larger n, use a 2-step lookahead heuristic
        edge = dm[c, U]
        
        if n <= 3:
            score = edge + dm[U, dest]
            return int(U[int(np.argmin(score))])

        # 2-step lookahead
        sub = dm[np.ix_(U, U)]
        dest_col = dm[U, dest]
        
        # best_cont[i] = min_{j != i} (dm[i, j] + dm[j, dest])
        # We can compute this efficiently
        # extended[i, j] = dm[i, j] + dest_col[j]
        extended = sub + dest_col[np.newaxis, :]
        # Set diagonal to inf so j != i
        np.fill_diagonal(extended, 1e10)
        best_cont = extended.min(axis=1)
        
        # Also consider direct return if it's better (though in a TSP tour, you must visit all nodes, 
        # so "direct return" is only valid if n==1. For n>1, you must go to another node.
        # However, the "lookahead" is an estimate of the cost of the remaining tour.
        # If we pick i, we pay dm[c, i] + (cost of visiting rest).
        # The estimate for cost of visiting rest starting from i is min_{j != i} (dm[i, j] + estimate_from_j).
        # Here estimate_from_j is approximated by dm[j, dest].
        # This is a valid 2-step lookahead.
        
        score = edge + best_cont
        return int(U[int(np.argmin(score))])
    except Exception:
        d = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[int(np.argmin(d))])
# EVOLVE-BLOCK-END
