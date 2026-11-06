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


def _held_karp_suffix(current_node, unvisited_nodes, distance_matrix, dest_node):
    """
    Compute the optimal suffix path starting from current_node, visiting all
    unvisited_nodes, and ending at dest_node, using Held-Karp DP.
    Returns the first node in the optimal path (int).
    """
    U = np.asarray(unvisited_nodes)
    n = U.size
    if n == 0:
        return int(current_node)
    if n == 1:
        return int(U[0])
    
    c = int(current_node)
    dest = int(dest_node)
    
    # Map unvisited nodes to indices 0..n-1
    node_map = {int(u): i for i, u in enumerate(U)}
    
    # dp[mask][i] = min cost to start at current_node, visit set 'mask' of unvisited nodes,
    # and end at unvisited node i (index in U)
    # mask is a bitmask of which unvisited nodes have been visited
    # We need to visit all n nodes, so full mask is (1 << n) - 1
    
    # For n up to 15, this is 2^15 * 15 = 491520 states, which is feasible.
    # But we need to be careful with time. Let's limit n to 15.
    if n > 15:
        # Fallback to greedy if too many nodes
        d = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[int(np.argmin(d))])
    
    INF = float('inf')
    dp = np.full((1 << n, n), INF, dtype=np.float64)
    parent = np.full((1 << n, n), -1, dtype=np.int32)
    
    # Base case: start at current_node, go to each unvisited node i
    for i in range(n):
        dp[1 << i, i] = distance_matrix[c, U[i]]
    
    # Fill DP table
    for mask in range(1, 1 << n):
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            if dp[mask, i] == INF:
                continue
            # Try to extend to node j not in mask
            for j in range(n):
                if mask & (1 << j):
                    continue
                new_mask = mask | (1 << j)
                new_cost = dp[mask, i] + distance_matrix[U[i], U[j]]
                if new_cost < dp[new_mask, j]:
                    dp[new_mask, j] = new_cost
                    parent[new_mask, j] = i
    
    full_mask = (1 << n) - 1
    
    # Find the best last node before dest
    best_last = -1
    best_cost = INF
    for i in range(n):
        cost = dp[full_mask, i] + distance_matrix[U[i], dest]
        if cost < best_cost:
            best_cost = cost
            best_last = i
    
    # Reconstruct the first node
    # We need to find the sequence. The first node is the one with mask = 1 << first
    # We can trace back from (full_mask, best_last)
    cur_mask = full_mask
    cur_node = best_last
    while parent[cur_mask, cur_node] != -1:
        cur_mask = cur_mask & ~(1 << cur_node)
        cur_node = parent[cur_mask, cur_node]
    
    return int(U[cur_node])


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Deadline guard: the evaluator kills after 150 s. If we are near the limit,
    # fall back to the cheap nearest-neighbour choice (no heavy lookahead).
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

        # Immediate edge cost current -> candidate.
        edge = dm[c, U]

        if n <= 3:
            # Exact: pick candidate that minimises edge + cost to close to destination.
            score = edge + dm[U, dest]
            return int(U[int(np.argmin(score))])

        # If we have few enough unvisited nodes, use Held-Karp for exact optimal suffix
        if n <= 15:
            # Check time budget: Held-Karp for n=15 is ~2^15 * 15^2 = 7.3M ops
            # Each op is a few numpy scalar ops, so maybe ~50ms-100ms in Python
            # With 50 cities, we call this 50 times. If we use HK for last 15,
            # that's 15 calls * 100ms = 1.5s. Should be fine.
            try:
                result = _held_karp_suffix(c, U, dm, dest)
                return result
            except Exception:
                pass

        # Lookahead penalty: how hard is it to reach the destination after
        # committing to a candidate? Approximated by a soft-min (log-sum-exp)
        # of distances from the candidate to other unvisited nodes, plus the
        # destination closure cost. The soft-min smooths the nonsmooth min
        # objective, giving a differentiable surrogate that penalises
        # candidates whose remaining neighbours are all far away.
        #
        # For each candidate u: softmin[u] = -beta * log(sum_{v in U, v != u} exp(-beta * dm[u, v]))
        # As beta -> inf, softmin -> min. For finite beta, it's a smooth
        # approximation that also considers the second-nearest, etc.
        sub = dm[np.ix_(U, U)]
        # Mask diagonal with a large value so exp(-beta*large) ~ 0
        beta = 5.0  # temperature: higher = closer to hard min
        sub_diag_masked = sub.copy()
        np.fill_diagonal(sub_diag_masked, 1e10)
        # Log-sum-exp for numerical stability: -beta * log(sum(exp(-beta * sub)))
        # = -beta * (max + log(sum(exp(-beta * (sub - max)))))
        # Let m = -beta * max(sub) so we compute log(sum(exp(-beta * sub)))
        neg_beta_sub = -beta * sub_diag_masked
        # Subtract row max for stability
        row_max = neg_beta_sub.max(axis=1, keepdims=True)
        # Handle case where row_max might be -inf (all masked) - shouldn't happen for n>=2
        row_max_safe = np.where(np.isfinite(row_max), row_max, 0.0)
        log_sum_exp = row_max_safe.squeeze() + np.log(np.sum(np.exp(neg_beta_sub - row_max_safe), axis=1))
        softmin = -log_sum_exp  # this is the soft-min value

        # Destination closure cost
        dest_cost = dm[U, dest]

        # Combined lookahead penalty
        pen = softmin + dest_cost

        # Blend immediate and lookahead costs. Immediate edge dominates; the
        # lookahead term breaks ties and avoids dead-end choices that strand
        # the tour far from the destination.
        score = edge + 0.6 * pen

        return int(U[int(np.argmin(score))])
    except Exception:
        d = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[int(np.argmin(d))])
# EVOLVE-BLOCK-END
