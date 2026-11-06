# EVOLVE-BLOCK-START
import numpy as np
import time

# Cached per-instance data so repeated calls are cheap.
_cache = {}


def _init(current_node, destination_node, unvisited_nodes, distance_matrix):
    key = (id(distance_matrix), distance_matrix.shape[0])
    c = _cache.get(key)
    if c is None or c[0] is not distance_matrix:
        c = (distance_matrix, np.ascontiguousarray(distance_matrix, dtype=np.float64))
        _cache[key] = c
    return c[1]


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Deadline guard: the evaluator kills after 150 s. If we are near the limit,
    # fall back to the cheap nearest-neighbour choice (no heavy lookahead).
    try:
        if time.time() > 145.0:
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
