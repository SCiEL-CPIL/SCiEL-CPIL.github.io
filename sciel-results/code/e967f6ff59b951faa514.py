"""Capacitated VRP heuristic: capacity-aware nearest neighbor with soft-min."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard: we have 120s per evaluation, use 110s as safety margin
_EVAL_START_TIME = None

def _get_elapsed():
    global _EVAL_START_TIME
    if _EVAL_START_TIME is None:
        _EVAL_START_TIME = time.time()
    return time.time() - _EVAL_START_TIME


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick next customer or depot (0) to close route.
    
    Strategy: capacity-aware scoring with soft-min surrogate.
    - Score each candidate by distance * (1 + penalty) where penalty accounts
      for how much remaining capacity is being consumed relative to the node's demand.
    - Use a smooth (soft) combination to avoid hard min over-penalizing outliers.
    - Return depot (0) if no feasible nodes remain.
    """
    # Check deadline: if we're getting close to the limit, return depot to be safe
    if _get_elapsed() > 110.0:
        return 0

    if len(unvisited_nodes) == 0:
        return 0

    # Filter to feasible nodes (demand fits in remaining capacity)
    node_arr = np.asarray(unvisited_nodes)
    node_demands = demands[node_arr]
    feasible_mask = node_demands <= rest_capacity
    
    if not np.any(feasible_mask):
        return 0

    feasible_nodes = node_arr[feasible_mask]
    feasible_demands = node_demands[feasible_mask]

    # If only depot is feasible, go to depot
    if len(feasible_nodes) == 0:
        return 0

    # Distances from current node to all feasible candidates
    dists = distance_matrix[current_node, feasible_nodes]

    # Capacity penalty: penalize nodes that consume a large fraction of remaining capacity
    # This encourages filling routes more efficiently
    # penalty_i = (demand_i / rest_capacity)^gamma
    # Higher demand relative to remaining capacity -> higher penalty
    gamma = 1.0
    cap_ratio = feasible_demands / max(rest_capacity, 1e-9)
    penalty = cap_ratio ** gamma

    # Combined score: distance + lambda * penalty
    # lambda balances distance vs. capacity efficiency
    lam = 0.3
    scores = dists + lam * penalty

    # Also consider the "savings" perspective: going to a node with high demand
    # when remaining capacity is low is wasteful. We already handle this via penalty.
    
    # Use soft-min to get a smoothed selection (avoids hard ties and numerical issues)
    # soft_min(x, tau) = -tau * log(sum(exp(-x_i / tau)))
    # argmin of soft-min is not exactly argmin of x, but for small tau it's close.
    # Here we just use it as a scoring mechanism; we pick argmin of scores.
    # Actually, for argmin purposes, hard min is fine. The soft-min idea is more
    # relevant for gradient-based optimization. Let's just use argmin.
    
    # But let's add a small smoothing: subtract a tiny amount based on rank to break ties
    # and avoid always picking the same node when distances are equal.
    
    idx = np.argmin(scores)
    return int(feasible_nodes[idx])
# EVOLVE-BLOCK-END
