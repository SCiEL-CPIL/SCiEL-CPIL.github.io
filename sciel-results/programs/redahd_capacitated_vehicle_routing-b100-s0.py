import numpy as np
import time

# Deadline guard: kill after 150s wall-clock, return with safety margin
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # 5s safety margin

def _time_ok():
    return time.time() < _DEADLINE

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic for CVRP: Soft-Savings with lookahead.
    
    Uses a smooth surrogate (soft-max via log-sum-exp) for the savings criterion
    to avoid being locked to a single successor when multiple are near-optimal.
    Also incorporates a small lookahead penalty for candidates whose nearest
    feasible neighbor is far away.
    
    Returns the next customer node index, or depot (0) to close the route.
    """
    # Deadline check
    if not _time_ok():
        # Return nearest feasible node as a fast fallback
        if len(unvisited_nodes) == 0:
            return depot
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return depot
        # Nearest feasible
        best = min(feasible, key=lambda n: distance_matrix[current_node, n])
        return int(best)
    
    if len(unvisited_nodes) == 0:
        return depot
    
    unvisited_arr = np.array(unvisited_nodes, dtype=np.int64)
    
    # Feasible nodes: demand fits in remaining capacity
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    
    if not np.any(feasible_mask):
        return depot
    
    feasible_nodes = unvisited_arr[feasible_mask]
    n_feasible = len(feasible_nodes)
    
    # Distances
    d_cur_j = distance_matrix[current_node, feasible_nodes]  # (n_feasible,)
    d_dep_j = distance_matrix[depot, feasible_nodes]         # (n_feasible,)
    d_cur_dep = distance_matrix[current_node, depot]         # scalar
    
    # Savings: d(cur,dep) + d(dep,j) - d(cur,j)
    # Positive saving means it's cheaper to go to j than to return to depot first
    savings = d_cur_dep + d_dep_j - d_cur_j
    
    # Lookahead: for each candidate j, find the min distance from j to any other feasible node
    # (excluding j itself). This penalizes candidates that are isolated.
    if n_feasible > 1:
        # Distance matrix among feasible nodes
        D_ff = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]  # (n_feasible, n_feasible)
        # Set diagonal to inf so we don't pick self
        np.fill_diagonal(D_ff, np.inf)
        # Nearest neighbor distance for each feasible node
        nn_dist = np.min(D_ff, axis=1)  # (n_feasible,)
    else:
        nn_dist = np.array([0.0])
    
    # Soft-min surrogate for the "cost of closing" decision:
    # Instead of hard argmax on savings, use a smooth penalty that accounts for
    # both the immediate savings and a capacity-aware lookahead.
    
    # Capacity-aware lookahead: penalize nodes that consume too much of remaining
    # capacity relative to their savings, as this may force premature route closure.
    # Also use soft-min (log-sum-exp) for the nearest-neighbor distance to smooth
    # the nonsmooth min operation.
    
    # Soft-min temperature: lower = more like hard min, higher = more uniform
    soft_min_temp = 2.0
    
    # Compute soft-min of distances to other feasible nodes (excluding self)
    if n_feasible > 1:
        # D_ff already has diagonal set to inf
        # Soft-min: -T * log(sum(exp(-x/T)))
        neg_dist_over_T = -D_ff / soft_min_temp
        # Subtract max for numerical stability
        neg_max = np.max(neg_dist_over_T, axis=1)
        soft_min_vals = -soft_min_temp * np.log(np.sum(np.exp(neg_dist_over_T - neg_max), axis=1))
        # Replace inf values (from diagonal) with 0
        soft_min_vals = np.where(np.isinf(soft_min_vals), 0.0, soft_min_vals)
    else:
        soft_min_vals = np.array([0.0])
    
    # Capacity ratio: how much of remaining capacity this node would consume
    # High ratio means we're using up capacity quickly, which may force early return
    capacity_ratio = demands[feasible_nodes] / max(rest_capacity, 1)
    
    # Combined score with capacity-aware penalty
    # Higher savings is better, lower soft-min distance is better,
    # but high capacity ratio is penalized more when savings are low
    beta_nn = 0.03  # weight for nearest-neighbor penalty
    beta_cap = 0.15  # weight for capacity consumption penalty
    
    # Adaptive capacity penalty: stronger when savings are low
    # This encourages taking high-savings nodes even if they consume more capacity
    savings_norm = savings - np.min(savings)
    savings_range = max(np.max(savings_norm), 1e-9)
    savings_frac = savings_norm / savings_range  # 0 to 1
    
    # When savings are high (close to 1), reduce capacity penalty
    # When savings are low (close to 0), increase capacity penalty
    adaptive_cap_weight = beta_cap * (1.0 - 0.5 * savings_frac)
    
    score = savings - beta_nn * soft_min_vals - adaptive_cap_weight * capacity_ratio * np.max(savings)
    
    # If the best saving is very negative, return to depot
    if np.max(savings) < -1e-9:
        return depot
    
    # Select the node with the highest combined score
    best_idx = np.argmax(score)
    best_node = feasible_nodes[best_idx]
    
    return int(best_node)
