"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _soft_min(values, temperature=0.5):
    """Soft minimum via log-sum-exp trick (lower temperature = sharper)."""
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    v = np.asarray(values, dtype=np.float64)
    v_min = np.min(v)
    # Avoid overflow: exp(-(v - v_min)/T) where T is temperature
    scaled = -(v - v_min) / temperature
    # Clip to avoid overflow
    scaled = np.clip(scaled, -500, 500)
    exp_sum = np.sum(np.exp(scaled))
    if exp_sum == 0:
        return float(v_min)
    return float(v_min + temperature * np.log(exp_sum))


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Deadline guard: if we're running long, just return nearest feasible
    start_time = time.time()
    
    unvisited = list(unvisited_nodes)
    if not unvisited:
        return depot
    
    # Filter by capacity
    feasible = [u for u in unvisited if demands[u] <= rest_capacity]
    
    if not feasible:
        return depot
    
    if len(feasible) == 1:
        return int(feasible[0])
    
    # Convert to numpy for vectorized ops
    feasible_arr = np.array(feasible, dtype=np.int64)
    
    # Distance from current node to each feasible node
    dist_from_current = distance_matrix[current_node, feasible_arr]
    
    # Distance from each feasible node to depot
    dist_to_depot = distance_matrix[feasible_arr, depot]
    
    # If only one feasible, return it
    if len(feasible) == 1:
        return int(feasible[0])
    
    # Heuristic scoring:
    # We want to minimize total route length. 
    # A good heuristic: prefer nodes that are close to current AND close to depot
    # (this helps close routes efficiently). But also consider not leaving many
    # hard-to-reach nodes for later.
    
    # Simple approach: score = dist_from_current + alpha * dist_to_depot
    # where alpha balances between going to nearby nodes vs. nodes near depot.
    # A lower alpha means we prioritize proximity to current (greedy nearest).
    # A higher alpha means we prioritize nodes near depot (to close routes sooner).
    
    # Let's try a combination:
    # - Primary: distance from current (greedy)
    # - Secondary: distance to depot (to avoid stragglers)
    # - Tertiary: demand (prefer higher demand nodes to fill capacity)
    
    # Normalize distances for combination
    max_dist_cur = np.max(dist_from_current) if np.max(dist_from_current) > 0 else 1.0
    max_dist_dep = np.max(dist_to_depot) if np.max(dist_to_depot) > 0 else 1.0
    
    # Score: lower is better
    # Weighted combination
    w1 = 0.7  # weight for distance from current
    w2 = 0.2  # weight for distance to depot
    w3 = 0.1  # weight for demand (prefer higher demand to fill capacity)
    
    norm_dist_cur = dist_from_current / max_dist_cur
    norm_dist_dep = dist_to_depot / max_dist_dep
    norm_demand = demands[feasible_arr] / float(np.max(demands[feasible_arr]))
    
    scores = w1 * norm_dist_cur + w2 * norm_dist_dep - w3 * norm_demand
    
    # Also consider: if we have very little capacity left, prefer to go to depot
    # unless there's a node we can serve that's very close
    if rest_capacity <= 2:
        # Only go to depot if no feasible node is very close
        min_dist = np.min(dist_from_current)
        if min_dist > 0.3:  # threshold: if closest is far, go to depot
            return depot
    
    # Pick the node with minimum score
    best_idx = np.argmin(scores)
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
