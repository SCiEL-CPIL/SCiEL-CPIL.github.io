"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0
_SAFETY_MARGIN = 5.0


def _time_exceeded():
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY_MARGIN)


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic for capacitated vehicle routing.
    Picks the next customer to visit, or returns depot (0) to close the route.
    
    Strategy: Multi-criteria scoring with lookahead and capacity awareness.
    """
    if len(unvisited_nodes) == 0:
        return int(depot)
    
    if _time_exceeded():
        # Fallback: nearest feasible node
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return int(depot)
        return int(feible[np.argmin(distance_matrix[current_node, feasible])])
    
    # Filter feasible nodes (those that fit in remaining capacity)
    feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    
    if not feasible:
        return int(depot)
    
    # If only one feasible, pick it
    if len(feasible) == 1:
        return int(feasible[0])
    
    # Compute distances from current node to all feasible nodes
    dists = distance_matrix[current_node, feasible]
    
    # Compute distance from each feasible node back to depot (to estimate route closure cost)
    dists_to_depot = distance_matrix[feasible, depot]
    
    # Compute remaining capacity after visiting each feasible node
    rem_caps = rest_capacity - np.array([demands[n] for n in feasible])
    
    # Estimate how many more nodes can be visited (rough: capacity / avg_demand)
    # This helps penalize nodes that leave too little capacity
    avg_demand = np.mean([demands[n] for n in unvisited_nodes]) if len(unvisited_nodes) > 0 else 1.0
    
    # Score each candidate:
    # 1. Immediate distance (lower is better)
    # 2. Distance to depot (lower is better - we want to be able to close route soon if needed)
    # 3. Remaining capacity after visit (higher is better - more flexibility)
    # 4. Lookahead: for each candidate, estimate cost of going to next-best node after it
    
    # Normalize components
    min_dist, max_dist = np.min(dists), np.max(dists)
    min_depot, max_depot = np.min(dists_to_depot), np.max(dists_to_depot)
    min_rem, max_rem = np.min(rem_caps), np.max(rem_caps)
    
    # Normalize to [0, 1] where 0 is best
    norm_dist = (dists - min_dist) / (max_dist - min_dist + 1e-9)
    norm_depot = (dists_to_depot - min_depot) / (max_depot - min_depot + 1e-9)
    norm_rem = (rem_caps - min_rem) / (max_rem - min_rem + 1e-9)
    norm_rem_inv = 1.0 - norm_rem  # invert: more remaining capacity is better
    
    # 2-step lookahead: for each candidate i, find min distance from i to any other feasible node j
    # that also fits in remaining capacity after visiting i
    lookahead = np.zeros(len(feasible))
    for i, node_i in enumerate(feasible):
        rem_after_i = rem_caps[i]
        # Find other feasible nodes that fit after visiting i
        other_feasible = [j for j in range(len(feasible)) if j != i and demands[feasible[j]] <= rem_after_i]
        if other_feasible:
            lookahead[i] = np.min(distance_matrix[node_i, [feasible[j] for j in other_feasible]])
        else:
            # No other node fits, so we'd have to go to depot
            lookahead[i] = distance_matrix[node_i, depot]
    
    min_look, max_look = np.min(lookahead), np.max(lookahead)
    norm_look = (lookahead - min_look) / (max_look - min_look + 1e-9)
    
    # Weighted combination
    w_dist = 0.4
    w_depot = 0.15
    w_rem = 0.2
    w_look = 0.25
    
    scores = w_dist * norm_dist + w_depot * norm_depot + w_rem * norm_rem_inv + w_look * norm_look
    
    # Pick the candidate with the lowest score
    best_idx = np.argmin(scores)
    return int(feasible[best_idx])
# EVOLVE-BLOCK-END
