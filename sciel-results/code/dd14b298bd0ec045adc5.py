"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from datetime import datetime, timedelta

# Global deadline guard
_EVAL_START = datetime.now()
DEADLINE = _EVAL_START + timedelta(seconds=145)  # Safety margin before 150s kill


def _time_remaining():
    return (DEADLINE - datetime.now()).total_seconds()


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic for selecting the next node in a capacitated vehicle routing problem.
    Uses a savings-based lookahead approach with capacity constraints.
    """
    if len(unvisited_nodes) == 0:
        return int(depot)
    
    # Check if we're running out of time
    if _time_remaining() < 1.0:
        # Fallback to simple nearest feasible
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return int(depot)
        dists = distance_matrix[current_node, feasible]
        return int(feasible[np.argmin(dists)])
    
    # Find feasible nodes (can fit in remaining capacity)
    feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    
    if len(feasible) == 0:
        return int(depot)
    
    if len(feasible) == 1:
        return int(feasible[0])
    
    # Use a savings-based heuristic with lookahead
    # The idea: for each feasible node i, compute a "savings" score that considers:
    # 1. Distance from current to i
    # 2. Distance from i back to depot (if we close after i)
    # 3. Potential savings from not going back to depot
    
    # Simple approach: use nearest neighbor with a twist
    # For each candidate, estimate the cost of visiting it vs. returning to depot
    # We want to minimize total route length, so we prefer nodes that are close
    # and also have good "continuation" potential
    
    # Compute distances from current to all feasible nodes
    dist_to_feasible = distance_matrix[current_node, feasible]
    
    # For each feasible node, compute a score:
    # score = dist(current, i) + min(dist(i, j) for j in remaining feasible after i) * weight
    # This is a 1-step lookahead
    
    # To keep it fast, use a simpler heuristic:
    # score_i = dist(current, i) - alpha * (dist(depot, i) - dist(depot, current))
    # The term (dist(depot, i) - dist(depot, current)) approximates the savings
    # of going to i instead of returning to depot from current
    
    # Actually, let's use the classic savings approach:
    # If we're at current and go to i, the "savings" compared to separate routes is:
    # savings_i = dist(depot, current) + dist(depot, i) - dist(current, i)
    # But since we're already at current, the relevant comparison is:
    # Option 1: Go to i, then continue
    # Option 2: Return to depot
    # The cost of going to i is dist(current, i)
    # The cost of returning to depot is dist(current, depot)
    # So we prefer i if dist(current, i) is small, but also consider what comes after
    
    # Let's use a combined score:
    # score_i = dist(current, i) + beta * min_{j in feasible, j != i} dist(i, j)
    # This favors nodes that are close to current AND have a close neighbor for next step
    
    if len(feasible) <= 20:
        # For small sets, do proper lookahead
        scores = np.zeros(len(feasible))
        for idx, i in enumerate(feasible):
            dist_ci = distance_matrix[current_node, i]
            # Find the nearest other feasible node from i
            if len(feasible) > 1:
                other_indices = [j for j in range(len(feasible)) if j != idx]
                dists_from_i = distance_matrix[i, feasible[other_indices]]
                min_next = np.min(dists_from_i)
            else:
                min_next = 0.0
            # Combine: prefer close to current, and close to next
            scores[idx] = dist_ci + 0.3 * min_next
        best_idx = np.argmin(scores)
        return int(feasible[best_idx])
    else:
        # For larger sets, use a faster approximation
        # Just use nearest neighbor with a slight penalty for nodes far from depot
        dists = distance_matrix[current_node, feasible]
        # Add a small bias toward nodes closer to depot (to keep routes compact)
        depot_dists = distance_matrix[depot, feasible]
        scores = dists + 0.1 * depot_dists
        best_idx = np.argmin(scores)
        return int(feasible[best_idx])
# EVOLVE-BLOCK-END
