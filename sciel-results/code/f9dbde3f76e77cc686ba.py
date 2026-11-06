"""Rewritten program: Savings-based (Clarke-Wright) heuristic with geometric tie-breaking."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_start_time = time.time()
_DEADLINE = 140.0  # Leave 10s safety margin before 150s limit


def _time_exceeded():
    return (time.time() - _start_time) > _DEADLINE


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next customer (or depot 0 to close) using a savings-based heuristic
    combined with geometric angle consistency to reduce backtracking.
    
    Strategy:
    1. Filter feasible customers (demand fits remaining capacity).
    2. If none feasible, return depot (0).
    3. Compute "savings" for each candidate: dist(depot, i) + dist(depot, j) - dist(i, j)
       where i = current_node, j = candidate.
    4. Use a composite score that balances savings and angular consistency.
    5. Return the best candidate.
    """
    unvisited = np.asarray(unvisited_nodes, dtype=int)
    n_unvisited = len(unvisited)
    
    if n_unvisited == 0:
        return 0
    
    # Check time
    if _time_exceeded():
        # Fallback: nearest feasible node
        feasible_mask = np.array([demands[u] <= rest_capacity for u in unvisited])
        if not np.any(feasible_mask):
            return 0
        feasible_nodes = unvisited[feasible_mask]
        dists = distance_matrix[current_node, feasible_nodes]
        return int(feasible_nodes[np.argmin(dists)])
    
    # Feasibility mask
    feasible_mask = np.array([demands[u] <= rest_capacity for u in unvisited])
    
    if not np.any(feasible_mask):
        return 0
    
    feasible_nodes = unvisited[feasible_mask]
    n_feasible = len(feasible_nodes)
    
    if n_feasible == 1:
        return int(feasible_nodes[0])
    
    # Get distances
    dist_from_current = distance_matrix[current_node, feasible_nodes]
    dist_depot_to_current = distance_matrix[depot, current_node]
    dist_depot_to_candidates = distance_matrix[depot, feasible_nodes]
    
    # Compute savings: dist(depot, current) + dist(depot, j) - dist(current, j)
    savings = dist_depot_to_current + dist_depot_to_candidates - dist_from_current
    
    # Compute angular consistency score
    # Vector from depot to current node
    # We need coordinates, but we only have distance matrix.
    # Instead, use a proxy: prefer candidates that are "in the same direction" as current from depot.
    # This can be approximated by: if dist(depot, current) + dist(current, j) ≈ dist(depot, j),
    # then j is roughly in the direction from depot through current.
    # So we want dist(depot, current) + dist(current, j) - dist(depot, j) to be small.
    # This is the "detour" cost.
    detour = dist_depot_to_current + dist_from_current - dist_depot_to_candidates
    # Smaller detour means better angular alignment.
    
    # Composite score: high savings is good, small detour is good.
    # Normalize both to [0, 1] range for combining.
    if n_feasible > 1:
        savings_min = np.min(savings)
        savings_max = np.max(savings)
        savings_range = savings_max - savings_min + 1e-10
        savings_norm = (savings - savings_min) / savings_range
        
        detour_min = np.min(detour)
        detour_max = np.max(detour)
        detour_range = detour_max - detour_min + 1e-10
        # Invert detour so that smaller detour = higher score
        detour_norm = 1.0 - (detour - detour_min) / detour_range
        
        # Also consider distance from current: closer is generally better
        dist_min = np.min(dist_from_current)
        dist_max = np.max(dist_from_current)
        dist_range = dist_max - dist_min + 1e-10
        dist_norm = 1.0 - (dist_from_current - dist_min) / dist_range
        
        # Weighted combination: savings most important, then angular, then proximity
        composite = 0.5 * savings_norm + 0.3 * detour_norm + 0.2 * dist_norm
    else:
        composite = np.array([0.0])
    
    best_idx = np.argmax(composite)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
