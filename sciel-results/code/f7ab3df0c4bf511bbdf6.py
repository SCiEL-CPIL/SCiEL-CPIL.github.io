"""Spatial clustering heuristic for Capacitated VRP: area-based demand consolidation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_TIME_LIMIT = 118.0  # Safety margin before 150s kill


def _check_deadline():
    """Return True if we should stop and return depot for safety."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return (time.time() - _start_time) > _TIME_LIMIT


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Spatial clustering heuristic: score candidates by density of remaining
    feasible demand in their local region. Prioritizes nodes that allow the
    current route to exhaust capacity within a localized geographic cluster.
    
    Returns 0 (depot) to close route if no feasible customers or deadline exceeded.
    """
    # Deadline guard
    if _check_deadline():
        return 0
    
    if len(unvisited_nodes) == 0:
        return 0
    
    unvisited_arr = np.array(unvisited_nodes)
    
    # Filter to only feasible customers (demand <= rest_capacity)
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return 0
    
    n_feasible = len(feasible_nodes)
    
    # If only one feasible node, return it
    if n_feasible == 1:
        return int(feasible_nodes[0])
    
    # Distances from current node to feasible customers
    dist_from_current = distance_matrix[current_node, feasible_nodes]
    
    # Distances from feasible customers to depot
    dist_to_depot = distance_matrix[feasible_nodes, depot]
    
    # Pairwise distances among feasible customers
    pair_dists = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
    
    demands_feasible = demands[feasible_nodes].astype(float)
    
    # --- Spatial clustering score ---
    # For each candidate i, compute a "cluster density" score that measures
    # how much remaining feasible demand is concentrated near i.
    
    # Step 1: Compute a spatial weight for each pair (i, j) based on distance.
    # Use a Gaussian kernel: w_ij = exp(-d_ij / sigma)
    # sigma is adaptive based on the scale of the problem
    median_dist = np.median(pair_dists[pair_dists > 0]) if np.any(pair_dists > 0) else 1.0
    sigma = max(median_dist * 0.8, 1e-6)
    
    # Spatial weights (excluding self)
    spatial_weights = np.exp(-pair_dists / sigma)
    np.fill_diagonal(spatial_weights, 0.0)
    
    # Step 2: Weighted demand density around each candidate
    # density_i = sum_j w_ij * demand_j
    demand_density = spatial_weights @ demands_feasible
    
    # Step 3: Capacity utilization efficiency
    # Prefer nodes where the remaining capacity after visiting i is well-matched
    # to the local demand cluster
    remaining_cap_after = rest_capacity - demands_feasible
    # Local demand that could still be served after visiting i
    # Approximate by: total demand in neighborhood minus demand_i
    local_demand = demand_density  # already excludes self
    # Efficiency: how well the remaining capacity matches local demand
    cap_match = np.where(remaining_cap_after > 0, 
                         local_demand / (remaining_cap_after + 1e-9),
                         0.0)
    # Clip to avoid extreme values
    cap_match = np.clip(cap_match, 0, 5)
    
    # Step 4: Distance-based score
    # Primary: distance from current node
    # Secondary: return-to-depot distance (weighted)
    lambda_return = 0.4
    closure_cost = dist_from_current + lambda_return * dist_to_depot
    
    # Normalize closure_cost
    max_closure = np.max(closure_cost)
    if max_closure > 0:
        closure_norm = closure_cost / max_closure
    else:
        closure_norm = closure_cost
    
    # Normalize demand_density
    max_density = np.max(demand_density)
    if max_density > 0:
        density_norm = demand_density / max_density
    else:
        density_norm = demand_density
    
    # Normalize cap_match
    max_cap_match = np.max(cap_match)
    if max_cap_match > 0:
        cap_match_norm = cap_match / max_cap_match
    else:
        cap_match_norm = cap_match
    
    # Step 5: Combine scores
    # We want: low closure_cost, high density, high cap_match
    # Score = closure_norm - beta * density_norm - gamma * cap_match_norm
    # Lower score is better
    
    beta = 0.5   # Weight for spatial demand density
    gamma = 0.3  # Weight for capacity match
    
    scores = closure_norm - beta * density_norm - gamma * cap_match_norm
    
    # Additional: prefer nodes with higher demand (better capacity utilization)
    max_demand = np.max(demands_feasible)
    if max_demand > 0:
        demand_norm = demands_feasible / max_demand
    else:
        demand_norm = demands_feasible
    
    delta_w = 0.15
    scores += -delta_w * demand_norm
    
    # Pick the customer with minimum score
    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
