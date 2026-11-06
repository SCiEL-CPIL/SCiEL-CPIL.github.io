"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for precomputed data to speed up repeated calls
_cache = {}


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic for Capacitated VRP:
    Score each feasible customer by a combination of:
      - distance from current node
      - demand (prefer lower demand to save capacity)
      - return-to-depot cost (prefer nodes that don't force early return)
    
    Returns the best customer or depot (0) if no feasible customer.
    """
    if len(unvisited_nodes) == 0:
        return 0

    # Check if any feasible customers exist
    feasible_mask = demands[unvisited_nodes] <= rest_capacity
    if not np.any(feasible_mask):
        return 0

    # Get feasible unvisited nodes
    feasible_nodes = unvisited_nodes[feasible_mask]

    # Compute distances from current node to feasible nodes
    dists = distance_matrix[current_node, feasible_nodes]
    
    # Compute return-to-depot distances from feasible nodes
    return_dists = distance_matrix[feasible_nodes, depot]
    
    # Compute demands of feasible nodes
    feas_demands = demands[feasible_nodes]
    
    # Score: weight distance, return cost, and demand
    # Lower score = better
    # We want to minimize: travel distance + penalty for high demand + penalty for far return
    w_dist = 1.0
    w_return = 0.3
    w_demand = 0.5
    
    scores = w_dist * dists + w_return * return_dists + w_demand * feas_demands
    
    # Pick the node with minimum score
    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
