# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Select the next node using a dynamic regret-minimization framework.
    
    Improvements:
    1. Incorporates a depot-return bias into the edge cost to encourage compact routes.
    2. Enhances the capacity penalty to consider the number of remaining nodes,
       preventing premature capacity exhaustion more effectively.
    """
    unvisited = np.asarray(unvisited_nodes, dtype=int)
    
    if len(unvisited) == 0:
        return 0
    
    # Filter to feasible candidates (demand fits in rest_capacity)
    feasible_mask = demands[unvisited] <= rest_capacity
    feasible_nodes = unvisited[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return 0
    
    # Edge costs from current_node to each feasible candidate
    dist_to_candidates = distance_matrix[current_node, feasible_nodes]
    
    # Depot return costs: distance from candidate to depot
    # We weight this to encourage nodes that are close to the depot as well,
    # leading to more compact routes and lower total length.
    dist_to_depot = distance_matrix[feasible_nodes, depot]
    
    # Effective cost: combination of distance from current and distance to depot
    # Weight of 0.5 for depot return balances immediate travel vs. route closure cost
    effective_costs = dist_to_candidates + 0.5 * dist_to_depot
    
    # Average effective cost over all feasible candidates
    avg_effective_cost = np.mean(effective_costs)
    
    # Regret score: maximize (avg_effective_cost - effective_cost)
    # This prefers candidates that are significantly cheaper than the average
    regret_scores = -(effective_costs - avg_effective_cost)
    
    # Capacity-aware penalty
    remaining_caps = rest_capacity - demands[feasible_nodes]
    
    # Calculate average demand of all unvisited nodes
    avg_demand_all = np.mean(demands[unvisited]) if len(unvisited) > 0 else 0
    
    # Improved penalty: Check if remaining capacity is sufficient for the *average*
    # demand of the *remaining* nodes (not just one).
    # If we pick this node, we have `remaining_caps` for `len(unvisited) - 1` nodes.
    # The average capacity per remaining node would be remaining_caps / (len(unvisited) - 1).
    # If this is less than the average demand, we are likely to run out of capacity.
    
    num_remaining_after_pick = len(unvisited) - 1
    if num_remaining_after_pick > 0 and avg_demand_all > 0:
        # Average capacity available per remaining node
        avg_cap_per_node = remaining_caps / num_remaining_after_pick
        # Penalty if avg_cap_per_node is significantly lower than avg_demand_all
        # We use a soft penalty that scales with the deficit
        deficit = np.maximum(0, avg_demand_all - avg_cap_per_node)
        capacity_penalty = deficit / avg_demand_all
    else:
        # If no nodes remain or no demand, penalty is 0
        capacity_penalty = np.zeros_like(remaining_caps, dtype=float)
    
    # Combined score: regret score minus capacity penalty
    combined_scores = regret_scores - capacity_penalty
    
    # Pick the candidate with the highest combined score
    best_idx = np.argmax(combined_scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
