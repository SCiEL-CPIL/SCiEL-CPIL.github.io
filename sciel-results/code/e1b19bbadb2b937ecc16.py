"""CVRP heuristic: select_next_node with physics-inspired force field + regret-based scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 120.0  # seconds
_SAFETY_MARGIN = 5.0  # seconds before hard limit


def _time_remaining():
    return _DEADLINE - _SAFETY_MARGIN - (time.time() - _START_TIME)


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Select the next customer to visit or return to depot (0).
    
    Strategy: Combine multiple scoring heuristics:
    1. Nearest neighbor with capacity feasibility
    2. Regret-based scoring (how much worse is the second-best option?)
    3. Force-field inspired: balance proximity with demand distribution
    4. Return to depot when capacity is tight or all remaining nodes are far
    """
    if _time_remaining() < 0.1:
        # Emergency: just pick nearest feasible or depot
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return depot
        return int(feasible[np.argmin(distance_matrix[current_node, feasible])])

    unvisited = list(unvisited_nodes)
    if not unvisited:
        return depot

    # Filter by capacity feasibility
    feasible = [n for n in unvisited if demands[n] <= rest_capacity]
    if not feasible:
        return depot

    # If only one feasible, take it
    if len(feasible) == 1:
        return int(feasible[0])

    # Convert to arrays for vectorized ops
    feasible_arr = np.array(feasible)
    dists = distance_matrix[current_node, feasible_arr]
    dems = demands[feasible_arr]
    
    # Score 1: Inverse distance (prefer closer)
    # Add small epsilon to avoid division by zero
    inv_dist = 1.0 / (dists + 1e-9)
    
    # Score 2: Regret-based - how much do we lose if we don't pick this one?
    # For each feasible node, compute what the best alternative would be
    # Approximate regret: distance to node - min distance to other feasible nodes
    # We want nodes where the alternative is much worse (high regret)
    # This encourages picking nodes that are "unique" in their location
    
    # Compute for each node: best distance among OTHER feasible nodes from current
    # This is expensive, so approximate with: dist - min_dist_overall
    min_dist_overall = np.min(dists)
    regret = dists - min_dist_overall  # Higher regret means this node is more "costly" to skip
    # Actually we want to PREFER nodes with high regret (if we skip them, we pay more later)
    # But also consider the distance to depot for future route closing
    
    # Score 3: Force-field inspired
    # Attractive force from customers weighted by demand, repulsive from depot
    # Net force direction should guide us to balance load and proximity
    # Approximate: prefer nodes that are close but also have higher demand (to fill capacity)
    demand_weight = dems / np.max(dems)  # Normalize demand
    proximity_weight = inv_dist / np.max(inv_dist)  # Normalize proximity
    
    # Combined force score: balance proximity and demand
    # Higher demand nodes are more important to visit (they consume more capacity)
    # But we also want to minimize total distance
    force_score = 0.6 * proximity_weight + 0.4 * demand_weight
    
    # Score 4: Consider distance to depot for route closing
    # If rest capacity is low, prefer nodes close to depot (easier to close route)
    depot_dists = distance_matrix[feasible_arr, depot]
    if rest_capacity < 20:  # Low remaining capacity
        depot_proximity = 1.0 / (depot_dists + 1e-9)
        depot_proximity_norm = depot_proximity / np.max(depot_proximity)
        # Blend in depot proximity
        combined_score = 0.5 * force_score + 0.3 * (1.0 - regret / (np.max(regret) + 1e-9)) + 0.2 * depot_proximity_norm
    else:
        # Normal case: focus on proximity and demand
        # Invert regret: lower regret is better (we don't lose much by skipping)
        # Actually, high regret means we should visit this node soon
        regret_norm = regret / (np.max(regret) + 1e-9)
        combined_score = 0.5 * force_score + 0.3 * regret_norm + 0.2 * (1.0 - regret_norm)
        # Simplify: combined_score = 0.5 * force_score + 0.1 * regret_norm
        combined_score = 0.7 * force_score + 0.3 * regret_norm

    # Pick the node with highest combined score
    best_idx = np.argmax(combined_score)
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
