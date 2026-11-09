"""Capacitated VRP next-node heuristic: savings-based with capacity feasibility."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_MODULE_START = time.time()
TIME_LIMIT = 110.0  # seconds; safety margin below 120s/150s limits


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic for capacitated VRP next-node selection.
    
    Strategy: For each feasible candidate (demand fits rest_capacity), compute a
    savings-based score. The idea is that visiting a node early saves the cost of
    going depot->node->depot vs depot->node->...->depot. We approximate this by
    favoring nodes that are close to the current node (continuity) and also close
    to the depot (so we don't end up far from depot with little capacity left).
    
    Additionally, we use a look-ahead: among feasible nodes, prefer those that
    have other feasible nodes nearby (cluster formation), which reduces total
    route length by grouping similar-demand nodes.
    
    If no feasible customer exists, return depot (0).
    """
    # Deadline guard
    elapsed = time.time() - _MODULE_START
    if elapsed > TIME_LIMIT:
        if len(unvisited_nodes) == 0:
            return 0
        feasible = [u for u in unvisited_nodes if demands[u] <= rest_capacity]
        if not feasible:
            return 0
        return int(min(feasible, key=lambda u: distance_matrix[current_node, u]))

    if len(unvisited_nodes) == 0:
        return 0

    # Filter by capacity feasibility
    feasible_mask = np.array([demands[u] <= rest_capacity for u in unvisited_nodes])
    if not np.any(feasible_mask):
        return 0

    feasible_nodes = unvisited_nodes[feasible_mask]
    n_feasible = len(feasible_nodes)

    if n_feasible == 1:
        return int(feasible_nodes[0])

    # Distances from current node to all feasible nodes
    dists_from_current = distance_matrix[current_node, feasible_nodes]
    
    # Distances from each feasible node to depot
    dists_to_depot = distance_matrix[feasible_nodes, depot]
    
    # For each feasible node, compute min distance to any OTHER feasible node
    # This encourages clustering
    min_dist_to_other_feasible = np.full(n_feasible, np.inf)
    for i in range(n_feasible):
        node_i = feasible_nodes[i]
        # Distance to all other feasible nodes
        if n_feasible > 1:
            other_indices = [j for j in range(n_feasible) if j != i]
            if other_indices:
                d_to_others = distance_matrix[node_i, feasible_nodes[other_indices]]
                min_dist_to_other_feasible[i] = np.min(d_to_others)
            else:
                min_dist_to_other_feasible[i] = dists_to_depot[i]
        else:
            min_dist_to_other_feasible[i] = dists_to_depot[i]
    
    # Composite score:
    # 1. Primary: distance from current to candidate (keep route contiguous)
    # 2. Secondary: distance to depot (prefer nodes that are closer to depot when
    #    capacity is getting tight, to avoid ending far from depot)
    # 3. Tertiary: min distance to other feasible nodes (encourage clustering)
    
    # Weight the components. As rest_capacity decreases relative to total demand,
    # we should care more about not ending up far from depot.
    # Simple adaptive weight: if rest_capacity is small, weight depot distance more.
    total_demand = np.sum(demands[unvisited_nodes])
    if total_demand > 0:
        cap_ratio = rest_capacity / total_demand
    else:
        cap_ratio = 1.0
    
    # When capacity is tight (low ratio), emphasize depot proximity
    w_depot = 0.5 * (1.0 - cap_ratio) + 0.1  # ranges from 0.1 to 0.6
    w_cluster = 0.2  # fixed weight for clustering
    
    scores = dists_from_current + w_depot * dists_to_depot + w_cluster * min_dist_to_other_feasible
    
    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
