"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    if len(unvisited_nodes) == 0:
        return 0

    # Filter to feasible nodes based on capacity
    feasible_mask = demands[unvisited_nodes] <= rest_capacity
    feasible_nodes = unvisited_nodes[feasible_mask]

    if len(feasible_nodes) == 0:
        return 0

    # If only one feasible node, go there
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])

    dm = distance_matrix
    current = current_node
    depot_node = depot

    # Distances from current node to feasible nodes
    dist_from_current = dm[current, feasible_nodes]

    # Compute a density-weighted centroid in "distance space"
    # We use the inverse of distance from current as a weight (closer = higher density)
    # Add small epsilon to avoid division by zero
    eps = 1e-8
    weights = 1.0 / (dist_from_current + eps)
    weights /= weights.sum()

    # Since we don't have 2D coordinates, we approximate centroid using
    # the node that minimizes weighted average distance to all feasible nodes
    # This is a proxy for the "center" of the feasible cluster
    # Compute pairwise distances among feasible nodes
    n_feasible = len(feasible_nodes)
    if n_feasible > 1:
        # Pairwise distance matrix among feasible nodes
        sub_dm = dm[np.ix_(feasible_nodes, feasible_nodes)]
        # Weighted average distance from each feasible node to all others
        weighted_avg_dist = sub_dm @ weights
        # The "centroid" node is the one with minimum weighted average distance
        centroid_idx = np.argmin(weighted_avg_dist)
        centroid_node = feasible_nodes[centroid_idx]
    else:
        centroid_node = feasible_nodes[0]

    # Score each feasible candidate:
    # - Prefer nodes that are close to the centroid (maintain cluster density)
    # - Prefer nodes that are not too far from current (minimize immediate travel)
    # - Penalize nodes that would cause high intra-cluster variance
    dist_to_centroid = dm[feasible_nodes, centroid_node]
    dist_from_current_normalized = dist_from_current / (dist_from_current.max() + eps)

    # Combined score: lower is better
    # Weight: 0.4 * normalized distance from current + 0.6 * normalized distance to centroid
    score = 0.4 * dist_from_current_normalized + 0.6 * (dist_to_centroid / (dist_to_centroid.max() + eps))

    best_idx = np.argmin(score)
    best_node = feasible_nodes[best_idx]

    # Check if returning to depot is better
    # Return to depot if:
    # 1. No more feasible nodes after visiting best_node
    # 2. The cost to go to best_node and then to depot is much higher than going directly to depot
    # 3. We're close to depot and remaining demand is small
    dist_current_to_depot = dm[current, depot_node]
    dist_best_to_depot = dm[best_node, depot_node]

    # If going to best node then depot is significantly worse than going directly to depot
    # and we have few remaining nodes, consider closing
    if len(feasible_nodes) <= 3:
        # If the best node is far from depot and we're already close to depot, close
        if dist_current_to_depot < 0.3 * (dist_from_current[best_idx] + dist_best_to_depot):
            return 0

    # Also close if no feasible nodes remain after this choice
    remaining_after = feasible_nodes[feasible_nodes != best_node]
    if len(remaining_after) > 0:
        remaining_feasible_mask = demands[remaining_after] <= (rest_capacity - demands[best_node])
        if remaining_feasible_mask.sum() == 0:
            # No more feasible nodes after visiting best_node, check if depot is better
            # If going to best then depot vs current then depot
            if dist_current_to_depot < dist_from_current[best_idx] + dist_best_to_depot:
                return 0

    return int(best_node)
# EVOLVE-BLOCK-END
