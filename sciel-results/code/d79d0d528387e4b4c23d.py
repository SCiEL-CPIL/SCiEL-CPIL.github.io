"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Static Regret Heuristic:
    Scores each feasible customer by the difference between the distance to its
    nearest feasible alternative and the distance to its second-nearest feasible
    alternative. Prioritizes nodes that are uniquely well-positioned (strongly
    clustered with a specific neighbor) to avoid future long detours.
    
    Tie-breaking: If regrets are equal, prefer the node closest to the current
    position to maintain local efficiency.
    """
    if len(unvisited_nodes) == 0:
        return depot

    # Filter to only feasible nodes (demand fits in remaining capacity)
    feasible_mask = demands[unvisited_nodes] <= rest_capacity
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        # No feasible customers left, go back to depot
        return depot

    # Get the feasible unvisited node indices (original node IDs)
    feasible_nodes = unvisited_nodes[feasible_indices]
    
    # If only one feasible node, go there
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])

    # Extract the submatrix of distances between feasible nodes
    # This allows us to compute distances between candidates, not just from current
    sub_dm = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
    
    # Set diagonal to infinity to exclude self-distances
    # Note: sub_dm is a copy, so this is safe
    np.fill_diagonal(sub_dm, np.inf)
    
    # Sort distances for each node to find nearest and second-nearest neighbors
    sorted_dists = np.sort(sub_dm, axis=1)
    
    d_nearest = sorted_dists[:, 0]
    d_second = sorted_dists[:, 1]
    
    # Calculate regret: difference between 2nd nearest and nearest
    # High regret indicates a strong, unique connection to a neighbor
    regrets = d_second - d_nearest
    
    # Distances from current node to feasible nodes for tie-breaking
    dists_to_current = distance_matrix[current_node, feasible_nodes]
    
    # We want to maximize regret. 
    # In case of ties, we want to minimize distance to current node.
    # np.lexsort uses the last key as the primary sort key.
    # Keys: (secondary: dists_to_current, primary: -regrets)
    # Sorting by -regrets ascending is equivalent to sorting by regrets descending.
    sort_indices = np.lexsort((dists_to_current, -regrets))
    
    best_idx = sort_indices[0]
    
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
