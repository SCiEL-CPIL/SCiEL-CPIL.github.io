"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node to visit using a heuristic that balances proximity to the current node
    and proximity to the depot to create compact routes.
    
    Args:
        current_node: The node currently being visited.
        depot: The index of the depot (usually 0).
        unvisited_nodes: An array-like object of indices of nodes not yet visited.
        rest_capacity: The remaining capacity of the vehicle.
        demands: An array of demands for each node.
        distance_matrix: A 2D array of distances between nodes.
        
    Returns:
        The index of the next node to visit, or 0 (depot) if the route should be closed.
    """
    if len(unvisited_nodes) == 0:
        return depot

    # Convert to numpy array for vectorized operations if not already
    unvisited = np.asarray(unvisited_nodes)
    
    # Filter feasible nodes: demand must be <= rest_capacity
    # demands is indexed by node id
    feasible_mask = demands[unvisited] <= rest_capacity
    feasible_nodes = unvisited[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot

    # Calculate scores for feasible nodes
    # Score = distance from current to node + distance from node to depot
    # This encourages visiting nodes that are close to the current position 
    # and also close to the depot, preventing long detours.
    dists_to_current = distance_matrix[current_node, feasible_nodes]
    dists_to_depot = distance_matrix[feasible_nodes, depot]
    
    scores = dists_to_current + dists_to_depot
    
    # Find the node with the minimum score
    best_idx = np.argmin(scores)
    next_node = feasible_nodes[best_idx]
    
    return int(next_node)
# EVOLVE-BLOCK-END
