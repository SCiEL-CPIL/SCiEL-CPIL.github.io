import numpy as np

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Convert to numpy arrays if not already for speed
    unvisited = np.array(unvisited_nodes)
    
    if len(unvisited) == 0:
        return int(depot)
        
    # Filter feasible nodes
    node_demands = np.array(demands)[unvisited]
    feasible_mask = node_demands <= rest_capacity
    feasible_nodes = unvisited[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return int(depot)
        
    # Calculate distances from current to feasible nodes
    dists_curr = distance_matrix[current_node, feasible_nodes]
    
    # Pick the node with the minimum distance (Nearest Neighbor)
    # This ensures we always visit a customer if one is feasible,
    # preventing premature return to depot.
    min_idx = np.argmin(dists_curr)
    best_node = feasible_nodes[min_idx]
    
    return int(best_node)
