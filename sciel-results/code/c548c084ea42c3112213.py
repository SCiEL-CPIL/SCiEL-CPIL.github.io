import numpy as np

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Convert to numpy arrays if necessary for performance and vectorization
    if not isinstance(distance_matrix, np.ndarray):
        distance_matrix = np.array(distance_matrix)
    if not isinstance(demands, np.ndarray):
        demands = np.array(demands)
    if not isinstance(unvisited_nodes, np.ndarray):
        unvisited_nodes = np.array(unvisited_nodes)
        
    # If no unvisited nodes, return to depot
    if len(unvisited_nodes) == 0:
        return int(depot)
        
    # Identify feasible nodes based on remaining capacity
    # demands is an array indexed by node ID
    # unvisited_nodes is an array of node IDs
    feasible_mask = demands[unvisited_nodes] <= rest_capacity
    feasible_nodes = unvisited_nodes[feasible_mask]
    
    # If no feasible nodes, return to depot
    if len(feasible_nodes) == 0:
        return int(depot)
        
    # Calculate Clarke-Wright savings for each feasible node
    # Savings = D(depot, current) + D(depot, j) - D(current, j)
    d_depot_current = distance_matrix[depot, current_node]
    d_depot_j = distance_matrix[depot, feasible_nodes]
    d_current_j = distance_matrix[current_node, feasible_nodes]
    
    savings = d_depot_current + d_depot_j - d_current_j
    
    # Add a small bias towards higher demand to break ties and prefer filling capacity
    # The bias is small enough not to override the savings logic significantly
    demand_bias = 0.01 * demands[feasible_nodes]
    
    scores = savings + demand_bias
    
    # Select the node with the highest score
    best_idx = np.argmax(scores)
    return int(feasible_nodes[best_idx])
