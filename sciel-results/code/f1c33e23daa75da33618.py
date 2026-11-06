import numpy as np

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node to visit using a Nearest Neighbor heuristic with a
    depot return penalty. This balances the immediate distance to the node
    with the distance from that node back to the depot, encouraging compact
    routes that are close to the depot.
    """
    if len(unvisited_nodes) == 0:
        return int(depot)
        
    unvisited_arr = np.array(unvisited_nodes)
    demands_arr = np.array(demands)
    dm = np.array(distance_matrix)
    
    node_demands = demands_arr[unvisited_arr]
    feasible_mask = node_demands <= rest_capacity
    
    if not np.any(feasible_mask):
        return int(depot)
        
    feasible_nodes = unvisited_arr[feasible_mask]
    
    # Calculate distance from current node to all feasible nodes
    d_current_to_feasible = dm[current_node, feasible_nodes]
    
    # Calculate distance from feasible nodes to depot
    d_feasible_to_depot = dm[feasible_nodes, depot]
    
    # Score: distance to node + penalty * distance to depot
    # A penalty of 0.5 balances immediate cost with future return cost
    penalty = 0.5
    scores = d_current_to_feasible + penalty * d_feasible_to_depot
    
    # Find the index of the minimum score
    min_idx = np.argmin(scores)
    
    return int(feasible_nodes[min_idx])
