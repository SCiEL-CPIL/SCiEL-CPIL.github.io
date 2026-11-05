import numpy as np

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    if len(unvisited_nodes) == 0:
        return depot
    
    # Filter to feasible nodes (demand fits in remaining capacity)
    feasible = []
    for j in unvisited_nodes:
        if demands[j] <= rest_capacity:
            feasible.append(j)
    
    if len(feasible) == 0:
        return depot
    
    feasible = np.array(feasible)
    
    # Demand-Aware Nearest Neighbor
    # Minimize distance from current node, with a small penalty for high demand
    # to ensure we don't use up all capacity on a single far/high-demand node
    dist_c_j = distance_matrix[current_node, feasible]
    demand_j = demands[feasible]
    
    # Normalize demand to have similar scale as distance?
    # Distances are likely in some units, demands are 1-9.
    # Let's use a small weight for demand.
    weight_demand = 0.5
    
    scores = dist_c_j + weight_demand * demand_j
    
    best_idx = np.argmin(scores)
    return int(feasible[best_idx])
