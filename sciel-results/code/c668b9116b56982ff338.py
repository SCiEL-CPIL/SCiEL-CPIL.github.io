"""Improved angular sweep heuristic for capacitated vehicle routing."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Select the next node using a hybrid Angular Sweep and Nearest Neighbor heuristic.
    
    This heuristic balances the direct travel distance with an angular penalty that
    encourages the route to sweep consistently around the depot. The angular penalty
    is scaled by the average radius to approximate arc length deviation.
    """
    depot = int(depot)
    current_node = int(current_node)
    
    # Filter unvisited nodes by capacity feasibility
    feasible = []
    for node in unvisited_nodes:
        node = int(node)
        if demands[node] <= rest_capacity:
            feasible.append(node)
    
    if not feasible:
        return 0  # Close route, return to depot
    
    # If starting from depot, use pure Nearest Neighbor to establish initial direction
    if current_node == depot:
        best_node = min(feasible, key=lambda x: distance_matrix[depot, x])
        return int(best_node)
    
    d_depot_current = distance_matrix[depot, current_node]
    
    best_node = feasible[0]
    best_score = float('inf')
    
    # Tune weight: Balance between distance and angular consistency
    # A value around 0.5-1.0 typically works well for Euclidean-like metrics
    angular_weight = 0.8
    
    for node in feasible:
        node = int(node)
        d_depot_node = distance_matrix[depot, node]
        d_current_node = distance_matrix[current_node, node]
        
        # Compute angle at depot using law of cosines
        a = d_depot_current
        b = d_depot_node
        c = d_current_node
        
        if a < 1e-12 or b < 1e-12:
            angle = 0.0
        else:
            cos_theta = (a*a + b*b - c*c) / (2.0 * a * b)
            cos_theta = max(-1.0, min(1.0, cos_theta))
            angle = np.arccos(cos_theta)
        
        # Scale angular penalty by average radius to represent arc length deviation
        avg_radius = 0.5 * (a + b) + 1e-12
        arc_penalty = angle * avg_radius
        
        # Combine distance and angular penalty
        # Distance is the primary cost, angular penalty discourages backtracking/zig-zag
        score = d_current_node + angular_weight * arc_penalty
        
        if score < best_score:
            best_score = score
            best_node = node
    
    return int(best_node)
# EVOLVE-BLOCK-END
