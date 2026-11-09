"""Direct heuristic for TSP next-node selection."""
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s hard kill


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node for a TSP tour construction.
    
    Structural change: Two-stage lookahead heuristic.
    - Early phase (n > 10): Use nearest neighbor with a 2-step lookahead to avoid
      choosing a node that leads to a very expensive next step.
    - Late phase (n <= 10): Use a 3-step lookahead to carefully close the tour.
    """
    # Time guard
    if time.time() - _START_TIME > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    if n_unvisited == 2:
        u, v = unvisited_nodes
        # Cost of closing via u then v, or v then u
        cost_u = distance_matrix[current_node, u] + distance_matrix[u, v] + distance_matrix[v, destination_node]
        cost_v = distance_matrix[current_node, v] + distance_matrix[v, u] + distance_matrix[u, destination_node]
        return int(unvisited_nodes[0] if cost_u <= cost_v else unvisited_nodes[1])
    
    # For early phase, use 2-step lookahead
    # For late phase, use 3-step lookahead
    if n_unvisited > 10:
        # 2-step lookahead: minimize dist(current, i) + min(dist(i, j) for j in unvisited\{i})
        d_curr = distance_matrix[current_node, unvisited_nodes]
        # For each candidate i, compute the minimum distance from i to any other unvisited node
        # This is a 2-step lookahead
        # To speed this up, we can approximate by using the nearest neighbor of i within unvisited
        # But that's O(n^2) which is fine for n=50
        
        best_score = np.inf
        best_node = unvisited_nodes[0]
        
        for i_idx in range(n_unvisited):
            i = unvisited_nodes[i_idx]
            cost_to_i = d_curr[i_idx]
            # Find nearest neighbor of i among unvisited nodes (excluding i itself)
            if n_unvisited > 1:
                d_i_to_others = distance_matrix[i, unvisited_nodes]
                d_i_to_others[i_idx] = np.inf
                min_next = np.min(d_i_to_others)
            else:
                min_next = 0.0
            
            total_cost = cost_to_i + min_next
            if total_cost < best_score:
                best_score = total_cost
                best_node = i
        
        return int(best_node)
    else:
        # Late phase: 3-step lookahead
        # Minimize dist(current, i) + dist(i, j) + dist(j, destination)
        # where i, j are in unvisited, i != j
        best_score = np.inf
        best_node = unvisited_nodes[0]
        
        for i_idx in range(n_unvisited):
            i = unvisited_nodes[i_idx]
            cost_to_i = distance_matrix[current_node, i]
            
            # Find best j (j != i) that minimizes dist(i, j) + dist(j, destination)
            d_i_to_others = distance_matrix[i, unvisited_nodes]
            d_i_to_others[i_idx] = np.inf
            d_others_to_dest = distance_matrix[unvisited_nodes, destination_node]
            d_others_to_dest[i_idx] = np.inf
            
            # For each j, cost is d_i_to_others[j_idx] + d_others_to_dest[j_idx]
            j_costs = d_i_to_others + d_others_to_dest
            min_j_cost = np.min(j_costs)
            
            total_cost = cost_to_i + min_j_cost
            if total_cost < best_score:
                best_score = total_cost
                best_node = i
        
        return int(best_node)
