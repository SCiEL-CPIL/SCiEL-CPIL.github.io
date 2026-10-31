"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = None
_DEADLINE_BUFFER = 10.0  # seconds safety margin before 150s limit

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 150.0 - _DEADLINE_BUFFER
    
    # Deadline guard: fall back to nearest neighbor if time is up
    if time.time() > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if n_unvisited == 2:
        # If only two left, pick the one that creates the shorter edge from current
        d0 = distance_matrix[current_node, unvisited_nodes[0]]
        d1 = distance_matrix[current_node, unvisited_nodes[1]]
        d_dest_0 = distance_matrix[unvisited_nodes[0], destination_node]
        d_dest_1 = distance_matrix[unvisited_nodes[1], destination_node]
        d_cross = distance_matrix[unvisited_nodes[0], unvisited_nodes[1]]
        
        # If we pick 0: path is ...->current->0->1->dest
        # Cost contribution: dist(current,0) + dist(0,1) + dist(1,dest)
        cost_pick_0 = d0 + d_cross + d_dest_1
        
        # If we pick 1: path is ...->current->1->0->dest
        # Cost contribution: dist(current,1) + dist(1,0) + dist(0,dest)
        cost_pick_1 = d1 + d_cross + d_dest_0
        
        if cost_pick_0 <= cost_pick_1:
            return int(unvisited_nodes[0])
        else:
            return int(unvisited_nodes[1])

    # --- Core Strategy: Nearest Neighbor with 1-step Look-Ahead ---
    # This is a robust constructive heuristic.
    # For each candidate node u in unvisited_nodes:
    # 1. Calculate immediate cost: dist(current, u)
    # 2. Calculate look-ahead cost:
    #    - If there are other unvisited nodes, find the nearest neighbor v of u (excluding u).
    #      Look-ahead cost = dist(u, v)
    #    - If u is the last unvisited node, look-ahead cost = dist(u, destination_node)
    # 3. Total score = immediate cost + look-ahead cost
    # Pick u with minimum total score.
    
    best_node = -1
    best_score = float('inf')
    
    for u in unvisited_nodes:
        # Immediate cost
        cost_curr_u = distance_matrix[current_node, u]
        
        # Look-ahead cost
        # Find other unvisited nodes
        others_mask = unvisited_nodes != u
        others = unvisited_nodes[others_mask]
        
        if len(others) > 0:
            # Find nearest neighbor to u among others
            dists_u_others = distance_matrix[u, others]
            min_dist_u_v = np.min(dists_u_others)
            look_ahead_cost = min_dist_u_v
        else:
            # u is the last one, must go to destination
            look_ahead_cost = distance_matrix[u, destination_node]
        
        total_score = cost_curr_u + look_ahead_cost
        
        if total_score < best_score:
            best_score = total_score
            best_node = u
            
    return int(best_node)

# EVOLVE-BLOCK-END
