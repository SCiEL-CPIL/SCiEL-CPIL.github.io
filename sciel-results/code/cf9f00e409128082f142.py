"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a lookahead greedy heuristic.
    
    Instead of just picking the nearest neighbor, this heuristic estimates the 
    cost of moving to a candidate node plus the cost of moving from that 
    candidate to its nearest unvisited neighbor (or destination if none remain).
    This helps avoid local traps where a node is close to the current position 
    but far from the rest of the tour.
    """
    if len(unvisited_nodes) == 0:
        # Should not happen based on problem statement, but safe guard
        return destination_node
        
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
        
    if len(unvisited_nodes) == 2:
        # For 2 nodes, we can calculate the exact cost of closing the loop 
        # or proceeding to the last one.
        # Path: current -> c1 -> c2 -> dest vs current -> c2 -> c1 -> dest
        # Since c2 is the last one before dest in the first case, the cost is 
        # D[current, c1] + D[c1, c2] + D[c2, dest]
        # Actually, the function just picks the NEXT node. The evaluator handles the rest.
        # If we pick c1, the next call will have unvisited=[c2]. It will pick c2.
        # Then the tour closes to dest.
        # Total path segment: current->c1->c2->dest. Cost: D[cur,c1]+D[c1,c2]+D[c2,dest]
        # If we pick c2: current->c2->c1->dest. Cost: D[cur,c2]+D[c2,c1]+D[c1,dest]
        # We want to minimize the total future cost from current.
        c1, c2 = unvisited_nodes
        cost_1 = distance_matrix[current_node, c1] + distance_matrix[c1, c2] + distance_matrix[c2, destination_node]
        cost_2 = distance_matrix[current_node, c2] + distance_matrix[c2, c1] + distance_matrix[c1, destination_node]
        
        if cost_1 <= cost_2:
            return int(c1)
        else:
            return int(c2)

    # General case: Lookahead heuristic
    # Score for candidate c: D[current, c] + min(D[c, u] for u in unvisited if u != c)
    best_score = np.inf
    best_node = unvisited_nodes[0]
    
    # Vectorized calculation for speed, though N=50 is small
    # We compute distances from current to all unvisited
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # For each candidate, we need the min distance to other unvisited nodes.
    # This is slightly tricky to vectorize perfectly without loops due to the 'exclude self' part,
    # but N is small so a loop is fine and clearer.
    for i, c in enumerate(unvisited_nodes):
        # Immediate cost
        immediate = dists_from_current[i]
        
        # Lookahead cost: min distance from c to any other unvisited node
        # If there were only 1 other node, this is just that distance.
        # We exclude c itself.
        other_nodes = np.concatenate((unvisited_nodes[:i], unvisited_nodes[i+1:]))
        if len(other_nodes) > 0:
            lookahead = np.min(distance_matrix[c, other_nodes])
        else:
            # This case is handled by the len==1 check above, but just in case
            lookahead = distance_matrix[c, destination_node]
            
        total_score = immediate + lookahead
        
        if total_score < best_score:
            best_score = total_score
            best_node = c
            
    return int(best_node)
# EVOLVE-BLOCK-END
