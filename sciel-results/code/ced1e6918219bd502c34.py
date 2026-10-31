import numpy as np
import time

# EVOLVE-BLOCK-START
import numpy as np
import time

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node in the TSP tour using a 2-step lookahead heuristic.
    
    Args:
        current_node: The node we are currently at.
        destination_node: The node we must return to (start node).
        unvisited_nodes: An array of indices of cities not yet visited.
        distance_matrix: A 2D array where dist[i, j] is the distance from i to j.
        
    Returns:
        The index of the next city to visit.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        # Should not happen in a valid TSP construction, but safe to return -1 or raise
        return -1
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    if n_unvisited == 2:
        # Exact optimal choice for the last two nodes
        c1, c2 = unvisited_nodes[0], unvisited_nodes[1]
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c1, c2]
        cost2 = distance_matrix[current_node, c2] + distance_matrix[c2, c1]
        return int(c1 if cost1 < cost2 else c2)

    # General case: 2-step lookahead
    # We want to minimize d(current, c) + min(d(c, u) for u in unvisited \ {c})
    
    # Optimization: Precompute the minimum outgoing distance for each unvisited node 
    # to the other unvisited nodes.
    # Let M[c] = min(d[c, u]) for u in unvisited, u != c.
    
    # Since n is small (<= 49 here), we can just compute it on the fly for each candidate
    # or precompute a matrix of min-distances among unvisited nodes.
    
    # Let's precompute the min distance from each unvisited node to any other unvisited node.
    # This takes O(k^2) time.
    
    # Extract the submatrix of distances among unvisited nodes
    # dist_sub[i, j] = distance_matrix[unvisited_nodes[i], unvisited_nodes[j]]
    # We need the min of each row, excluding the diagonal (self).
    
    # However, creating the submatrix and finding mins is O(k^2).
    # Doing it inside the loop for each candidate is also O(k^2) total if done naively.
    # Let's just do it directly for clarity and speed on small k.
    
    best_score = float('inf')
    best_node = unvisited_nodes[0]
    
    # Deadline guard: if we are taking too long, fall back to nearest neighbor
    # Although for k=50 this is microseconds, we include it per instructions.
    start_time = time.time()
    deadline = start_time + 0.05 # 50ms per step is very generous
    
    for c in unvisited_nodes:
        # Cost to go from current to c
        cost_first = distance_matrix[current_node, c]
        
        # Cost to go from c to the best next node (excluding c itself)
        # We need min(distance_matrix[c, u]) for u in unvisited, u != c
        # Create a mask or just iterate
        
        min_next_cost = float('inf')
        for u in unvisited_nodes:
            if u == c:
                continue
            d = distance_matrix[c, u]
            if d < min_next_cost:
                min_next_cost = d
                
        total_score = cost_first + min_next_cost
        
        if total_score < best_score:
            best_score = total_score
            best_node = c
            
        # Check deadline
        if time.time() > deadline:
            # Fallback to nearest neighbor for the rest of the candidates or just break
            # Since we've already checked some, let's just pick the best found so far
            # or quickly finish the loop if it's fast enough.
            # Given the low cost, breaking is safer.
            break

    return int(best_node)
# EVOLVE-BLOCK-END
