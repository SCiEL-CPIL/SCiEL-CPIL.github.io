import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 148.0
_SAFETY_MARGIN = 2.0

def _time_exceeded():
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY_MARGIN)

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    if _time_exceeded():
        return depot

    if len(unvisited_nodes) == 0:
        return depot

    # 1. Filter feasible nodes
    # demands is a list/array of length N+1
    # unvisited_nodes is a list/array of indices
    feasible_mask = np.array([demands[u] <= rest_capacity for u in unvisited_nodes])
    feasible_nodes = unvisited_nodes[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot

    # If only one feasible, return it
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])

    # 2. Score each feasible candidate
    # Score = dist(current, j) + alpha * min(dist(j, k) for k in feasible_after_j)
    # If no k feasible, min(dist(j, k)) is dist(j, depot)
    
    alpha = 1.0 # Weight for lookahead
    
    best_score = float('inf')
    best_node = feasible_nodes[0]
    
    # Convert to numpy arrays for speed if needed, but loops are fine for N=50
    # Let's use a loop for clarity and potential early exit
    
    for j in feasible_nodes:
        d1 = distance_matrix[current_node, j]
        
        # Calculate min cost to go from j to any other feasible node or depot
        # New capacity after visiting j
        new_cap = rest_capacity - demands[j]
        
        # Find min distance from j to any node in unvisited_nodes that is feasible with new_cap
        # OR to depot
        min_next_dist = distance_matrix[j, depot] # Default is depot
        
        # Check other unvisited nodes
        # We can vectorize this part
        # Candidates for next: unvisited_nodes where demands[k] <= new_cap
        # Note: j itself is in unvisited_nodes, but we are moving FROM j, so we exclude j? 
        # Actually, k should be different from j.
        
        # Get distances from j to all unvisited nodes
        dists_to_unvisited = distance_matrix[j, unvisited_nodes]
        
        # Mask for feasible next nodes (excluding j itself is handled by logic or just allowing it? 
        # If we pick j, we remove it. So k != j.
        # But demands[j] <= new_cap is true. So j is a candidate. 
        # However, we are already at j. We need to go to a *different* node.
        # So we should exclude j from the "next" candidates.
        
        # Let's find the index of j in unvisited_nodes to exclude it
        # This is O(N) per j, so O(N^2) total.
        
        # Simpler: Just compute min over all unvisited nodes that are feasible, 
        # but if the min is j itself, that's a problem. 
        # Actually, if we pick j, j is no longer unvisited. 
        # So the set of "next" nodes is (unvisited_nodes \ {j}) filtered by capacity.
        
        # Let's construct the mask for next nodes
        # demands[k] <= new_cap
        # k != j
        
        # Vectorized approach:
        # feasible_next_mask = (demands[unvisited_nodes] <= new_cap)
        # But we must exclude j.
        
        # Let's just iterate unvisited_nodes for the inner loop? 
        # N=50. Inner loop 50. Outer loop 50. 2500 ops. Fine.
        
        min_next = distance_matrix[j, depot]
        
        for k in unvisited_nodes:
            if k == j:
                continue
            if demands[k] <= new_cap:
                dk = distance_matrix[j, k]
                if dk < min_next:
                    min_next = dk
        
        score = d1 + alpha * min_next
        
        if score < best_score:
            best_score = score
            best_node = j

    return int(best_node)
