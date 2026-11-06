import numpy as np
import time

# EVOLVE-BLOCK-START
_start_time = time.time()
_TIME_LIMIT = 140.0  # Leave 10s margin for overhead/termination

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT:
        return depot

    if len(unvisited_nodes) == 0:
        return depot

    unvisited = np.array(unvisited_nodes)
    unvisited_demands = demands[unvisited]
    
    # Feasibility mask: demand must fit in remaining capacity
    feasible_mask = unvisited_demands <= rest_capacity
    
    if not np.any(feasible_mask):
        return depot

    feasible_indices = np.where(feasible_mask)[0]
    feasible_nodes = unvisited[feasible_indices]
    
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])

    # Lookahead Heuristic:
    # For each candidate j, estimate the cost of the next step after j.
    # This avoids picking a node that is close to current but far from all other nodes/depot.
    
    scores = np.zeros(len(feasible_nodes))
    
    # Precompute distance from current to all feasible nodes
    dist_from_current = distance_matrix[current_node, feasible_nodes]
    
    for idx, j in enumerate(feasible_nodes):
        # Immediate cost
        cost_immediate = dist_from_current[idx]
        
        # Lookahead cost:
        # We want to know the best move from j to another feasible node k, or back to depot.
        # To save time, we can approximate the "best next" by checking the closest feasible node to j.
        # However, we must ensure k is feasible (demand fits) and k != j.
        
        # Get distances from j to all other feasible nodes
        # Note: feasible_nodes contains j. We need distances to feasible_nodes excluding j? 
        # Actually, we can just take min distance to any other node in feasible_nodes.
        
        # Optimization: Instead of looping over all k for each j (O(M^2)), 
        # we can just pick the nearest neighbor of j among the feasible set.
        
        # Distance from j to all feasible nodes
        dist_j_to_feasible = distance_matrix[j, feasible_nodes]
        
        # Mask out j itself (index idx)
        # We want the min distance to any other node in feasible_nodes
        # If there are no other nodes, the min will be distance to self (0) or we handle it.
        
        if len(feasible_nodes) > 1:
            # Create a copy to ignore self-distance
            # We can't easily mask in numpy slice without copy, but M is small (<=50)
            # Simple loop for inner min is fast enough for M=50
            min_next_dist = np.inf
            # We can use np.argmin on a masked array or just a small python loop
            # Given M is small, a python loop over M items for each of M items is O(M^2) = 2500 ops, trivial.
            for k_idx, k_node in enumerate(feasible_nodes):
                if k_idx == idx:
                    continue
                d = dist_j_to_feasible[k_idx]
                if d < min_next_dist:
                    min_next_dist = d
            
            # If no other node was found (shouldn't happen if len>1), min_next_dist remains inf
            # But we also consider returning to depot if the route ends.
            # A common heuristic is to compare the "continuation" cost vs "return" cost.
            # If the next node is very far, maybe returning to depot is better?
            # For simplicity and robustness, we use the min of (nearest neighbor, distance to depot).
            # Actually, if we pick j, we don't *have* to go to the nearest neighbor next. 
            # But the "savings" logic suggests that connecting to a nearby node saves the depot trip.
            # So, a good surrogate for "future cost" is the distance to the nearest other feasible node.
            # If that distance is very large, the effective cost is high.
            
            # Let's refine: The cost of adding j is d(current,j).
            # The benefit is that we don't have to go depot->j->depot.
            # A simpler, very effective heuristic is:
            # Score = d(current, j) + 0.5 * d(j, depot)
            # This balances proximity to current and proximity to depot.
            
            # Let's try the Lookahead Score: d(current, j) + min(d(j, k) for k in feasible, k!=j)
            # This encourages picking nodes that are part of a dense cluster.
            
            # If min_next_dist is inf (no other nodes), it means j is the last one, so cost is d(j, depot).
            if min_next_dist == np.inf:
                future_cost = distance_matrix[j, depot]
            else:
                # It might be better to return to depot if the next node is far away.
                # Let's use min(min_next_dist, distance_matrix[j, depot])? 
                # No, if we return to depot, the route closes. 
                # If we go to k, the route continues.
                # The "Savings" heuristic essentially says: go to k if d(i,j) + d(j,k) < d(i,depot) + d(depot,k) ... 
                # This is getting complex for a single step.
                
                # Let's stick to a weighted sum which is robust:
                # Score = d(current, j) + alpha * d(j, depot)
                # With alpha = 0.5.
                # This is simpler and often performs similarly to complex lookaheads.
                
                # However, the prompt mentions "Savings heuristic prioritizes nodes that reduce total tour length".
                # The standard savings for connecting i->j->k instead of i->depot and depot->k is:
                # s(i,k) = d(i,depot) + d(depot,k) - d(i,k)
                # Here we are at i (current). We are choosing j.
                # If we choose j, we effectively form edge (current, j).
                # The "savings" of this edge compared to closing the route and starting new is:
                # saved = d(current, depot) + d(depot, j) - d(current, j)
                # We want to MAXIMIZE savings, which is MINIMIZING (d(current, j) - d(current, depot) - d(depot, j)).
                # This simplifies to minimizing: d(current, j) - d(depot, j)  (since d(current, depot) is constant for this step).
                
                # So, Score = d(current, j) - d(j, depot)
                # Let's test this logic.
                # If j is very close to current, d(current, j) is small.
                # If j is close to depot, d(j, depot) is small, so -d(j, depot) is large (negative).
                # Wait, maximizing savings means we want high (d(current,depot) + d(depot,j) - d(current,j)).
                # Constant C = d(current,depot).
                # Maximize C + d(depot,j) - d(current,j).
                # Minimize d(current,j) - d(depot,j).
                
                # Let's use this "Savings-like" score.
                pass 

        # Let's implement the Score = d(current, j) - d(j, depot)
        # This directly implements the Clarke-Wright savings principle for the single edge (current, j).
        
        score = dist_from_current[idx] - distance_matrix[j, depot]
        scores[idx] = score

    # Select the node with the minimum score (maximum savings)
    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
