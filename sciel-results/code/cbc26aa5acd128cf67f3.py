"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: track start time once at module load
_START_TIME = time.time()
_DEADLINE = 150.0  # seconds
_SAFETY_MARGIN = 2.0  # return safe result this far before deadline

def _time_exceeded():
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY_MARGIN)

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic: 2-step lookahead regret minimization.
    
    Strategy:
    1. Filter unvisited nodes that fit in rest_capacity.
    2. If none fit, return depot (0).
    3. For each feasible node, calculate a score based on:
       a. Immediate distance from current_node.
       b. Lookahead: The minimum distance from that node to any other feasible node
          that can be visited after it (considering remaining capacity after visiting it).
       c. Regret: If a node has a high immediate distance but a very low lookahead 
          (meaning it's a "bridge" to a cluster), it might be good. 
          However, standard regret logic suggests picking the node where the 
          "cost of waiting" is highest.
          
    A simpler and often effective robust heuristic for CVRP is the "Savings" or "Insertion" 
    style, but since we are greedy, let's use a weighted score:
    Score = distance(current, i) + alpha * min_distance(i, j) for j in feasible_after_i
    
    If we pick i, we must be able to return to depot or visit another node.
    Actually, a strong heuristic is: Minimize (dist(current, i) + dist(i, depot)) 
    is too myopic.
    
    Let's try: Minimize dist(current, i) + beta * min_{j in feasible_after_i} (dist(i, j) + dist(j, depot))
    If no j exists, just dist(i, depot).
    This looks ahead one step to see where we end up.
    
    However, calculating feasible_after_i for every i is O(N^2). With N=50, this is 2500 ops per step.
    50 steps * 8 instances * 2500 = 1,000,000 ops. Very fast.
    """
    if _time_exceeded():
        # Fallback to nearest feasible to ensure termination
        feasible = [u for u in unvisited_nodes if demands[u] <= rest_capacity]
        if not feasible:
            return 0
        return int(feasible[np.argmin(distance_matrix[current_node, feasible])])

    # 1. Filter feasible nodes
    unvisited_arr = np.array(unvisited_nodes)
    if len(unvisited_arr) == 0:
        return 0
    
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return 0

    # 2. Score each feasible node
    best_node = 0
    best_score = float('inf')
    
    # Precompute distances from current to all feasible
    dists_from_current = distance_matrix[current_node, feasible_nodes]
    
    # For lookahead, we need to know which nodes are feasible after visiting a candidate.
    # Let's iterate through candidates
    for idx, node in enumerate(feasible_nodes):
        # Immediate cost
        cost_immediate = dists_from_current[idx]
        
        # Remaining capacity after visiting this node
        rem_cap_after = rest_capacity - demands[node]
        
        # Find nodes feasible after visiting this node
        # These are unvisited nodes (excluding current candidate) that fit in rem_cap_after
        # Note: We can only visit nodes that are in the original unvisited set.
        # We don't need to re-filter by "unvisited" strictly for the lookahead min distance,
        # but we should exclude the node itself.
        
        # Get demands of all unvisited nodes
        demands_unvisited = demands[unvisited_arr]
        
        # Feasible after mask: 
        # 1. Demand <= rem_cap_after
        # 2. Node is not the current candidate (index in unvisited_arr)
        
        # To be efficient, let's just check all unvisited nodes for the min distance from 'node'
        # that are feasible capacity-wise.
        
        if rem_cap_after < 0:
            continue # Should not happen due to filter
            
        # Get indices of unvisited nodes that are feasible after visiting 'node'
        # We need to map back to unvisited_arr indices to avoid checking visited nodes (though they are in unvisited_arr)
        
        # Simple approach:
        # For each unvisited node j, if j != node and demands[j] <= rem_cap_after, it's a candidate for next.
        
        # Vectorized check:
        # demands_unvisited is an array of demands for unvisited_arr
        # We need to exclude the specific node 'node'.
        
        # Create a mask for unvisited_arr where demand fits
        fit_mask = demands_unvisited <= rem_cap_after
        
        # Exclude the current node 'node'
        # We need to find the index of 'node' in unvisited_arr to set it to False
        # This is O(N) per node, making total O(N^2). Acceptable for N=50.
        
        # Alternative: Just compute min distance from 'node' to all unvisited nodes that fit, 
        # but we must ensure we don't pick 'node' itself if it fits again (it won't, demand > 0 usually, 
        # but if demand is 0? Problem says demands 1-9, so node itself won't fit in rem_cap_after 
        # if demands[node] > 0, which is true).
        
        # So we can just take min of distance_matrix[node, unvisited_arr[fit_mask]]
        
        if np.any(fit_mask):
            next_candidates = unvisited_arr[fit_mask]
            # Ensure 'node' is not in next_candidates. 
            # Since demands[node] >= 1 and rem_cap_after = rest_cap - demands[node],
            # demands[node] <= rem_cap_after implies demands[node] <= rest_cap - demands[node] 
            # => 2*demands[node] <= rest_cap. 
            # It is possible for a node to be feasible again if its demand is small and capacity is large.
            # So we must explicitly exclude it.
            
            # Filter next_candidates to exclude 'node'
            next_candidates = next_candidates[next_candidates != node]
            
            if len(next_candidates) > 0:
                # Lookahead cost: min distance from node to any next_candidate + estimated return?
                # Let's use: dist(node, next_best) + dist(next_best, depot)
                # This approximates the completion of the route segment.
                dists_to_next = distance_matrix[node, next_candidates]
                min_idx = np.argmin(dists_to_next)
                best_next = next_candidates[min_idx]
                lookahead_cost = dists_to_next[min_idx] + distance_matrix[best_next, depot]
            else:
                # No further nodes can be visited, so we must return to depot
                lookahead_cost = distance_matrix[node, depot]
        else:
            # No further nodes fit
            lookahead_cost = distance_matrix[node, depot]

        # Total score: Immediate + Lookahead
        # We can weight them. 1.0 and 1.0 is a good start.
        # Sometimes "regret" is better: 
        # Regret = (Cost if we wait) - (Cost if we go now)
        # But simple sum often works well for routing.
        
        score = cost_immediate + lookahead_cost
        
        if score < best_score:
            best_score = score
            best_node = node

    return int(best_node)
# EVOLVE-BLOCK-END
