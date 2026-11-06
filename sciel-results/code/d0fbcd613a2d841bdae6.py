"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: 150s limit, stop at 145s
    if time.time() - _start_time > 145:
        return int(depot)
    
    if len(unvisited_nodes) == 0:
        return int(depot)
        
    unvisited_arr = np.array(unvisited_nodes, dtype=int)
    n_unvisited = len(unvisited_arr)
    
    # Filter feasible nodes by capacity
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return int(depot)
        
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])
        
    # Regret-2 heuristic: for each feasible node, compute the cost of visiting it
    # now vs. the second-best alternative. The node with the highest regret is most
    # "urgent" to visit in its current position.
    
    # Pre-compute distances from current to all unvisited nodes
    d_current_all = distance_matrix[current_node, unvisited_arr]
    
    # Pre-compute distances from depot to all unvisited nodes
    d_depot_all = distance_matrix[depot, unvisited_arr]
    
    # Map from node index to its position in unvisited_arr
    node_to_idx = {int(node): i for i, node in enumerate(unvisited_arr)}
    
    # For each feasible node, compute the "cost" of visiting it now
    # Cost = distance from current to node + distance from node to best next stop
    # We'll compute this for all feasible nodes, then find regret-2
    
    n_feasible = len(feasible_nodes)
    
    # Compute cost for each feasible node
    costs = np.zeros(n_feasible)
    
    for i, j in enumerate(feasible_nodes):
        d_ij = d_current_all[node_to_idx[j]]
        
        # Find the best next stop after visiting j
        remaining_cap = rest_capacity - demands[j]
        
        # Nodes that can be visited after j: in unvisited, not j, demand <= remaining_cap
        cap_mask = demands[unvisited_arr] <= remaining_cap
        not_j_mask = unvisited_arr != j
        next_feasible_mask = cap_mask & not_j_mask
        next_feasible_nodes = unvisited_arr[next_feasible_mask]
        
        if len(next_feasible_nodes) > 0:
            d_next = np.min(distance_matrix[j, next_feasible_nodes])
        else:
            d_next = distance_matrix[j, depot]
        
        # Also consider going to depot
        d_to_depot = distance_matrix[j, depot]
        
        # The "cost" of visiting j now is the distance to j plus the minimum
        # additional distance to complete this part of the route
        costs[i] = d_ij + min(d_next, d_to_depot)
    
    # For regret-2: for each feasible node j, find the two smallest costs among
    # all feasible nodes that could be visited "instead" of j.
    # The regret is the difference between the best and second-best alternatives.
    
    # Actually, let me reconsider. The standard regret-2 approach is:
    # For each unvisited node j, consider it as the next node to visit.
    # Compute the cost of the route if j is visited next.
    # The regret of j is the difference between the minimum cost and the second minimum cost
    # when j is excluded from consideration... 
    # 
    # Let me use a simpler interpretation: 
    # For each feasible node j, compute its "immediate cost" = d(current, j)
    # Then for each j, the regret is how much worse the second-best option is compared to the best.
    # We want to pick the node with the highest regret.
    
    # Simpler regret-2: 
    # 1. Compute cost(j) = d(current, j) + min_next_distance(j) for each feasible j
    # 2. Sort by cost
    # 3. The regret of the best node is cost(second_best) - cost(best)
    # 4. But we want to pick the node whose absence would cause the most damage
    #    i.e., for each node j, compute what the best cost would be if j were not available
    #    The regret of j = best_cost_without_j - best_cost_with_j
    #    Pick the j with maximum regret
    
    # This is O(n^2) which is fine for 50 nodes
    
    # Compute cost for each feasible node
    # cost[j] = d(current, j) + min over k != j (feasible after j) of d(j, k) or d(j, depot)
    
    # Let me compute a cost matrix approach
    # For each pair of feasible nodes (j, k), compute d(j, k)
    # This allows us to compute min_next_distance(j) efficiently
    
    # Build index mapping for feasible nodes
    feasible_set = set(int(x) for x in feasible_nodes)
    
    # For each feasible node j, compute the minimum distance to any other feasible node
    # that can be visited after j (capacity constraint)
    
    min_next_dists = np.zeros(n_feasible)
    
    for i, j in enumerate(feasible_nodes):
        remaining_cap = rest_capacity - demands[j]
        
        # Find feasible next nodes after j
        best_next = float('inf')
        
        for k_idx, k in enumerate(feasible_nodes):
            if k == j:
                continue
            if demands[k] <= remaining_cap:
                d_jk = distance_matrix[j, k]
                if d_jk < best_next:
                    best_next = d_jk
        
        # Also consider going to depot
        d_j_depot = distance_matrix[j, depot]
        
        min_next_dists[i] = min(best_next, d_j_depot)
    
    # Now compute cost for each feasible node
    costs = np.zeros(n_feasible)
    for i, j in enumerate(feasible_nodes):
        d_ij = distance_matrix[current_node, j]
        costs[i] = d_ij + min_next_dists[i]
    
    # Regret-2: for each feasible node j, compute the best cost if j were excluded
    # The regret of j = min cost among other nodes - min cost overall
    # We want to pick the node with the maximum regret
    
    # Get the overall minimum cost and its index
    min_cost = np.min(costs)
    min_cost_idx = np.argmin(costs)
    
    # For each node, compute the best cost excluding that node
    regrets = np.zeros(n_feasible)
    
    for i in range(n_feasible):
        if i == min_cost_idx:
            # If this is the min cost node, the best alternative is the second min
            # Find second min
            sorted_costs = np.sort(costs)
            regrets[i] = sorted_costs[1] - sorted_costs[0]
        else:
            # If this is not the min cost node, the best cost without it is still min_cost
            # The regret is how much worse this node is compared to the best
            # But for regret-2, we want: if we don't pick this node, what's the cost?
            # The regret should reflect the opportunity cost of not picking this node now
            # 
            # Standard regret-2: for node j, regret = second_best_cost - best_cost
            # where best_cost is the minimum cost among all nodes, and second_best_cost
            # is the minimum cost among nodes excluding j
            # 
            # So for node i (not the min), regret = min_cost_excluding_i - min_cost
            # min_cost_excluding_i = min_cost (since i is not the min)
            # So regret = 0? That doesn't seem right.
            # 
            # Let me reconsider. The regret of assigning node j to position p is:
            # regret(j, p) = second_best_cost(j) - best_cost(j)
            # where best_cost(j) is the minimum cost of assigning j to any position
            # and second_best_cost(j) is the second minimum.
            # 
            # In our case, we're deciding which node to visit next (position 1).
            # For each node j, the "cost" of visiting j next is costs[i].
            # The regret of node j is how much worse the second-best option is
            # compared to the best option, when considering j as the candidate.
            # 
            # I think the correct interpretation is:
            # For each node j, if we visit j next, the total route cost will be
            # at least costs[i]. The regret of j is the difference between the
            # best possible next node cost and the second-best, but weighted by
            # how critical j is.
            # 
            # Actually, I think I'm overcomplicating this. Let me use a simpler
            # regret-2 formulation:
            # 
            # For each feasible node j:
            #   best_cost_if_j_next = costs[i]
            #   second_best_cost_if_j_next = min over k != j of (d(current, k) + min_next_dist(k))
            #   regret(j) = second_best_cost_if_j_next - best_cost_if_j_next
            # 
            # But this is the same for all j since the second best doesn't depend on j.
            # 
            # Let me try a different approach: use the savings-based regret.
            # 
            # For each node j, compute the savings of visiting j next:
            # savings(j) = d(current, depot) + d(depot, j) - d(current, j)
            # 
            # Then use regret-2 on the savings:
            # For each j, regret(j) = second_best_savings - best_savings
            # where best_savings is the maximum savings among all nodes, and
            # second_best_savings is the maximum savings excluding j.
            # 
            # Pick the node with maximum regret.
            
            # Compute savings for each feasible node
            pass
    
    # Let me implement a cleaner regret-2:
    # For each feasible node j, compute:
    #   cost(j) = d(current, j) + min_next_dist(j)
    # 
    # The regret of node j is defined as:
    #   regret(j) = min_{k != j} cost(k) - min_all cost(all)
    # 
    # But this is 0 for all non-min nodes and positive for the min node.
    # 
    # I think the standard regret-2 in VRP is:
    # For each unvisited node j, consider the cost of visiting j next.
    # Let C1(j) be the minimum cost of the route if j is visited next.
    # Let C2(j) be the second minimum cost.
    # regret(j) = C2(j) - C1(j)
    # 
    # But in our context, we're not computing full route costs, just the immediate step.
    # 
    # Let me use a pragmatic approach: combine immediate distance and savings with
    # a regret term.
    
    # Compute savings for each feasible node
    d_current_depot = distance_matrix[current_node, depot]
    savings = np.zeros(n_feasible)
    for i, j in enumerate(feasible_nodes):
        d_depot_j = distance_matrix[depot, j]
        d_current_j = distance_matrix[current_node, j]
        savings[i] = d_current_depot + d_depot_j - d_current_j
    
    # Compute a combined score: minimize cost, maximize savings
    # score(j) = costs[i] - alpha * savings[i]
    alpha = 1.0
    scores = costs - alpha * savings
    
    # Now apply regret-2: for each node j, regret(j) = second_best_score - best_score
    # where best_score is the minimum score, and second_best_score is the minimum
    # score excluding j.
    
    min_score = np.min(scores)
    min_score_idx = np.argmin(scores)
    
    # Sort scores to get second min
    sorted_scores = np.sort(scores)
    second_min_score = sorted_scores[1]
    
    regrets = np.zeros(n_feasible)
    for i in range(n_feasible):
        if i == min_score_idx:
            regrets[i] = second_min_score - min_score
        else:
            # For non-min nodes, the best score excluding i is still min_score
            # The regret is how much worse i is compared to the best
            regrets[i] = scores[i] - min_score
    
    # Pick the node with maximum regret
    best_idx = np.argmax(regrets)
    best_node = feasible_nodes[best_idx]
    
    return int(best_node)
# EVOLVE-BLOCK-END
