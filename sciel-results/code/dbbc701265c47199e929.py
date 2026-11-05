"""
Strategy: Enhanced Clarke-Wright Savings with Vectorized Multi-Step Lookahead.
Refines the savings heuristic by using vectorized NumPy operations to evaluate
2-3 step lookahead costs for the top candidates, effectively smoothing the 
nonsmooth objective of greedy selection. This avoids the "dead end" trap of 
pure nearest-neighbor or simple savings where a node looks good locally but 
blocks better subsequent moves.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _select_savings_heuristic(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node based on a savings score with a vectorized lookahead.
    """
    if len(unvisited_nodes) == 0:
        return depot
    
    # 1. Filter feasible nodes
    unvisited_arr = np.array(unvisited_nodes, dtype=int)
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible_arr = unvisited_arr[feasible_mask]
    
    if len(feasible_arr) == 0:
        return depot
    
    if len(feasible_arr) == 1:
        return int(feasible_arr[0])

    # Time check: if critical, fall back to pure nearest feasible
    if _time_remaining() < 0.05:
        d = distance_matrix[current_node, feasible_arr]
        return int(feasible_arr[np.argmin(d)])

    # 2. Compute Base Savings Score
    # Saving = d(depot, current) + d(depot, n) - d(current, n)
    # We want to maximize saving (minimize incremental cost).
    d_depot_current = distance_matrix[depot, current_node]
    d_current_feasible = distance_matrix[current_node, feasible_arr]
    d_feasible_depot = distance_matrix[feasible_arr, depot]
    
    # Incremental cost if we go to depot now vs via n (simplified savings proxy)
    # Cost_via_n = d(current, n) + d(n, depot)
    # Cost_close = d(current, depot)
    # Delta = Cost_via_n - Cost_close
    # Minimize Delta.
    delta_costs = d_current_feasible + d_feasible_depot - distance_matrix[current_node, depot]
    
    # 3. Select Top K candidates for lookahead
    # If we have many nodes, restrict K to keep computation low.
    # If time is ample, we can afford larger K.
    time_rem = _time_remaining()
    if time_rem > 10.0:
        k = min(10, len(feasible_arr))
    elif time_rem > 2.0:
        k = min(5, len(feasible_arr))
    else:
        k = min(3, len(feasible_arr))
        
    # Get indices of the k best (lowest delta_costs)
    top_k_indices = np.argsort(delta_costs)[:k]
    candidates = feasible_arr[top_k_indices]
    candidate_deltas = delta_costs[top_k_indices]
    
    # 4. Vectorized Lookahead for Candidates
    # We estimate the total cost of a short path starting with each candidate.
    # Path: current -> cand -> next_best -> depot (or further if time permits)
    
    # We will evaluate a 2-step lookahead by default.
    # For each candidate i, find the best next node j (feasible after i) that minimizes d(i, j) + d(j, depot).
    # If no such j exists or is infeasible, the cost is d(i, depot).
    
    # Pre-compute d(i, feasible_arr) for all candidates? 
    # candidates is small (K <= 10), feasible_arr is up to 50.
    # Matrix: K x N_feasible
    
    # Distances from candidates to all feasible nodes
    # d_cand_feasible[i, j] = distance(candidates[i], feasible_arr[j])
    d_cand_feasible = distance_matrix[candidates, feasible_arr]
    
    # Distances from feasible nodes to depot (already have d_feasible_depot)
    
    # Demands of feasible nodes
    d_feasible = demands[feasible_arr]
    
    # Capacity check for next step
    # Remaining capacity after visiting candidate i:
    rem_cap_after_cand = rest_capacity - demands[candidates]
    
    # Initialize lookahead costs with the cost of closing the route immediately after cand
    # This serves as the baseline (1-step lookahead: current->cand->depot)
    # Total cost for this branch: d(current, cand) + d(cand, depot)
    # Note: delta_costs already is (d(current, cand) + d(cand, depot) - d(current, depot))
    # So total_path_cost = d(current, depot) + delta_costs.
    # Since d(current, depot) is constant, minimizing delta_costs is equivalent.
    # But for lookahead, we compare absolute path segments from the candidate onward.
    
    # Let's calculate the cost "from candidate onwards" for 2 steps.
    # Cost2[i] = min( d(cand[i], depot), min_{j feasible} ( d(cand[i], j) + d(j, depot) ) )
    # Subject to: demands[j] <= rem_cap_after_cand[i] AND j != cand[i] (implicit if we treat j from full list, but j must be unvisited)
    # Since feasible_arr contains all unvisited feasible nodes, and candidates are a subset,
    # we must exclude the candidate itself from the next-step options to avoid loops (though demands would likely prevent it if demands > 0, but safe to mask).
    
    # Matrix for next-step costs: K x N_feasible
    # cost_next[i, j] = d(cand[i], j) + d(j, depot)
    cost_next = d_cand_feasible + d_feasible_depot[None, :]
    
    # Mask out infeasible next nodes
    # For each candidate i, j is valid if demands[j] <= rem_cap_after_cand[i]
    # rem_cap_after_cand is shape (K,)
    # d_feasible is shape (N,)
    # valid_next_mask[i, j] = d_feasible[j] <= rem_cap_after_cand[i]
    valid_next_mask = d_feasible[None, :] <= rem_cap_after_cand[:, None]
    
    # Also, we should not pick the candidate itself as the next node?
    # Actually, the candidate is already visited. So we should mask it out.
    # candidates is a subset of feasible_arr.
    # Create a mask where col j is False if feasible_arr[j] is in candidates.
    # This is tricky with sets. Simpler: just allow it? If demands[cand] > rem_cap_after_cand, it's masked.
    # rem_cap_after_cand = rest_cap - d[cand]. So d[cand] <= rest_cap - d[cand] => 2*d[cand] <= rest_cap.
    # If 2*d[cand] > rest_cap, it's masked. If not, it's allowed.
    # Visiting the same node twice is bad. We should mask the candidate itself.
    
    # Map candidates to their indices in feasible_arr to mask them
    # Since candidates are a subset of feasible_arr, we can find their positions.
    # This is a bit expensive to do generally, but K is small.
    # Alternative: Just accept the risk? In VRP, demands are positive. 
    # If we pick j=i, cost is d(i,i) + d(i,depot) = d(i,depot).
    # This is the same as closing the route. So it's a safe "fallback" in the min.
    # It effectively allows the "close now" option in the min calculation.
    # So we don't strictly need to mask j=i if we interpret the min as "best next move OR close".
    # Wait, the min is over j in feasible_arr. If j=i is allowed, it computes d(i,i)+d(i,depot) = d(i,depot).
    # The cost of closing is d(i, depot). So it correctly models "close the route".
    # So we just need to ensure we don't pick a node that doesn't fit. The mask handles capacity.
    
    # Apply mask: set invalid costs to inf
    cost_next_masked = np.where(valid_next_mask, cost_next, np.inf)
    
    # The best next-step cost for each candidate is the minimum over all j
    min_next_cost = np.min(cost_next_masked, axis=1)
    
    # If min is inf (no valid next node), it means we MUST close the route?
    # No, if no j is valid, it means capacity is exhausted after cand.
    # Then the route must end. Cost is d(cand, depot).
    # Our formula: min_next_cost would be inf.
    # We should replace inf with d(cand, depot).
    d_cand_depot = distance_matrix[candidates, depot]
    min_next_cost = np.where(np.isinf(min_next_cost), d_cand_depot, min_next_cost)
    
    # Total score for each candidate:
    # We want to minimize: d(current, cand) + min_next_cost
    # d(current, cand) is d_current_feasible[top_k_indices]
    
    total_scores = d_current_feasible[top_k_indices] + min_next_cost
    
    # Pick the candidate with the lowest total score
    best_idx_in_k = np.argmin(total_scores)
    best_node = candidates[best_idx_in_k]
    
    return int(best_node)

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Main entry point.
    """
    if _time_remaining() < 0.05:
        # Fallback to fast nearest feasible
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return depot
        if len(feasible) == 1:
            return int(feasible[0])
        d = distance_matrix[current_node, feasible]
        return int(feasible[np.argmin(d)])

    if len(unvisited_nodes) == 0:
        return depot

    return _select_savings_heuristic(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix)

# EVOLVE-BLOCK-END
