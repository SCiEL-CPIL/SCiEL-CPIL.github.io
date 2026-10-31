"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
TIME_LIMIT = 110.0

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a 2-Step Lookahead with Geometric Consistency.
    
    Strategy:
    1. For each candidate next node, identify the best subsequent node (2nd step).
    2. Calculate the total path length for the 2 steps.
    3. Apply a geometric penalty for sharp turns (deviation from straight line) 
       at the candidate node, promoting smoother tours.
    4. Combine with a bias towards the destination to ensure progress.
    """
    # Time check
    if time.time() - _START_TIME > TIME_LIMIT:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # Distances from current to all candidates
    d1 = distance_matrix[current_node, unvisited_nodes]
    
    # We need to evaluate 2-step paths: current -> u -> v
    # where u is in unvisited_nodes, v is in unvisited_nodes \ {u}
    
    # Optimization: 
    # Instead of O(N^2) full evaluation which might be slow for 50 cities if called 50 times 
    # (50 * 50^2 = 125k ops, actually quite fast in numpy), we can vectorize.
    # However, finding the "best v" for each "u" requires excluding u.
    
    # Let's use a simpler but effective heuristic that approximates lookahead:
    # Cost(u) = d(current, u) + min_{v != u} [ d(u, v) + beta * d(v, dest) ]
    # Plus a geometric penalty for the turn at u.
    
    # To keep it fast and robust, we will compute a "forward cost" for each candidate u.
    # The forward cost considers the best immediate successor v.
    
    # Step 1: Compute d(u, v) matrix for unvisited nodes
    # U is array of indices
    U = unvisited_nodes
    n = len(U)
    
    # Sub-matrix of distances between unvisited nodes
    # D_uu[i, j] = dist(U[i], U[j])
    D_uu = distance_matrix[np.ix_(U, U)]
    
    # Distances from each unvisited node to destination
    d_u_dest = distance_matrix[U, destination_node]
    
    # For each candidate u (index i in U), we want to find the best v (index j in U, j!=i)
    # to minimize: d(u, v) + lambda * d(v, dest)
    # Let's set lambda = 0.5 (moderate bias to destination)
    lambda_dest = 0.5
    
    # Cost matrix for the 2nd step: C2[i, j] = D_uu[i, j] + lambda_dest * d_u_dest[j]
    C2 = D_uu + lambda_dest * d_u_dest[np.newaxis, :]
    
    # We need the min over j != i.
    # We can set the diagonal to infinity, then take argmin.
    np.fill_diagonal(C2, np.inf)
    
    # Best 2nd step cost for each u
    min_cost_2nd_step = np.min(C2, axis=1)
    best_next_idx = np.argmin(C2, axis=1)
    
    # Now, compute the total 2-step path length for each u:
    # PathLen[i] = d1[i] + min_cost_2nd_step[i]
    # But we also want to penalize sharp turns.
    # Turn angle at u between vector (current->u) and (u->v).
    # This is hard to vectorize easily without coordinate extraction.
    # Since we only have distance_matrix, we can approximate "turn quality" 
    # using the triangle inequality deviation or simply rely on the 2-step cost.
    
    # Alternative Geometric Insight:
    # A straight line is the shortest path. If u is far from the direct line current->dest,
    # the 2-step cost will naturally be higher.
    # Let's add a small penalty for "wasted distance" relative to the direct path.
    
    d_current_to_dest = distance_matrix[current_node, destination_node]
    
    # If we are close to destination, just go nearest.
    if d_current_to_dest < 1e-9:
        return int(U[np.argmin(d1)])
    
    # Heuristic Score:
    # Score[i] = d1[i] + min_cost_2nd_step[i] + gamma * |d1[i] + d_u_dest[i] - d_current_to_dest|
    # The last term penalizes deviation from the triangle equality (i.e., how much longer the path is than direct).
    # Actually, a simpler effective heuristic for TSP is "Nearest Neighbor with Backtracking Penalty".
    
    # Let's try a different structural approach: 
    # **Savings Heuristic Approximation**
    # Instead of pure lookahead, we can look at the "savings" of inserting u into the path.
    # But that requires knowing the previous node, which we don't have in this signature.
    
    # Let's stick to the 2-step lookahead but refine the cost function.
    # The term min_cost_2nd_step already includes d(u,v) + lambda*d(v,dest).
    # So Total Cost = d(current, u) + d(u, best_v) + lambda * d(best_v, dest).
    
    total_cost = d1 + min_cost_2nd_step
    
    # Add a geometric consistency term:
    # We want to avoid zig-zags. 
    # A proxy for zig-zag: if d(u, dest) is much larger than d(current, dest), we are moving away.
    # Penalty = max(0, d_u_dest - d_current_to_dest)
    penalty = np.maximum(0, d_u_dest - d_current_to_dest)
    
    # Combine
    alpha = 0.3
    final_scores = total_cost + alpha * penalty
    
    min_idx = np.argmin(final_scores)
    
    return int(U[min_idx])
# EVOLVE-BLOCK-END
