import numpy as np
import time

def _run_aco(prizes, distance_matrix, max_len, heuristic, num_ants=20, num_iters=30, seed=0):
    """
    Runs the ACO simulation to evaluate the heuristic.
    """
    n = len(prizes)
    rng = np.random.default_rng(seed)
    
    eta = heuristic.copy()
    # Ensure positive values as per task description
    eta[eta <= 0] = 1e-9
    
    tau = np.ones((n, n))
    
    d = distance_matrix
    d_j0 = d[0, :] 
    # Precompute cost matrix for validity check: d[i, j] + d[j, 0]
    cost = d + d_j0[None, :]
    
    best_global = 0.0
    
    for _ in range(num_iters):
        best_iter = 0.0
        
        for _ in range(num_ants):
            cur = 0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            length = 0.0
            prize = 0.0
            
            while True:
                remaining = max_len - length
                # Valid nodes j: not visited, and length + d[cur, j] + d[j, 0] <= max_len
                valid_mask = (~visited) & (cost[cur, :] <= remaining)
                valid_nodes = np.where(valid_mask)[0]
                
                if len(valid_nodes) == 0:
                    break
                
                w = tau[cur, valid_nodes] * eta[cur, valid_nodes]
                sum_w = w.sum()
                if sum_w < 1e-10:
                    break
                p = w / sum_w
                
                next_node = rng.choice(valid_nodes, p=p)
                
                visited[next_node] = True
                length += d[cur, next_node]
                prize += prizes[next_node]
                cur = next_node
            
            if prize > best_iter:
                best_iter = prize
        
        if best_iter > best_global:
            best_global = best_iter
            
        # Pheromone update
        tau = 0.9 * tau + (best_iter / 10.0)
        
    return best_global

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    # Hard deadline of 280s to be safe within 300s limit
    deadline = start_time + 280.0
    
    n = len(prizes)
    d = distance_matrix
    p = prizes
    eps = 1e-9
    
    d_j0 = d[0, :]
    
    # Strategy: Greedy Density / Value-per-Cost Heuristic
    # The orienteering problem with a small max_len (3) is tight.
    # A node is attractive if it offers high prize relative to the "cost" of including it.
    # The cost of going from i to j and then being able to return to 0 is C[i, j] = d[i, j] + d[j, 0].
    # However, the "budget" consumed is just d[i, j] for the step, but the feasibility
    # depends on d[i, j] + d[j, 0] <= remaining_budget.
    # A good static heuristic should favor nodes that are "cheap" to visit in terms of
    # the total path extension they cause.
    
    # Let's define the heuristic H[i, j] based on the "density" of node j: p[j] / (d[i, j] + d[j, 0]).
    # This is a classic knapsack-style density.
    # To make it more robust, we can also consider the "regret" of not picking a high density node.
    
    # Precompute total cost matrix C
    C = d + d_j0[None, :]
    
    # Base heuristic: Prize / Cost
    # Avoid division by zero (C[i,i] might be small, but we don't visit self usually)
    # Also, if C is very small, density is huge.
    
    H_density = np.zeros((n, n))
    for i in range(n):
        # Mask where C[i, j] is effectively zero to avoid inf
        valid_cost = C[i, :] > 1e-6
        H_density[i, valid_cost] = p[valid_cost] / C[i, valid_cost]
        # For invalid costs (shouldn't happen for distinct points, but safe), set to 0
        H_density[i, ~valid_cost] = 0.0
        
    # Ensure positive
    H_density[H_density <= 0] = eps
    
    # Alternative 2: Pure Prize Heuristic (Baseline)
    H_prize = np.tile(p[None, :], (n, 1))
    H_prize[H_prize <= 0] = eps
    
    # Alternative 3: Inverse Distance Heuristic
    H_dist = np.zeros((n, n))
    valid_d = d > 1e-6
    H_dist[valid_d] = 1.0 / d[valid_d]
    H_dist[~valid_d] = 0.0
    # Scale by prize to make it meaningful? No, let's just use 1/d.
    # Or p[j] / d[i,j]? That's similar to density but without the return cost.
    H_dist_prize = np.zeros((n, n))
    H_dist_prize[valid_d] = p[None, :][valid_d] / d[valid_d]
    H_dist_prize[~valid_d] = 0.0
    H_dist_prize[H_dist_prize <= 0] = eps

    # Candidate list
    candidates = [
        H_density,
        H_prize,
        H_dist_prize,
        # Blend of density and prize
        H_density / (H_density.max() + eps) * H_prize / (H_prize.max() + eps)
    ]
    
    best_score = -1
    best_heuristic = candidates[0]
    
    # Evaluate candidates with a quick ACO run
    # Use fewer ants/iterations for speed, but enough to distinguish quality
    # 20 ants, 30 iters is the real deal, but we have 3 instances.
    # Total time limit 300s.
    # Let's use 10 ants, 10 iters for screening.
    
    for idx, cand in enumerate(candidates):
        if time.time() > deadline - 20:
            break
            
        cand_clean = cand.copy()
        cand_clean[cand_clean <= 0] = eps
        
        try:
            score = _run_aco(p, d, max_len, cand_clean, num_ants=10, num_iters=10, seed=0)
            if score > best_score:
                best_score = score
                best_heuristic = cand_clean
        except Exception:
            continue
            
    # If we have time, we might try to refine the best one with a full run?
    # No, the heuristic is static. The ACO result is just a proxy for "how well does this guide the ants".
    # The actual scoring uses the full ACO.
    
    # Final sanity check
    if not np.all(np.isfinite(best_heuristic)) or best_heuristic.shape != (n, n):
        # Fallback to simple prize heuristic
        best_heuristic = H_prize
        best_heuristic[best_heuristic <= 0] = eps
        
    return best_heuristic
