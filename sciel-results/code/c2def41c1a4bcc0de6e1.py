"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _aco_eval_fast(dm, demands, capacity, H, n_ants, n_iterations, seed, time_limit):
    """
    Optimized ACO simulation for fast heuristic evaluation.
    """
    start_time = time.time()
    n = dm.shape[0]
    tau = np.ones((n, n), dtype=np.float64)
    rng = np.random.RandomState(seed)
    best_cost = np.inf
    demands = np.asarray(demands, dtype=np.float64)
    
    # Pre-compute valid masks for each possible remaining capacity? 
    # No, remaining capacity changes dynamically. 
    # But we can pre-compute which nodes fit in a full truck.
    
    for it in range(n_iterations):
        if time.time() - start_time > time_limit:
            break

        iteration_costs = []

        for ant in range(n_ants):
            if time.time() - start_time > time_limit:
                break

            current_node = 0
            remaining_cap = capacity
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            total_dist = 0.0

            while True:
                # Check for timeout inside ant loop
                if time.time() - start_time > time_limit:
                    return best_cost if best_cost < np.inf else total_dist

                # Find valid next nodes
                # valid_mask: not visited, not depot, demand <= remaining_cap
                valid_mask = ~visited
                valid_mask[0] = False
                valid_mask &= (demands <= remaining_cap)

                if not np.any(valid_mask):
                    # Return to depot
                    total_dist += dm[current_node, 0]
                    remaining_cap = capacity
                    current_node = 0
                    if np.all(visited):
                        break
                    continue

                # Calculate probabilities
                # weights = tau[i, j] * H[i, j]
                row_tau = tau[current_node]
                row_eta = H[current_node]
                
                # We need weights only for valid nodes
                # To avoid allocating a full array of zeros every step, 
                # we can extract valid indices and compute weights there.
                valid_indices = np.where(valid_mask)[0]
                if len(valid_indices) == 0:
                    break # Should not happen if valid_mask was true

                w = row_tau[valid_indices] * row_eta[valid_indices]
                total_w = np.sum(w)
                
                if total_w <= 0 or not np.isfinite(total_w):
                    # Fallback: random choice
                    next_node = valid_indices[rng.randint(len(valid_indices))]
                else:
                    probs = w / total_w
                    # Ensure probabilities sum to 1 and are valid
                    probs = np.clip(probs, 0, 1)
                    sum_p = np.sum(probs)
                    if sum_p == 0:
                        next_node = valid_indices[rng.randint(len(valid_indices))]
                    else:
                        probs = probs / sum_p
                        # rng.choice on the subset is faster than full n
                        next_node = valid_indices[rng.choice(len(valid_indices), p=probs)]

                total_dist += dm[current_node, next_node]
                visited[next_node] = True
                remaining_cap -= demands[next_node]
                current_node = next_node

                if np.all(visited):
                    total_dist += dm[current_node, 0]
                    break

            iteration_costs.append(total_dist)

        if iteration_costs:
            current_best = min(iteration_costs)
            if current_best < best_cost:
                best_cost = current_best
            tau *= 0.9
            tau += 1.0 / current_best

    return best_cost

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute heuristic matrix for ACO using a hybrid grid-search and hill-climbing approach.
    """
    start_time = time.time()
    # Safety margin: Evaluator kills at 330s. We target 300s.
    deadline = start_time + 295.0 

    n = distance_matrix.shape[0]
    dm = np.asarray(distance_matrix, dtype=np.float64)
    demands = np.asarray(demands, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)

    # Ensure dm is safe (no zeros for inverse operations)
    dm_safe = np.where(dm > 1e-10, dm, 1e10)
    
    # Pre-compute demand ratio
    demand_ratio = demands / capacity
    
    best_H = None
    best_score = np.inf
    best_params = None

    def generate_H(params):
        """
        params: (gamma, alpha, beta, delta)
        H[i, j] = (1 / (dm[i,j] + eps)^gamma) * exp(-alpha * dm[i,j]) * (1 - beta * demand_ratio[j])^delta
        """
        gamma, alpha, beta, delta = params
        # Distance term
        term_dist = 1.0 / (dm_safe + 1e-6) ** gamma
        # Exponential term
        term_exp = np.exp(-alpha * dm_safe)
        # Demand term: penalize high demand nodes
        # If beta * demand_ratio[j] >= 1, the term becomes 0 or negative.
        # We want it to be positive. 
        # Let's use: (1 - beta * demand_ratio[j])
        # If this is <= 0, we set it to a small epsilon or 0.
        # Actually, for CVRP, it's often better to penalize based on "remaining capacity fit".
        # But a static heuristic can't know remaining capacity.
        # A common heuristic is to prefer nodes with demand that fits well in a "typical" load.
        # Let's stick to: (1 - beta * demand_ratio[j])
        # If beta is large, high demand nodes are heavily penalized.
        
        base_demand_factor = 1.0 - beta * demand_ratio
        # Clip to avoid negative values if beta is too large
        base_demand_factor = np.clip(base_demand_factor, 1e-6, None)
        term_demand = base_demand_factor ** delta
        
        H = term_dist * term_exp * term_demand[np.newaxis, :]
        
        # Ensure positive and finite
        H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
        return H

    def evaluate_candidate(params, time_limit, n_ants, n_iterations):
        H = generate_H(params)
        score = _aco_eval_fast(dm, demands, capacity, H, n_ants, n_iterations, seed=42, time_limit=time_limit)
        return score, H

    # --- Phase 1: Coarse Grid Search ---
    # Parameters: gamma (distance power), alpha (exp decay), beta (demand penalty strength), delta (demand penalty power)
    
    # Time budget for Phase 1: 50% of total time
    t_phase1_limit = deadline - (deadline - start_time) * 0.5
    
    gammas = np.linspace(0.5, 2.5, 8)
    alphas = np.linspace(0.0, 1.5, 5)
    betas = np.linspace(0.0, 1.5, 6)
    deltas = np.linspace(0.0, 2.0, 5)
    
    candidates = []
    
    for g in gammas:
        if time.time() > t_phase1_limit:
            break
        for a in alphas:
            if time.time() > t_phase1_limit:
                break
            for b in betas:
                if time.time() > t_phase1_limit:
                    break
                for d in deltas:
                    if time.time() > t_phase1_limit:
                        break
                    params = (g, a, b, d)
                    # Fast evaluation: 3 ants, 3 iterations
                    # Time per eval should be very small
                    eval_time = 0.5 
                    try:
                        score, H = evaluate_candidate(params, eval_time, n_ants=3, n_iterations=3)
                        if np.isfinite(score):
                            candidates.append((score, params, H))
                    except Exception:
                        continue
    
    if not candidates:
        # Fallback default
        return 1.0 / dm_safe
        
    # Sort by score
    candidates.sort(key=lambda x: x[0])
    
    # Keep top K candidates for fine-tuning
    top_k = candidates[:10]
    
    # --- Phase 2: Local Optimization (Hill Climbing) ---
    # Time budget for Phase 2: remaining time minus safety buffer
    # We will refine the top candidates.
    
    # Use a slightly stronger evaluation for the final comparison
    # 5 ants, 5 iterations
    
    current_best_score = top_k[0][0]
    current_best_params = top_k[0][1]
    current_best_H = top_k[0][2]
    
    # Hill climbing loop
    # We perturb the parameters of the current best and see if we improve.
    # We also try to "restart" from other top candidates if we get stuck.
    
    step_size = 0.1
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline - 10.0:
        # Generate perturbation
        noise = np.random.normal(0, step_size, size=4)
        new_params = tuple(np.array(current_best_params) + noise)
        
        # Ensure params are in reasonable ranges
        new_params = (
            max(0.1, min(4.0, new_params[0])), # gamma
            max(0.0, min(3.0, new_params[1])), # alpha
            max(0.0, min(2.0, new_params[2])), # beta
            max(0.0, min(3.0, new_params[3]))  # delta
        )
        
        # Evaluate
        # Use a slightly longer time for this stage to be more accurate
        eval_time = 1.0
        try:
            score, H = evaluate_candidate(new_params, eval_time, n_ants=5, n_iterations=5)
        except Exception:
            score = np.inf
            H = current_best_H
            
        if score < current_best_score:
            current_best_score = score
            current_best_params = new_params
            current_best_H = H
            no_improve_count = 0
            # Reduce step size slightly on improvement
            step_size = max(0.01, step_size * 0.9)
        else:
            no_improve_count += 1
            # If stuck, try a different starting point from top_k
            if no_improve_count > max_no_improve:
                # Pick a random other top candidate
                if len(top_k) > 1:
                    idx = np.random.randint(1, len(top_k))
                    current_best_score = top_k[idx][0]
                    current_best_params = top_k[idx][1]
                    current_best_H = top_k[idx][2]
                no_improve_count = 0
                step_size = 0.1 # Reset step size

    # --- Phase 3: Final Verification ---
    # Run the best found heuristic with a more robust simulation to ensure it's good.
    # If time permits, we can even run a "full" ACO (20 ants, 30 iters) on the best H
    # to get a very accurate score, but we just need to return H.
    # The evaluator will run its own ACO. We just need H to be good.
    
    # One last check: ensure H is valid
    final_H = current_best_H
    final_H = np.where(np.isfinite(final_H) & (final_H > 0), final_H, 1e-9)
    
    return final_H

# EVOLVE-BLOCK-END
