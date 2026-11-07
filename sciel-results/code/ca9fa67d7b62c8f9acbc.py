"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

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
                if time.time() - start_time > time_limit:
                    return best_cost if best_cost < np.inf else total_dist

                # Find valid next nodes
                valid_mask = ~visited
                valid_mask[0] = False
                valid_mask &= (demands <= remaining_cap)

                if not np.any(valid_mask):
                    total_dist += dm[current_node, 0]
                    remaining_cap = capacity
                    current_node = 0
                    if np.all(visited):
                        break
                    continue

                valid_indices = np.where(valid_mask)[0]
                if len(valid_indices) == 0:
                    break

                w = tau[current_node, valid_indices] * H[current_node, valid_indices]
                total_w = np.sum(w)
                
                if total_w <= 0 or not np.isfinite(total_w):
                    next_node = valid_indices[rng.randint(len(valid_indices))]
                else:
                    probs = w / total_w
                    probs = np.clip(probs, 0, 1)
                    sum_p = np.sum(probs)
                    if sum_p == 0:
                        next_node = valid_indices[rng.randint(len(valid_indices))]
                    else:
                        probs = probs / sum_p
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
    Compute heuristic matrix for ACO using a fast, bounded search over demand-aware parameters.
    """
    start_time = time.time()
    deadline = start_time + 295.0 

    n = distance_matrix.shape[0]
    dm = np.asarray(distance_matrix, dtype=np.float64)
    demands = np.asarray(demands, dtype=np.float64)
    
    # Ensure dm is safe
    dm_safe = np.where(dm > 1e-10, dm, 1e10)
    
    # Strategy: eta[i,j] = d(i,j)^-gamma * (1 / (1 + demand[j]))^delta
    # We search over gamma and delta.
    
    best_H = None
    best_score = np.inf
    
    # Pre-compute demand term bases for different deltas
    # demand[j] is 1..9. 1/(1+demand[j]) is ~0.1 to 0.5
    
    # Small grid of candidates
    gammas = [1.0, 1.5, 2.0, 2.5, 3.0]
    deltas = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    candidates = []
    
    # Evaluate each candidate with a very short time limit (0.1s) and small colony
    for g in gammas:
        if time.time() > deadline - 5.0:
            break
        for d in deltas:
            if time.time() > deadline - 5.0:
                break
            
            # Generate H
            # term_dist = dm_safe^-g
            # term_demand = (1 / (1 + demands))^d
            # H[i,j] = term_dist[i,j] * term_demand[j]
            
            term_dist = 1.0 / (dm_safe ** g)
            # Avoid division by zero if demand is 0 (depot), but demand[0]=0 so 1/(1+0)=1.
            # If demand is large, this is small.
            base_demand_factor = 1.0 / (1.0 + demands)
            term_demand = base_demand_factor ** d
            
            H = term_dist * term_demand[np.newaxis, :]
            
            # Ensure positive and finite
            H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
            
            # Quick evaluation
            try:
                score = _aco_eval_fast(dm, demands, capacity, H, n_ants=5, n_iterations=5, seed=42, time_limit=0.5)
                if np.isfinite(score):
                    candidates.append((score, H))
            except Exception:
                continue
    
    if not candidates:
        # Fallback: standard inverse distance
        H = 1.0 / dm_safe
        H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
        return H
        
    # Sort by score
    candidates.sort(key=lambda x: x[0])
    
    # If we have time, refine the top 3 candidates with a slightly longer evaluation
    if time.time() < deadline - 10.0:
        top_k = candidates[:3]
        refined_candidates = []
        for score, H in top_k:
            if time.time() > deadline - 10.0:
                break
            try:
                # Longer evaluation to confirm
                score2 = _aco_eval_fast(dm, demands, capacity, H, n_ants=10, n_iterations=10, seed=42, time_limit=2.0)
                if np.isfinite(score2):
                    refined_candidates.append((score2, H))
            except Exception:
                continue
        
        if refined_candidates:
            refined_candidates.sort(key=lambda x: x[0])
            best_score, best_H = refined_candidates[0]
        else:
            best_score, best_H = candidates[0]
    else:
        best_score, best_H = candidates[0]
        
    # Final safety check
    best_H = np.where(np.isfinite(best_H) & (best_H > 0), best_H, 1e-9)
    
    return best_H

# EVOLVE-BLOCK-END
