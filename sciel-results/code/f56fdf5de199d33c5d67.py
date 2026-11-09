import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes the heuristic matrix for the ant colony optimization.
    
    Uses a budget-aware expected utility model:
    eta_ij = prize_j / (d_ij + d_j0)^1.5 * exp(-lambda * (d_ij + d_j0) / max_len)
    
    This explicitly penalizes moves that consume a disproportionate fraction
    of the remaining length budget, preventing ants from getting stuck far
    from the depot with insufficient range to return.
    
    A parametric grid search over lambda and the exponent is performed,
    evaluated via a lightweight ACO simulation.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = distance_matrix.shape[0]
    if n < 2:
        return np.ones((n, n), dtype=np.float64)
    
    start_time = time.time()
    deadline = start_time + 320.0  # safety margin before 330s limit
    
    # Precompute useful arrays
    d = distance_matrix.copy()
    d_to_depot = d[:, 0].copy()
    
    # Avoid division by zero
    eps = 1e-9
    d_safe = d + eps
    d_to_depot_safe = d_to_depot + eps
    
    # Total route cost if going from i to j and then back to depot
    # This is the "budget consumption" for that move
    total_cost = d_safe + d_to_depot_safe[None, :]  # shape (n, n), total_cost[i,j] = d[i,j] + d[j,0]
    
    best_score = -1
    best_eta = None
    
    # Grid search over lambda (exponential penalty strength) and exponent
    lambda_vals = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    exp_vals = [1.0, 1.5, 2.0, 2.5]
    
    # Also include the old parametric form for comparison
    a_vals = [0.5, 1.0, 1.5, 2.0]
    b_vals = [0.5, 1.0, 1.5, 2.0]
    c_vals = [0.5, 1.0, 1.5, 2.0]
    
    def evaluate_heuristic(eta, n_ants=8, n_iter=8):
        """Quick ACO simulation to estimate heuristic quality."""
        if time.time() > deadline:
            return -1
        
        pheromone = np.ones((n, n))
        best_total = 0
        
        for _ in range(n_iter):
            iter_best = 0
            for _ in range(n_ants):
                visited = set([0])
                current = 0
                length = 0.0
                total_prize = prizes[0]
                
                while True:
                    # Find candidate nodes
                    candidates = []
                    for j in range(1, n):
                        if j in visited:
                            continue
                        new_length = length + d[current, j] + d[j, 0]
                        if new_length <= max_len:
                            candidates.append(j)
                    
                    if not candidates:
                        break
                    
                    # Compute probabilities
                    probs = np.zeros(len(candidates))
                    for idx, j in enumerate(candidates):
                        probs[idx] = pheromone[current, j] * eta[current, j]
                    
                    total_prob = probs.sum()
                    if total_prob < 1e-15:
                        probs = np.ones(len(candidates))
                        total_prob = len(candidates)
                    else:
                        probs /= total_prob
                    
                    # Sample
                    r = np.random.random()
                    cumsum = 0
                    chosen_idx = len(candidates) - 1
                    for idx in range(len(candidates)):
                        cumsum += probs[idx]
                        if r <= cumsum:
                            chosen_idx = idx
                            break
                    
                    j = candidates[chosen_idx]
                    length += d[current, j]
                    total_prize += prizes[j]
                    visited.add(j)
                    current = j
                
                if total_prize > iter_best:
                    iter_best = total_prize
                
                # Update pheromone
                pheromone *= 0.9
                pheromone += iter_best / 10.0
            
            best_total = max(best_total, iter_best)
        
        return best_total
    
    # Generate candidate heuristics
    candidates = []
    
    # Budget-aware expected utility form:
    # eta_ij = prize_j / (d_ij + d_j0)^exp * exp(-lambda * (d_ij + d_j0) / max_len)
    for lam in lambda_vals:
        for exp_val in exp_vals:
            # Avoid division by zero and negative values
            cost_safe = np.maximum(total_cost, eps)
            base = prizes[None, :] / (cost_safe ** exp_val)
            # Exponential penalty: penalize moves that consume too much of the budget
            penalty = np.exp(-lam * total_cost / max_len)
            eta = base * penalty
            eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
            candidates.append(eta)
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Also try the old parametric form for comparison
    for a in a_vals:
        for b in b_vals:
            for c in c_vals:
                node_part = (prizes + eps) ** a / (d_to_depot_safe ** c)
                eta = (node_part[None, :] / (d_safe ** b))
                eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
                candidates.append(eta)
                
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Evaluate each candidate
    for idx, eta in enumerate(candidates):
        if time.time() > deadline:
            break
        
        score = evaluate_heuristic(eta)
        if score > best_score:
            best_score = score
            best_eta = eta
    
    # If no valid candidate was found, return a default heuristic
    if best_eta is None:
        # Default: budget-aware form with lambda=2.0, exp=1.5
        cost_safe = np.maximum(total_cost, eps)
        base = prizes[None, :] / (cost_safe ** 1.5)
        penalty = np.exp(-2.0 * total_cost / max_len)
        best_eta = base * penalty
        best_eta = np.where(np.isfinite(best_eta) & (best_eta > 0), best_eta, 1e-9)
    
    return best_eta
