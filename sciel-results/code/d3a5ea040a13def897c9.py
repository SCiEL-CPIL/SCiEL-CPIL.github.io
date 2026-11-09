import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes the heuristic matrix for the ant colony optimization.
    
    Uses a parametric grid search over a budget-aware heuristic form:
    eta_ij = (p_j^alpha) / (d_ij + d_j0)^gamma * exp(-lambda * (d_ij + d_j0) / max_len)
    
    This form explicitly penalizes moves that consume a large fraction of the
    total length budget, helping ants preserve flexibility to visit more nodes.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = distance_matrix.shape[0]
    if n < 2:
        return np.ones((n, n), dtype=np.float64)
    
    start_time = time.time()
    deadline = start_time + 300.0  # Safety margin
    
    d = distance_matrix
    d_to_depot = d[:, 0].copy()
    
    # Precompute the "effective cost" of moving i->j considering the return trip
    # This represents the distance added to the route if j is the last node visited
    # before returning to 0.
    # cost_ij = d[i, j] + d[j, 0]
    cost_matrix = d + d_to_depot[None, :]
    
    # Avoid division by zero in power/exponential calculations
    eps = 1e-9
    cost_safe = cost_matrix + eps
    
    # Normalize cost by max_len to make lambda scale-invariant
    cost_norm = cost_safe / max_len if max_len > 0 else cost_safe
    
    best_score = -1
    best_eta = None
    
    # Grid search parameters
    # alpha: prize exponent (higher = more greedy on prize)
    # gamma: cost exponent (higher = more sensitive to distance)
    # lambda: exponential decay rate (higher = stronger budget penalty)
    
    alpha_vals = [0.5, 1.0, 1.5, 2.0]
    gamma_vals = [0.5, 1.0, 1.5, 2.0, 2.5]
    lambda_vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    
    # Precompute prize powers for speed
    prize_powers = {}
    for a in alpha_vals:
        prize_powers[a] = (prizes + eps) ** a
    
    def evaluate_heuristic(eta, n_ants=10, n_iter=10):
        """
        Quick ACO simulation to estimate heuristic quality.
        Uses a fixed seed for consistent comparison.
        """
        if time.time() > deadline:
            return -1
        
        # Use a local RNG for reproducibility in evaluation
        rng = np.random.RandomState(42)
        
        pheromone = np.ones((n, n))
        best_total = 0
        
        for _ in range(n_iter):
            iter_best = 0
            for _ in range(n_ants):
                visited = [0]
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
                    r = rng.random()
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
                    visited.append(j)
                    current = j
                
                if total_prize > iter_best:
                    iter_best = total_prize
                
                # Update pheromone
                pheromone *= 0.9
                pheromone += iter_best / 10.0
            
            best_total = max(best_total, iter_best)
        
        return best_total

    # Generate and evaluate candidates
    for a in alpha_vals:
        for g in gamma_vals:
            for lam in lambda_vals:
                if time.time() > deadline:
                    break
                
                # Calculate heuristic
                # eta_ij = prize_j^a / (cost_ij^g) * exp(-lambda * cost_norm_ij)
                # Note: cost_norm is (d_ij + d_j0) / max_len
                
                numerator = prize_powers[a][None, :]
                denominator = cost_safe ** g
                decay = np.exp(-lam * cost_norm)
                
                eta = (numerator / denominator) * decay
                
                # Ensure finite and positive
                eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
                
                score = evaluate_heuristic(eta)
                if score > best_score:
                    best_score = score
                    best_eta = eta
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Fallback if no candidate was evaluated or all failed
    if best_eta is None:
        # Use a simple robust heuristic
        eta = (prizes[None, :] + eps) / (cost_safe ** 1.0)
        best_eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    return best_eta
