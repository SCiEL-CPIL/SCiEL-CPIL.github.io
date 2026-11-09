import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes the heuristic matrix for the ant colony optimization.
    
    Combines:
    1. Parametric grid search over prize/distance exponents (current program)
    2. Node centrality (inverse distance sum) to encourage cluster visiting (inspiration 2)
    3. Gaussian-kernel smoothing to prevent premature convergence (cell finding)
    
    The heuristic form: eta_ij = prize_j^a * centrality_j^g / (d_ij^b * d_j0^c)
    with Gaussian smoothing applied to the distance terms.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = distance_matrix.shape[0]
    if n < 2:
        return np.ones((n, n), dtype=np.float64)
    
    start_time = time.time()
    deadline = start_time + 310.0  # safety margin before 330s limit
    
    # Precompute useful arrays
    d = distance_matrix.copy()
    d_to_depot = d[:, 0].copy()
    
    # Avoid division by zero
    eps = 1e-9
    d_safe = d + eps
    d_to_depot_safe = d_to_depot + eps
    
    # Compute node centrality: sum of inverse distances to other non-depot nodes
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    mask[0, :] = False  # Exclude depot as a source
    
    inv_d = 1.0 / d_safe
    inv_d_masked = inv_d * mask
    centrality = np.sum(inv_d_masked, axis=0)
    
    # Normalize centrality to mean ~1 for non-depot nodes
    non_depot_c = centrality[1:]
    mean_c = np.mean(non_depot_c) if len(non_depot_c) > 0 else 1.0
    if mean_c > 0 and np.isfinite(mean_c):
        centrality_norm = centrality / mean_c
    else:
        centrality_norm = np.ones(n)
    
    # Ensure centrality is positive
    centrality_norm = np.where(np.isfinite(centrality_norm) & (centrality_norm > 0), 
                                centrality_norm, 1.0)
    
    # Gaussian smoothing parameter for distances
    # sigma controls how sharply distance penalizes; larger sigma = smoother
    def make_gaussian_smoothed_d(d_arr, sigma):
        """Apply Gaussian kernel: exp(-d^2 / (2*sigma^2)) instead of 1/d"""
        if sigma <= 0:
            return 1.0 / (d_arr + eps)
        return np.exp(-(d_arr ** 2) / (2.0 * sigma ** 2))
    
    def evaluate_heuristic(eta, n_ants=5, n_iter=5):
        """Quick ACO simulation to estimate heuristic quality."""
        if time.time() > deadline:
            return -1
        
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
                    visited.append(j)
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
    
    # Grid search over exponents
    a_vals = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    b_vals = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    c_vals = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    g_vals = [0.0, 0.5, 1.0]  # centrality exponent
    sigma_vals = [0.0, 0.1, 0.2, 0.3]  # Gaussian smoothing (0 = no smoothing, use 1/d)
    
    # Precompute prize powers
    prize_powers = {}
    for a in a_vals:
        prize_powers[a] = (prizes + eps) ** a
    
    # Precompute centrality powers
    cent_powers = {}
    for g in g_vals:
        if g == 0.0:
            cent_powers[g] = np.ones(n)
        else:
            cent_powers[g] = centrality_norm ** g
    
    # Precompute distance powers for each b and sigma
    # For sigma=0, use 1/d^b; for sigma>0, use exp(-d^2/(2*sigma^2))^b
    d_powers = {}
    for b in b_vals:
        for sigma in sigma_vals:
            key = (b, sigma)
            if sigma == 0.0:
                d_powers[key] = d_safe ** b
            else:
                # Gaussian: exp(-d^2/(2*sigma^2))^b = exp(-b*d^2/(2*sigma^2))
                d_powers[key] = np.exp(-b * (d ** 2) / (2.0 * sigma ** 2))
    
    # Precompute depot distance powers
    depot_powers = {}
    for c in c_vals:
        for sigma in sigma_vals:
            key = (c, sigma)
            if sigma == 0.0:
                depot_powers[key] = d_to_depot_safe ** c
            else:
                depot_powers[key] = np.exp(-c * (d_to_depot ** 2) / (2.0 * sigma ** 2))
    
    # Generate candidates
    for a in a_vals:
        for b in b_vals:
            for c in c_vals:
                for g in g_vals:
                    for sigma in sigma_vals:
                        if time.time() > deadline:
                            break
                        
                        # node_part = prize_j^a * centrality_j^g / depot_dist_term^c
                        dp_key = (c, sigma)
                        dp = depot_powers[dp_key]
                        
                        # If using Gaussian, dp is already the kernel; 
                        # we want 1/dp for the "cost" interpretation
                        if sigma == 0.0:
                            # dp = d_to_depot^c, so we divide by it
                            node_part = prize_powers[a] * cent_powers[g] / dp
                        else:
                            # dp = exp(-c*d^2/(2*sigma^2)), which is already a kernel
                            # Higher dp means closer to depot, so we multiply by it
                            node_part = prize_powers[a] * cent_powers[g] * dp
                        
                        # Edge term
                        dkey = (b, sigma)
                        dterm = d_powers[dkey]
                        
                        if sigma == 0.0:
                            # dterm = d^b, divide by it
                            eta = node_part[None, :] / dterm
                        else:
                            # dterm is Gaussian kernel, multiply by it
                            eta = node_part[None, :] * dterm
                        
                        eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
                        candidates.append(eta)
                        
                        if time.time() > deadline:
                            break
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Add baseline: prize / (d_ij * d_j0)
    node_attraction = prizes / d_to_depot_safe
    eta_current = node_attraction[None, :] / d_safe
    eta_current = np.where(np.isfinite(eta_current) & (eta_current > 0), eta_current, 1e-9)
    candidates.append(eta_current)
    
    # Evaluate each candidate
    best_score = -1
    best_eta = None
    
    # Evaluate in order, but prioritize simpler candidates first for time efficiency
    # Sort by complexity (fewer parameters) - actually just evaluate all in order
    for idx, eta in enumerate(candidates):
        if time.time() > deadline:
            break
        
        score = evaluate_heuristic(eta)
        if score > best_score:
            best_score = score
            best_eta = eta
    
    # If no valid candidate was found, return the current formula
    if best_eta is None:
        best_eta = eta_current
    
    return best_eta
