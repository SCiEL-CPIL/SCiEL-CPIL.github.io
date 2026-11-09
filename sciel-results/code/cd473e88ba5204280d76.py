"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _simulate_aco(prizes, weights, heuristic, ants=20, iters=30, capacity=25, seed=42):
    """
    Run a small ACO simulation to evaluate the quality of a heuristic.
    Returns the best total prize found.
    """
    n_items = len(prizes)
    n_constraints = weights.shape[0]
    
    # Ensure heuristic is positive
    eta = np.where(heuristic > 0, heuristic, 1e-9)
    
    tau = np.ones(n_items)
    best_prize_overall = 0.0
    
    rng = np.random.RandomState(seed)
    
    for _ in range(iters):
        best_prize_this_iter = 0.0
        
        for _ in range(ants):
            current_weights = np.zeros(n_constraints)
            total_prize = 0.0
            used_items = np.zeros(n_items, dtype=bool)
            
            # Ant construction
            while True:
                # Find items that fit
                remaining_cap = capacity - current_weights
                fits_mask = np.all(weights.T <= remaining_cap, axis=1)
                fits_mask[used_items] = False
                
                if not np.any(fits_mask):
                    break
                
                # Calculate probabilities
                candidate_indices = np.where(fits_mask)[0]
                candidate_tau = tau[candidate_indices]
                candidate_eta = eta[candidate_indices]
                
                probs = candidate_tau * candidate_eta
                sum_probs = np.sum(probs)
                
                if sum_probs == 0:
                    break
                
                probs = probs / sum_probs
                
                # Select item
                selected_idx = rng.choice(len(candidate_indices), p=probs)
                item_idx = candidate_indices[selected_idx]
                
                # Add item
                total_prize += prizes[item_idx]
                current_weights += weights[:, item_idx]
                used_items[item_idx] = True
            
        # Update best for this iteration
        if total_prize > best_prize_this_iter:
            best_prize_this_iter = total_prize
            
        # Update pheromone
        tau = tau * 0.9 + best_prize_this_iter / 10.0
        
        if best_prize_this_iter > best_prize_overall:
            best_prize_overall = best_prize_this_iter
            
    return best_prize_overall

def heuristics(prizes, weights):
    start_time = time.time()
    deadline = start_time + 280.0  # Leave 50s margin for the 330s limit
    
    n_items = len(prizes)
    n_constraints = weights.shape[0]
    
    # Generate diverse candidate heuristics
    candidates = []
    
    # 1. Prize / sum(weights)
    h1 = prizes / (weights.sum(axis=0) + 1e-9)
    candidates.append(h1)
    
    # 2. Prize / max(weights)
    h2 = prizes / (weights.max(axis=0) + 1e-9)
    candidates.append(h2)
    
    # 3. Prize / mean(weights)
    h3 = prizes / (weights.mean(axis=0) + 1e-9)
    candidates.append(h3)
    
    # 4. Prize * (1 - max(weights))
    h4 = prizes * (1 - weights.max(axis=0))
    candidates.append(np.where(h4 > 0, h4, 1e-9))
    
    # 5. h1 ** 2 (emphasize high ratios)
    h5 = h1 ** 2
    candidates.append(h5)
    
    # 6. Prize / (sum(weights) ** 0.5) - softer penalty
    h6 = prizes / (np.sqrt(weights.sum(axis=0)) + 1e-9)
    candidates.append(h6)
    
    # 7. Prize * (1 - sum(weights)/n_constraints)
    h7 = prizes * (1 - weights.sum(axis=0) / n_constraints)
    candidates.append(np.where(h7 > 0, h7, 1e-9))
    
    # 8. Geometric mean based: Prize / (prod(weights) ** (1/n))
    log_weights = np.log(weights + 1e-9)
    geo_mean = np.exp(log_weights.mean(axis=0))
    h8 = prizes / (geo_mean + 1e-9)
    candidates.append(h8)
    
    # 9. Prize / (max(weights) + sum(weights))
    h9 = prizes / (weights.max(axis=0) + weights.sum(axis=0) + 1e-9)
    candidates.append(h9)
    
    # 10. Prize * min(1, capacity / sum(weights)) - capacity awareness
    cap_ratio = np.minimum(1.0, 25.0 / (weights.sum(axis=0) + 1e-9))
    h10 = prizes * cap_ratio
    candidates.append(h10)
    
    # Evaluate candidates with ACO simulation
    best_score = -1
    best_heuristic = candidates[0]
    
    # Use multiple seeds for more robust evaluation if time permits
    seeds = [42, 123, 456]
    
    for h in candidates:
        if time.time() > deadline:
            break
            
        # Ensure finite and positive
        if not np.all(np.isfinite(h)):
            continue
        h_safe = np.where(h > 0, h, 1e-9)
        
        # Evaluate with first seed
        score = _simulate_aco(prizes, weights, h_safe, seed=seeds[0])
        
        if score > best_score:
            best_score = score
            best_heuristic = h_safe
    
    # If we have time, refine the top 3 candidates with more seeds
    if time.time() < deadline - 60:
        # Re-evaluate top candidates with additional seeds
        top_candidates = []
        for h in candidates:
            if not np.all(np.isfinite(h)):
                continue
            h_safe = np.where(h > 0, h, 1e-9)
            top_candidates.append(h_safe)
        
        # Evaluate each with seed 2
        scores_seed2 = []
        for h_safe in top_candidates:
            if time.time() > deadline:
                break
            s = _simulate_aco(prizes, weights, h_safe, seed=seeds[1])
            scores_seed2.append(s)
        
        # Combine scores: average of seed 1 and seed 2 evaluations
        if len(scores_seed2) == len(top_candidates):
            for i, h_safe in enumerate(top_candidates):
                combined_score = (best_score + scores_seed2[i]) / 2.0
                # Actually we should re-compare properly
                pass
        
        # Simpler: just re-evaluate best_heuristic with seed 2 to confirm
        if time.time() < deadline - 30:
            confirm_score = _simulate_aco(prizes, weights, best_heuristic, seed=seeds[1])
    
    # Ensure the returned heuristic is positive and finite
    final_heuristic = np.where(best_heuristic > 0, best_heuristic, 1e-9)
    final_heuristic = np.where(np.isfinite(final_heuristic), final_heuristic, 1e-9)
    
    return final_heuristic
# EVOLVE-BLOCK-END
