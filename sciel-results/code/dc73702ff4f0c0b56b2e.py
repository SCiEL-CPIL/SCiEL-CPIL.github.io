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
    
    # Pheromone is uniform due to global update, so we only need eta for probabilities
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
                candidate_eta = eta[candidate_indices]
                
                sum_probs = np.sum(candidate_eta)
                
                if sum_probs == 0:
                    break
                
                probs = candidate_eta / sum_probs
                
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
            
        if best_prize_this_iter > best_prize_overall:
            best_prize_overall = best_prize_this_iter
            
    return best_prize_overall

def heuristics(prizes, weights):
    start_time = time.time()
    deadline = start_time + 280.0
    
    n_items = len(prizes)
    n_constraints = weights.shape[0]
    
    # Precompute weight features
    sum_w = weights.sum(axis=0)
    max_w = weights.max(axis=0)
    mean_w = weights.mean(axis=0)
    
    candidates = []
    
    # Search grid for exponents: eta = v / (sum_w^alpha * max_w^beta)
    for alpha in [0, 0.5, 1, 1.5, 2]:
        for beta in [0, 0.5, 1, 1.5, 2]:
            denom = 1.0
            if alpha > 0:
                denom *= (sum_w + 1e-9) ** alpha
            if beta > 0:
                denom *= (max_w + 1e-9) ** beta
            
            h = prizes / denom
            
            if np.all(np.isfinite(h)) and np.all(h > 0):
                candidates.append(h)
    
    # Add standard heuristics
    candidates.append(prizes / (sum_w + 1e-9))
    candidates.append(prizes / (max_w + 1e-9))
    candidates.append(prizes / (mean_w + 1e-9))
    
    # Quick evaluation to filter candidates
    quick_scores = []
    for h in candidates:
        if time.time() > deadline - 30:
            break
        score = _simulate_aco(prizes, weights, h, ants=5, iters=5, seed=42)
        quick_scores.append(score)
    
    # Sort candidates by quick score
    sorted_indices = np.argsort(quick_scores)[::-1]
    
    best_score = -1
    best_heuristic = candidates[0]
    
    # Full evaluation for top candidates
    for idx in sorted_indices[:10]:
        if time.time() > deadline - 10:
            break
        h = candidates[idx]
        score = _simulate_aco(prizes, weights, h, ants=20, iters=30, seed=42)
        if score > best_score:
            best_score = score
            best_heuristic = h
    
    final_heuristic = np.where(best_heuristic > 0, best_heuristic, 1e-9)
    final_heuristic = np.where(np.isfinite(final_heuristic), final_heuristic, 1e-9)
    
    return final_heuristic
# EVOLVE-BLOCK-END
