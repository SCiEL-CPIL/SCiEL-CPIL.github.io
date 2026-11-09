"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _simulate_aco(prizes, weights, heuristic, ants=10, iters=10, capacity=25):
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
    
    for _ in range(iters):
        best_prize_this_iter = 0.0
        best_ant_items = []
        
        for _ in range(ants):
            current_weights = np.zeros(n_constraints)
            total_prize = 0.0
            used_items = np.zeros(n_items, dtype=bool)
            ant_items = []
            
            # Ant construction
            while True:
                # Find items that fit
                # weights[j, i] <= capacity - current_weights[j] for all j
                # We can vectorize the fit check
                remaining_cap = capacity - current_weights
                # fits[i] is True if for all j, weights[j, i] <= remaining_cap[j]
                # weights is (5, 100), remaining_cap is (5,)
                # weights.T is (100, 5)
                fits_mask = np.all(weights.T <= remaining_cap, axis=1)
                fits_mask[used_items] = False
                
                if not np.any(fits_mask):
                    break
                
                # Calculate probabilities
                # Only consider items that fit
                candidate_indices = np.where(fits_mask)[0]
                candidate_tau = tau[candidate_indices]
                candidate_eta = eta[candidate_indices]
                
                probs = candidate_tau * candidate_eta
                sum_probs = np.sum(probs)
                
                if sum_probs == 0:
                    break
                
                probs = probs / sum_probs
                
                # Select item
                selected_idx = np.random.choice(len(candidate_indices), p=probs)
                item_idx = candidate_indices[selected_idx]
                
                # Add item
                ant_items.append(item_idx)
                total_prize += prizes[item_idx]
                current_weights += weights[:, item_idx]
                used_items[item_idx] = True
            
        # Update best for this iteration
        if total_prize > best_prize_this_iter:
            best_prize_this_iter = total_prize
            best_ant_items = ant_items
            
        # Update pheromone
        # tau = tau * 0.9 + best_prize / 10
        # Note: The prompt says "every entry gains (best prize of that iteration) / 10"
        # It does not say only items in the best ant gain it. "every entry gains".
        tau = tau * 0.9 + best_prize_this_iter / 10.0
        
        if best_prize_this_iter > best_prize_overall:
            best_prize_overall = best_prize_this_iter
            
    return best_prize_overall

def heuristics(prizes, weights):
    start_time = time.time()
    deadline = start_time + 250.0  # Leave 50s margin for overhead/other evaluations if any, though this is called once.
    
    n_items = len(prizes)
    
    # Candidate heuristics
    # 1. Baseline: prize / sum(weights)
    h1 = prizes / (weights.sum(axis=0) + 1e-9)
    
    # 2. Prize / max(weights)
    h2 = prizes / (weights.max(axis=0) + 1e-9)
    
    # 3. Prize / mean(weights)
    h3 = prizes / (weights.mean(axis=0) + 1e-9)
    
    # 4. Prize * (1 - max(weights))
    h4 = prizes * (1 - weights.max(axis=0))
    h4 = np.where(h4 > 0, h4, 1e-9)
    
    # 5. Normalized h1
    mean_h1 = np.mean(h1)
    if mean_h1 > 0:
        h5 = h1 / mean_h1
    else:
        h5 = h1
        
    # 6. A "greedy" based heuristic. 
    # Sort by h1, simulate a greedy fill, items picked early get higher score?
    # Or just use h1 but with a penalty for items that are "heavy" in the tightest constraint.
    # Let's stick to the 5 above for now, maybe add one more.
    
    # 7. h1 ** 2 (emphasize high ratios)
    h7 = h1 ** 2
    
    candidates = [h1, h2, h3, h4, h5, h7]
    
    best_score = -1
    best_heuristic = h1 # Default to baseline
    
    for h in candidates:
        if time.time() > deadline:
            break
            
        # Ensure finite and positive
        if not np.all(np.isfinite(h)):
            continue
        h_safe = np.where(h > 0, h, 1e-9)
        
        score = _simulate_aco(prizes, weights, h_safe)
        
        if score > best_score:
            best_score = score
            best_heuristic = h_safe
            
    # Optional: If we have time, refine the best heuristic with a slightly larger simulation?
    # Or just return the best one found.
    
    # Ensure the returned heuristic is positive and finite
    final_heuristic = np.where(best_heuristic > 0, best_heuristic, 1e-9)
    final_heuristic = np.where(np.isfinite(final_heuristic), final_heuristic, 1e-9)
    
    return final_heuristic
# EVOLVE-BLOCK-END
