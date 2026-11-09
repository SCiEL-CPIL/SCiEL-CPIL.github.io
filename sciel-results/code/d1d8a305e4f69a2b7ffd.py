"""Heuristic design for ACO-based multiple-constraint knapsack."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _simulate_aco_fast(prizes, weights, heuristic, ants=20, iters=30, capacity=25, seed=42):
    """
    Fast vectorized ACO simulation.
    Returns the best total prize found.
    """
    n_items = len(prizes)
    n_constraints = weights.shape[0]
    
    eta = np.where(heuristic > 0, heuristic, 1e-9)
    
    tau = np.ones(n_items)
    best_prize_overall = 0.0
    
    rng = np.random.RandomState(seed)
    
    # Precompute weights transposed for faster column access? 
    # weights is (n_constraints, n_items). We need weights[:, item] often.
    # In vectorized simulation, we handle state per ant.
    
    for _ in range(iters):
        best_prize_this_iter = 0.0
        
        # State for all ants
        current_weights = np.zeros((ants, n_constraints))
        total_prizes = np.zeros(ants)
        used_items = np.zeros((ants, n_items), dtype=bool)
        
        # We need to simulate item addition. 
        # A fully vectorized "while any ant can add an item" loop is complex 
        # because different ants finish at different times.
        # However, we can process step-by-step.
        
        active_ants = np.ones(ants, dtype=bool)
        
        # Max possible items per ant is limited by capacity/min_weight, but practically < 100.
        # We loop until no active ant can add another item.
        
        while np.any(active_ants):
            # Calculate remaining capacity for active ants
            remaining_cap = capacity - current_weights[active_ants]
            
            # Check which items fit for which active ants
            # active_ants_indices = np.where(active_ants)[0]
            
            # fits_mask: (num_active_ants, n_items)
            # We need to check if weights[:, item] <= remaining_cap[ant] for all constraints.
            
            # weights is (C, N). We want (A, C, N) <= (A, C, 1)
            # Let A be number of active ants.
            # remaining_cap is (A, C).
            # weights is (C, N).
            # We need to broadcast weights to (C, N) and compare with remaining_cap.T (C, A).
            
            # It is more efficient to compute for all items at once if possible, 
            # but 'fits' depends on the specific ant's remaining capacity.
            
            # Let's extract indices of active ants to keep arrays small
            active_idx = np.where(active_ants)[0]
            num_active = len(active_idx)
            
            if num_active == 0:
                break
            
            # remaining_cap_active: (num_active, C)
            rc = current_weights[active_idx]
            rem = capacity - rc
            
            # Check feasibility: weights (C, N) <= rem (A, C)
            # rem.T is (C, A). weights is (C, N).
            # We need (C, A, N) comparison? No.
            # For each ant a and item j: all_c(weights[c, j] <= rem[a, c])
            
            # Vectorized check:
            # rem is (A, C). weights is (C, N).
            # We want a boolean matrix F of shape (A, N) where F[a, j] is True if item j fits ant a.
            
            # Use broadcasting:
            # weights_expanded: (1, C, N) -> no, weights is (C, N).
            # Let's transpose weights to (N, C).
            # wT = weights.T (N, C)
            # rem is (A, C)
            # We want wT[j, c] <= rem[a, c] for all c.
            # (N, 1, C) <= (1, A, C) -> (N, A, C) -> all along C -> (N, A) -> transpose to (A, N)
            
            wT = weights.T # (N, C)
            # rem is (A, C). 
            # Compare wT (N, C) with rem (A, C).
            # We need to broadcast.
            # wT[None, :, :] is (1, N, C)
            # rem[:, None, :] is (A, 1, C)
            # Comparison: (1, N, C) <= (A, 1, C) -> (A, N, C)
            # This creates a large array (A, N, C). A=20, N=100, C=5. 20*100*5 = 10,000 floats. Very fast.
            
            fits_3d = (wT[None, :, :] <= rem[:, None, :]) # (A, N, C)
            fits_mask = np.all(fits_3d, axis=2) # (A, N)
            
            # Mask out already used items
            used_active = used_items[active_idx] # (A, N)
            fits_mask &= ~used_active
            
            # Find items that fit for each ant
            # If no item fits for an ant, it is done.
            
            # Calculate probabilities for each ant
            # tau: (N,), eta: (N,)
            # probs for ant a: tau * eta * fits_mask[a]
            
            # (A, N) * (N,) -> (A, N)
            base_probs = tau * eta
            probs = base_probs[None, :] * fits_mask # (A, N)
            
            # Normalize
            sums = probs.sum(axis=1, keepdims=True)
            
            # Handle zeros
            valid_sums = sums > 0
            probs[valid_sums] /= sums[valid_sums]
            
            # Determine which ants can still add an item
            can_add = sums.flatten() > 0
            active_ants[active_idx[~can_add]] = False
            
            if not np.any(can_add):
                break
            
            # For ants that can add, select an item
            # We need to sample from each row of probs[can_add]
            
            # np.random.choice doesn't vectorize over rows directly.
            # We can use Gumbel trick or just loop over the small number of active ants?
            # A <= 20. Looping over 20 ants is fast.
            
            active_can_add_idx = active_idx[can_add]
            
            for k, a_idx in enumerate(active_can_add_idx):
                # a_idx is the global ant index
                # Get the probabilities for this ant
                p = probs[k] # This is the row corresponding to this ant in the 'can_add' subset?
                # Wait, probs was computed for all active_idx.
                # We need to map k back to the row in probs.
                # Actually, active_idx[can_add] gives the global indices.
                # The rows in probs correspond to active_idx in order.
                # So the k-th row in probs corresponds to active_idx[k].
                # But we filtered by can_add.
                # Let's just iterate over the active_idx that can add.
                
                # Re-fetch p correctly
                # The index of this ant in the 'active_idx' array:
                # pos_in_active = np.where(active_idx == a_idx)[0][0]
                # p = probs[pos_in_active]
                
                # Simpler: iterate over all active_idx, check if it can add
                pass
            
            # Let's do it properly to avoid mapping errors
            # Iterate over the active ants that can add
            for i, a_idx in enumerate(active_idx):
                if not can_add[i]:
                    continue
                
                p = probs[i]
                # Sample item
                # rng.choice expects p to sum to 1. It does.
                item_j = rng.choice(n_items, p=p)
                
                # Update state
                total_prizes[a_idx] += prizes[item_j]
                current_weights[a_idx] += weights[:, item_j]
                used_items[a_idx, item_j] = True
        
        # End while loop
        
        # Update best
        best_prize_this_iter = np.max(total_prizes)
        if best_prize_this_iter > best_prize_overall:
            best_prize_overall = best_prize_this_iter
        
        # Update Pheromone
        tau = tau * 0.9 + best_prize_this_iter / 10.0
        
    return best_prize_overall

def _compute_features(prizes, weights):
    """
    Compute a set of features for the items.
    """
    n_items = len(prizes)
    n_constraints = weights.shape[0]
    
    # F1: Prize
    f1 = prizes
    
    # F2: Prize / Sum Weights
    sum_w = weights.sum(axis=0)
    f2 = prizes / (sum_w + 1e-9)
    
    # F3: Prize / Max Weights
    max_w = weights.max(axis=0)
    f3 = prizes / (max_w + 1e-9)
    
    # F4: Prize / Min Weights (if min is 0, this is inf, handle carefully)
    min_w = weights.min(axis=0)
    # If min_w is 0, the item is "free" in at least one constraint, which is good.
    # But division by zero is bad.
    # Use reciprocal of (min_w + eps)
    f4 = prizes / (min_w + 1e-9)
    
    # F5: Prize * (1 - Max Weights)
    f5 = prizes * (1 - max_w)
    
    # F6: Prize / Geometric Mean Weights
    log_w = np.log(weights + 1e-9)
    geo_mean = np.exp(log_w.mean(axis=0))
    f6 = prizes / (geo_mean + 1e-9)
    
    # Normalize features to [0, 1] range to help optimization stability
    def norm(f):
        min_v = f.min()
        max_v = f.max()
        if max_v - min_v < 1e-9:
            return np.ones_like(f)
        return (f - min_v) / (max_v - min_v)
    
    features = np.column_stack([
        norm(f1),
        norm(f2),
        norm(f3),
        norm(f4),
        norm(f5),
        norm(f6)
    ])
    
    return features

def heuristics(prizes, weights):
    start_time = time.time()
    deadline = start_time + 290.0 # Safety margin
    
    n_items = len(prizes)
    
    # 1. Compute features
    features = _compute_features(prizes, weights)
    n_features = features.shape[1]
    
    # 2. Define objective function for optimization
    # We want to maximize the ACO performance.
    # Minimize negative performance.
    
    def objective(params):
        # params: weights for linear combination of features
        # heuristic = params @ features
        h = params @ features
        
        # Ensure positive
        h = np.where(h > 0, h, 1e-9)
        
        # Run simulation
        # Use a fixed seed for deterministic evaluation during optimization
        score = _simulate_aco_fast(prizes, weights, h, seed=42)
        
        return -score
    
    # 3. Initial guess
    # Start with uniform weights or simple ratio
    x0 = np.ones(n_features) / n_features
    
    # 4. Optimize
    best_params = x0
    best_score = objective(x0)
    
    # Nelder-Mead is derivative-free and good for small dimensions
    # We have 6 features, so 6 parameters.
    
    # We can also try to optimize the "exponent" of the heuristic?
    # Let's stick to linear combination first.
    
    try:
        # Set time limit for minimize
        # scipy minimize doesn't have a direct time limit, but we can check in callback
        # Or just run it and hope it's fast.
        # With vectorized sim, each eval is ~10-50ms.
        # Nelder-Mead might take 100-500 evals. 500 * 50ms = 25s. Safe.
        
        result = minimize(
            objective, 
            x0, 
            method='Nelder-Mead', 
            options={
                'maxiter': 200,
                'xatol': 1e-4,
                'fatol': 1e-4,
                'adaptive': True
            }
        )
        
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x
            
    except Exception:
        pass
    
    # 5. Generate final heuristic
    final_heuristic = best_params @ features
    final_heuristic = np.where(final_heuristic > 0, final_heuristic, 1e-9)
    
    # 6. Validation
    if not np.all(np.isfinite(final_heuristic)):
        # Fallback to simple ratio
        final_heuristic = prizes / (weights.sum(axis=0) + 1e-9)
    
    return final_heuristic
# EVOLVE-BLOCK-END
