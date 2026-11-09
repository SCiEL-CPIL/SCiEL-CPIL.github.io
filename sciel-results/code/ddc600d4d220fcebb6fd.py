import numpy as np
import time
from scipy.optimize import minimize

def _run_aco_simulation(prizes, weights, heuristic, n_ants=20, n_iterations=30, capacity=25, seed=42):
    """
    Run a simplified ACO simulation to evaluate how well a heuristic performs.
    Returns the best total prize found.
    """
    num_constraints, n_items = prizes.shape
    rng = np.random.default_rng(seed)
    
    # Pheromone starts at 1
    pheromone = np.ones(n_items, dtype=np.float64)
    
    # Ensure heuristic is positive
    heuristic_safe = np.maximum(heuristic, 1e-9)
    
    best_overall = 0.0
    
    for iteration in range(n_iterations):
        iteration_best = 0.0
        
        for ant in range(n_ants):
            # Track remaining capacity per constraint
            remaining = np.full(num_constraints, capacity, dtype=np.float64)
            total_prize = 0.0
            selected = np.zeros(n_items, dtype=bool)
            
            while True:
                # Find items that still fit in all constraints
                fits = np.all(weights <= remaining[:, None], axis=0)
                fits[selected] = False
                
                if not np.any(fits):
                    break
                
                # Compute selection probabilities
                probs = pheromone * heuristic_safe
                probs[~fits] = 0.0
                
                prob_sum = probs.sum()
                if prob_sum <= 0:
                    break
                
                probs = probs / prob_sum
                
                # Select an item
                item_idx = rng.choice(n_items, p=probs)
                
                # Add it
                selected[item_idx] = True
                total_prize += prizes[:, item_idx].sum()
                remaining -= weights[:, item_idx]
            
        if total_prize > iteration_best:
            iteration_best = total_prize
        
        if total_prize > best_overall:
            best_overall = total_prize
        
        pheromone = pheromone * 0.9 + iteration_best / 10.0
    
    return best_overall

def _compute_heuristics(prizes, weights, deadline=None):
    start_time = time.time()
    if deadline is None:
        deadline = start_time + 290
        
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    num_constraints, n_items = prizes.shape
    
    # Basis functions
    total_prize = prizes.sum(axis=0)
    sum_weights = weights.sum(axis=0)
    max_weights = weights.max(axis=0)
    min_weights = weights.min(axis=0)
    eps = 1e-9
    
    # Normalize bases to [0,1] to keep exp() inputs reasonable
    def norm(x):
        mn = x.min()
        mx = x.max()
        if mx - mn < 1e-6:
            return np.ones_like(x)
        return (x - mn) / (mx - mn)
        
    b1 = norm(total_prize / (sum_weights + eps))
    b2 = norm(total_prize / (max_weights + eps))
    b3 = norm(total_prize)
    b4 = norm(1.0 / (sum_weights + eps))
    b5 = norm(1.0 / (max_weights + eps))
    
    bases = np.vstack([b1, b2, b3, b4, b5]) # shape (5, 100)
    
    def objective(c):
        # c is a vector of 5 weights
        # heuristic = exp(sum(c_i * b_i))
        # Since scale doesn't matter, we can just use the linear combination 
        # but we must ensure positivity. 
        # Using exp ensures positivity.
        # However, if c is large, exp overflows. 
        # We can normalize the linear combination or just use a softplus-like or just max(0, ...) 
        # Actually, relative values matter. 
        # Let's use h = exp(c @ bases). 
        # To prevent overflow, we can subtract the max of (c @ bases) before exp, 
        # but that shifts the scale. 
        # Since scale is irrelevant, we can do:
        # log_h = c @ bases
        # log_h = log_h - log_h.max()  # This makes max(h) = 1
        # h = np.exp(log_h)
        
        log_h = c @ bases
        # Shift to prevent overflow and keep numbers stable
        log_h -= np.max(log_h)
        h = np.exp(log_h)
        
        # Evaluate with 3 seeds to reduce variance
        scores = []
        for s in [1, 2, 3]:
            if time.time() > deadline - 5:
                break
            score = _run_aco_simulation(prizes, weights, h, seed=s)
            scores.append(score)
        
        if not scores:
            return 0.0
        return -np.mean(scores) # Minimize negative score
    
    # Initial guess: favor prize/weight (b1)
    x0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    
    best_result = None
    best_fun = np.inf
    
    try:
        # Nelder-Mead is robust for low dimensions
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
        
        if res.success or res.fun < best_fun:
            best_fun = res.fun
            best_result = res.x
    except Exception:
        pass
    
    # If optimization failed or didn't find anything, fallback to x0
    if best_result is None:
        best_result = x0
    
    # Compute final heuristic
    log_h = best_result @ bases
    log_h -= np.max(log_h)
    final_heuristic = np.exp(log_h)
    
    # Ensure positive and finite
    final_heuristic = np.maximum(final_heuristic, 1e-9)
    final_heuristic = np.where(np.isfinite(final_heuristic), final_heuristic, 1e-9)
    
    return final_heuristic

def heuristics(prizes, weights):
    start_time = time.time()
    try:
        result = _compute_heuristics(prizes, weights, start_time + 290)
        result = np.asarray(result, dtype=np.float64)
        
        if result.ndim != 1:
            raise ValueError("Heuristic must be a 1D array")
        
        if not np.all(np.isfinite(result)):
            raise ValueError("Heuristic must contain only finite values")
        
        prizes_arr = np.asarray(prizes)
        expected_n_items = prizes_arr.shape[1] if prizes_arr.ndim == 2 else len(prizes_arr)
        
        if result.shape[0] != expected_n_items:
            raise ValueError(f"Heuristic shape {result.shape} doesn't match expected ({expected_n_items},)")
        
        return result
        
    except Exception:
        prizes_arr = np.asarray(prizes, dtype=np.float64)
        if prizes_arr.ndim == 2:
            n_items = prizes_arr.shape[1]
        else:
            n_items = len(prizes_arr)
        return np.ones(n_items, dtype=np.float64)
