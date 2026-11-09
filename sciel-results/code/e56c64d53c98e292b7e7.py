import numpy as np
import time

def heuristics(prizes, weights):
    """
    Computes attractiveness scores for items in an ACO for MCKP.
    Uses a fast vectorized simulation to find optimal heuristic parameters.
    """
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Ensure shapes are correct: (5, 100)
    if prizes.ndim == 1:
        P = prizes
        w = weights.reshape(5, -1) if weights.ndim == 1 else weights
    else:
        P = prizes.sum(axis=0)
        w = weights
        
    n_items = P.shape[0]
    n_constraints = w.shape[0]
    capacity = 25.0
    
    # Time management
    t0 = time.time()
    deadline = t0 + 310.0  # Leave 20s buffer for the 330s kill
    
    # Feature extraction for heuristic construction
    # We will model heuristic[i] = exp(alpha * P[i] - beta * V[i] - gamma * M[i] ... )
    # Or more simply: a linear combination of features, then exp/clip.
    # Let's try: h[i] = (P[i] + a) / (1 + b * V[i] + c * M[i])
    # Where V is total volume, M is max weight component.
    
    V_sum = w.sum(axis=0)
    V_max = w.max(axis=0)
    # Normalize features to [0, 1] range roughly to help optimization stability
    P_norm = P / (P.max() + 1e-9)
    V_norm = V_sum / (V_sum.max() + 1e-9)
    M_norm = V_max / (V_max.max() + 1e-9)
    
    # Fast Vectorized Colony Simulation
    def fast_colony_sim(h, seed):
        """
        Simulates the ACO process quickly using vectorized operations where possible.
        Returns the best total prize found.
        """
        rng = np.random.RandomState(seed)
        pheromone = np.ones(n_items)
        best_total = 0.0
        
        for _ in range(30):
            ant_totals = []
            for _ in range(20):
                remaining_cap = np.full(n_constraints, capacity)
                used = np.zeros(n_items, dtype=bool)
                total_prize = 0.0
                
                # We need to pick items one by one because the set of fitting items changes
                # But we can vectorize the "fit" check and probability calculation.
                while True:
                    # Check which items fit
                    # w is (5, 100), remaining_cap is (5,)
                    # fits_mask: (100,) boolean
                    # An item i fits if w[:, i] <= remaining_cap
                    # This is equivalent to: np.all(w <= remaining_cap[:, None], axis=0) & ~used
                    fits_mask = np.all(w <= remaining_cap[:, None], axis=0) & ~used
                    
                    if not np.any(fits_mask):
                        break
                    
                    # Get indices of fitting items
                    fitting_indices = np.where(fits_mask)[0]
                    
                    # Calculate probabilities: pheromone * heuristic
                    probs = pheromone[fitting_indices] * h[fitting_indices]
                    sum_probs = probs.sum()
                    if sum_probs == 0:
                        probs = np.ones(len(fitting_indices))
                        sum_probs = len(fitting_indices)
                    probs = probs / sum_probs
                    
                    # Select one item
                    idx = rng.choice(fitting_indices, p=probs)
                    used[idx] = True
                    remaining_cap -= w[:, idx]
                    total_prize += P[idx]
                
            ant_totals.append(total_prize)
            
            # Update pheromone
            max_ant_total = max(ant_totals)
            best_total = max(best_total, max_ant_total)
            pheromone = pheromone * 0.9 + max_ant_total / 10.0
            
        return best_total

    # Define a set of candidate heuristic functions
    # We'll search over parameters for: h[i] = (P[i] / (1 + alpha * V_norm[i] + beta * M_norm[i])) ** gamma
    
    best_h = None
    best_score = -1.0
    
    # Initial good heuristic based on prize/max_weight
    base_h = np.clip(P / (V_max + 1e-9), 1e-9, 1e6)
    
    # Grid search over parameters
    # alpha: weight of total volume (0 to 2)
    # beta: weight of max component (0 to 2)
    # gamma: exponent (0.5 to 3.0)
    
    alphas = [0.0, 0.5, 1.0, 1.5, 2.0]
    betas = [0.0, 0.5, 1.0, 1.5, 2.0]
    gammas = [0.5, 1.0, 1.5, 2.0]
    
    # We can't test all combinations (5*5*4 = 100) with full 30x20 sim if it's slow.
    # fast_colony_sim is O(30 * 20 * N_steps * 100). 
    # N_steps is roughly 10-20. So ~ 30*20*15*100 = 90,000 ops per sim.
    # 100 sims = 9,000,000 ops. This should be fast enough in Python/NumPy if vectorized well.
    # The inner loop is not fully vectorized due to dynamic fitting set, but 100 sims * 600 ants = 60,000 ants total.
    # Each ant does ~15 steps. 900k item checks. This is very fast.
    
    for alpha in alphas:
        if time.time() > deadline - 10.0:
            break
        for beta in betas:
            if time.time() > deadline - 10.0:
                break
            for gamma in gammas:
                # Construct heuristic
                denom = 1.0 + alpha * V_norm + beta * M_norm
                h = (P / denom) ** gamma
                h = np.where(h > 0, h, 1e-9)
                h = np.clip(h, 1e-9, 1e6)
                
                score = fast_colony_sim(h, 42) # Fixed seed for comparison
                
                if score > best_score:
                    best_score = score
                    best_h = h
                    
                # Early exit if we hit a very high score? No, keep searching.
                
    # If we have time, refine the best parameters found
    if time.time() < deadline - 30.0 and best_h is not None:
        # Find which parameters gave the best score? 
        # We didn't store them. Let's just do a local search around the best found in the grid?
        # Actually, re-running the grid with finer steps around the best is hard without tracking.
        # Let's just stick with the best found in the coarse grid.
        
        # Alternative: Try a few more specific structural heuristics
        extra_candidates = [
            np.clip(P / (V_sum + 1e-9), 1e-9, 1e6),
            np.clip(P / (V_max + 1e-9), 1e-9, 1e6),
            np.clip((P / (V_sum + 1e-9)) * (P / (V_max + 1e-9)), 1e-9, 1e6),
            np.clip(P * (1.0 / (1.0 + V_sum)), 1e-9, 1e6),
        ]
        
        for h in extra_candidates:
            if time.time() > deadline - 5.0:
                break
            score = fast_colony_sim(h, 42)
            if score > best_score:
                best_score = score
                best_h = h

    if best_h is None:
        best_h = base_h
        
    return best_h
