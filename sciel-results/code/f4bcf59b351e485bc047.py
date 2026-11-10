"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multiple-constraint knapsack.
    
    Strategy: Simulated Annealing Lagrangian Dual + Dynamic Contextual Density.
    
    1. Problem Framing:
       The static Lagrangian dual approach often gets stuck in local minima of the 
       dual function, leading to suboptimal shadow prices. This results in reduced 
       costs that do not accurately reflect the true marginal value of items in a 
       tight multi-constraint environment.
       
    2. Method:
       A) Simulated Annealing (SA) for Lagrangian Dual:
          Instead of simple gradient descent, we use SA to explore the space of 
          shadow prices (lambda). This allows escaping local minima and finding 
          a global (or near-global) minimum of the Lagrangian dual function.
          The objective is to maximize the dual lower bound: 
          L(lambda) = sum_i max(0, p_i - lambda^T w_i) + lambda^T C.
          
       B) Dynamic Contextual Density:
          Once we have good shadow prices, we compute the "effective capacity" 
          remaining for each constraint. We then simulate a greedy packing using 
          these prices to identify a "core" set of items. 
          For each item, we compute a contextual score based on:
          - Its reduced cost (prize - shadow_cost).
          - Its "blocking potential": how much it constrains the tightest 
            remaining resources relative to its value.
          - A dynamic density ratio that adjusts for the specific residual 
            capacities found in the best dual solution.
            
    3. Heuristic Construction:
       - Run SA to optimize lambda.
       - Compute reduced costs rc = prizes - weights @ lambda.
       - Identify the "critical" constraints (those with high lambda).
       - Compute a dynamic density: 
         score_i = (rc_i + epsilon) / (1 + max_c (w_ic / C_c))
         This penalizes items that are heavy in any constraint, but rewards 
         high reduced cost.
       - Combine with a feasibility check: items that cannot fit in the 
         residual capacity of the best dual solution are penalized.
    """
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Handle shapes: prizes (5, 100) -> (100,), weights (5, 100)
    if prizes.ndim == 2:
        prizes = prizes.sum(axis=0)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    
    n_items = weights.shape[1]
    n_constraints = weights.shape[0]
    capacity = 25.0
    C = np.full(n_constraints, capacity)
    
    # Time budget guard
    start_time = time.time()
    time_limit = 290.0  # Safety margin before 300s limit
    
    # --- Part 1: Simulated Annealing for Lagrangian Dual ---
    # Objective: Maximize L(lambda) = sum_i max(0, p_i - w_i @ lambda) + lambda @ C
    # We minimize -L(lambda).
    
    def lagrangian_value(lam):
        shadow_costs = weights @ lam  # (n_items,)
        reduced_costs = prizes - shadow_costs
        # Sum of max(0, rc)
        pos_rc = np.maximum(0, reduced_costs)
        dual_obj = np.sum(pos_rc) + np.dot(lam, C)
        return dual_obj
    
    # Initialize lambda randomly or with a simple heuristic
    rng = np.random.RandomState(42)
    lam_current = rng.uniform(0, 1, size=n_constraints)
    lam_best = lam_current.copy()
    best_val = lagrangian_value(lam_current)
    
    T = 1.0  # Initial temperature
    cooling_rate = 0.95
    min_T = 1e-4
    max_steps = 5000
    
    step_size = 0.5
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
        if T < min_T:
            break
            
        # Generate neighbor
        lam_neighbor = lam_current + rng.normal(0, step_size, size=n_constraints)
        lam_neighbor = np.maximum(0, lam_neighbor)  # Shadow prices must be non-negative
        
        val_neighbor = lagrangian_value(lam_neighbor)
        delta_val = val_neighbor - best_val
        
        # Accept if better or with probability exp(delta/T)
        if delta_val > 0 or rng.uniform() < np.exp(delta_val / T):
            lam_current = lam_neighbor
            if val_neighbor > best_val:
                best_val = val_neighbor
                lam_best = lam_neighbor.copy()
        
        T *= cooling_rate
        step_size *= 0.99 # Slowly reduce step size
    
    # Final reduced costs
    shadow_costs = weights @ lam_best
    rc = prizes - shadow_costs
    
    # --- Part 2: Dynamic Contextual Density ---
    # We want to score items based on their value relative to their "cost" in the 
    # tightest constraints, adjusted by the dual solution.
    
    # Compute the residual capacity if we were to pick items with high rc greedily
    # This is a proxy for the "context" in which items are evaluated.
    
    # Sort by reduced cost
    order = np.argsort(-rc)
    rem_cap = C.copy()
    selected = np.zeros(n_items, dtype=bool)
    total_prize = 0.0
    
    # Simulate a greedy pack to see what fits
    for i in order:
        if rc[i] <= 0:
            break
        if np.all(weights[:, i] <= rem_cap):
            selected[i] = True
            rem_cap -= weights[:, i]
            total_prize += prizes[i]
    
    # The "contextual" residual capacity is rem_cap.
    # An item is attractive if it has high rc and fits easily into the residual 
    # capacity, OR if it is small enough to be a "filler" for tight constraints.
    
    # Compute a "fit" score for each item relative to the residual capacity.
    # If an item fits in the residual capacity, it's a strong candidate.
    # If not, it's only useful if it replaces a lower-value item (which we can't 
    # fully model statically, so we penalize heavy items).
    
    # Dynamic Density: 
    # score_i = rc_i / (1 + sum_c (w_ic / max(rem_cap_c, epsilon)))
    # This penalizes items that consume a large fraction of the remaining capacity.
    
    eps = 1e-9
    safe_rem_cap = np.maximum(rem_cap, eps)
    
    # Normalized weight consumption relative to residual capacity
    # Shape: (n_constraints, n_items)
    weight_ratios = weights / safe_rem_cap[:, np.newaxis]
    
    # Max ratio across constraints (bottleneck)
    bottleneck_ratio = weight_ratios.max(axis=0)
    
    # Score: High rc, Low bottleneck ratio
    # We use a soft denominator to avoid division by zero and to smooth the penalty
    scores = rc / (1.0 + bottleneck_ratio)
    
    # Ensure positive scores
    scores = np.maximum(scores, 1e-9)
    
    # --- Part 3: Refinement with Multi-Start Greedy (Optional if time permits) ---
    # If we have time, we can boost the scores of items that appear in multiple 
    # good greedy solutions to capture discrete blocking effects.
    
    if time.time() - start_time < time_limit - 50.0:
        weighted_counts = np.zeros(n_items, dtype=float)
        
        # Strategies: 
        # 1. By rc
        # 2. By prize
        # 3. By prize/weight_sum
        # 4. By score (current)
        
        sum_w = weights.sum(axis=0)
        ratio_pw = prizes / (sum_w + eps)
        ratio_score = scores / (sum_w + eps)
        
        strategies = [
            np.argsort(-rc),
            np.argsort(-prizes),
            np.argsort(-ratio_pw),
            np.argsort(-ratio_score)
        ]
        
        for strat_order in strategies:
            if time.time() - start_time > time_limit:
                break
                
            rem_cap_sim = C.copy()
            selected_sim = []
            total_prize_sim = 0.0
            
            for idx in strat_order:
                if np.all(weights[:, idx] <= rem_cap_sim):
                    rem_cap_sim -= weights[:, idx]
                    selected_sim.append(idx)
                    total_prize_sim += prizes[idx]
            
            if total_prize_sim > 0:
                for idx in selected_sim:
                    weighted_counts[idx] += total_prize_sim
        
        # Normalize counts
        wc_max = weighted_counts.max()
        if wc_max > 0:
            wc_norm = weighted_counts / wc_max
        else:
            wc_norm = np.ones(n_items)
        
        # Combine: Geometric mean of contextual density and empirical frequency
        # This ensures items are both theoretically good (dual) and practically 
        # useful (greedy).
        scores = (scores ** 0.5) * (wc_norm ** 0.5)
        
        # Final normalization
        s_max = scores.max()
        if s_max > 0:
            scores = scores / s_max
    
    # Ensure all values are positive and finite
    scores = np.maximum(scores, 1e-9)
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
