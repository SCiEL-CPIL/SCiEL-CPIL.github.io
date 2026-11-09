import numpy as np

def heuristics(prizes, weights):
    # prizes: (5, 100), weights: (5, 100)
    n_items = prizes.shape[1]
    n_constraints = prizes.shape[0]
    capacity = 25.0
    
    # Scalar prize per item
    item_prizes = np.sum(prizes, axis=0)
    
    # Pre-compute weight statistics
    sum_w = weights.sum(axis=0)
    max_w = weights.max(axis=0)
    total_w_per_constraint = weights.sum(axis=1)
    tightness = total_w_per_constraint / capacity
    
    # Candidate 1: Density
    h1 = item_prizes / (sum_w + 1e-9)
    
    # Candidate 2: Max Weight Density
    h2 = item_prizes / (max_w + 1e-9)
    
    # Candidate 3: Tightness Weighted Density
    weighted_sum = np.sum(weights * tightness, axis=0)
    h3 = item_prizes / (weighted_sum + 1e-9)
    
    # Candidate 4: Prize Only
    h4 = item_prizes.copy()
    
    # Candidate 5: Inverse Square Sum
    h5 = item_prizes / ((sum_w ** 2) + 1e-9)
    
    # Candidate 6: Prize / (Sum W * Max W)
    h6 = item_prizes / ((sum_w * max_w) + 1e-9)

    candidates = [h1, h2, h3, h4, h5, h6]
    
    best_score = -1
    best_heuristic = h1
    
    for h in candidates:
        # Ensure positive
        h = np.maximum(h, 1e-9)
        
        # Greedy Simulation
        sorted_indices = np.argsort(h)[::-1]
        
        remaining_cap = np.full(n_constraints, capacity)
        total_prize = 0.0
        
        for idx in sorted_indices:
            if np.all(weights[:, idx] <= remaining_cap):
                remaining_cap -= weights[:, idx]
                total_prize += item_prizes[idx]
                
        if total_prize > best_score:
            best_score = total_prize
            best_heuristic = h
            
    # Ensure output is 1D and positive
    best_heuristic = np.maximum(best_heuristic, 1e-9)
    
    return best_heuristic
