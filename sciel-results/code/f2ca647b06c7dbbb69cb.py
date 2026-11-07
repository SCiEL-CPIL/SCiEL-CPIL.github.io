import numpy as np

def heuristics(prizes, weights):
    # prizes and weights are shape (5, 100)
    # Total prize per item
    total_prize = np.sum(prizes, axis=0)
    
    # Weight metrics
    sum_weights = np.sum(weights, axis=0)
    max_weights = np.max(weights, axis=0)
    
    # Hybrid denominator: penalizes both total weight and bottleneck weight
    # This favors items that are light in all constraints (low max) 
    # and light in total (low sum)
    denom = sum_weights + max_weights + 1e-9
    
    return total_prize / denom
