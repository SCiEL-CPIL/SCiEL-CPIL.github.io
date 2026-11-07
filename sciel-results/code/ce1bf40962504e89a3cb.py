import numpy as np

def heuristics(prizes, weights):
    # prizes: (5, 100)
    # weights: (5, 100)
    
    # Calculate total prize per item
    total_prize = np.sum(prizes, axis=0)
    
    # Calculate total weight per item
    total_weight = np.sum(weights, axis=0)
    
    # Calculate the maximum relative weight per item
    # Capacity is 25 for each constraint
    capacity = 25.0
    rel_weights = weights / capacity
    max_rel_weight = np.max(rel_weights, axis=0)
    
    # Penalty factor: 1 + max relative weight
    # This penalizes items that consume a large fraction of any single constraint
    penalty = 1.0 + max_rel_weight
    
    # Heuristic: Total Prize / (Total Weight * Penalty)
    # This favors items with high value density that also don't bottleneck any constraint
    eta = total_prize / (total_weight * penalty + 1e-9)
    
    # Ensure positivity
    eta = np.where(eta > 0, eta, 1e-9)
    
    return eta
