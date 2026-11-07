import numpy as np

def heuristics(prizes, weights):
    # Calculate total prize per item
    total_prizes = np.sum(prizes, axis=0)
    
    # Calculate the maximum weight per item across all constraints
    # This identifies the "bottleneck" constraint for each item
    max_weights = np.max(weights, axis=0)
    
    # Normalize by capacity (25) to get relative usage
    # Add epsilon to avoid division by zero
    relative_usage = max_weights / 25.0 + 1e-9
    
    # Heuristic: Value density relative to bottleneck constraint
    # Items with high prize and low bottleneck usage are more attractive
    eta = total_prizes / relative_usage
    
    # Ensure positive values as per contract
    eta = np.maximum(eta, 1e-9)
    
    return eta
