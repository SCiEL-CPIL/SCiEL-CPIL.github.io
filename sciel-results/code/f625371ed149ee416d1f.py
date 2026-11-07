import numpy as np

def heuristics(prizes, weights):
    # prizes: (5, 100), weights: (5, 100)
    # Calculate total prize for each item
    total_prizes = np.sum(prizes, axis=0)
    
    # Calculate the maximum weight consumption for each item across all constraints
    # This represents the "bottleneck" constraint for that item
    max_weights = np.max(weights, axis=0)
    
    # Heuristic: Value density relative to the most constraining dimension
    # Using max weight in denominator penalizes items that are heavy in any single constraint
    heuristic = total_prizes / (max_weights + 1e-9)
    
    return heuristic
