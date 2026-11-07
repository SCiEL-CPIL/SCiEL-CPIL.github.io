import numpy as np

def heuristics(prizes, weights):
    # prizes: (5, 100), weights: (5, 100)
    
    # 1. Total prize per item
    total_prizes = np.sum(prizes, axis=0) # (100,)
    
    # 2. Total weight per constraint (to estimate scarcity)
    total_weights_per_constraint = np.sum(weights, axis=1) # (5,)
    
    # Avoid division by zero
    total_weights_per_constraint = np.where(total_weights_per_constraint > 0, total_weights_per_constraint, 1.0)
    
    # 3. Normalized weight contribution
    # weight_normalized[c, i] = weights[c, i] / total_weights_per_constraint[c]
    weight_normalized = weights / total_weights_per_constraint[:, np.newaxis] # (5, 100)
    
    # 4. Total normalized cost per item
    # This represents the item's "share" of the total resource consumption.
    normalized_cost = np.sum(weight_normalized, axis=0) # (100,)
    
    # 5. Heuristic: Prize / Cost
    # Add epsilon to avoid division by zero
    heuristic = total_prizes / (normalized_cost + 1e-9)
    
    # 6. Power scaling to sharpen the heuristic
    # This makes high-value items significantly more attractive than low-value ones
    # helping the ACO converge to better solutions.
    heuristic = heuristic ** 1.5
    
    return heuristic
