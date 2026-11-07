import numpy as np

def heuristics(prizes, weights):
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Handle potential shape issues
    if prizes.ndim == 1:
        prizes = prizes.reshape(1, -1)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    
    # Ensure both have the same number of rows
    if prizes.shape[0] == 1 and weights.shape[0] > 1:
        prizes = np.repeat(prizes, weights.shape[0], axis=0)
    elif weights.shape[0] == 1 and prizes.shape[0] > 1:
        weights = np.repeat(weights, prizes.shape[0], axis=0)
        
    # In case shapes are still mismatched in a weird way, try to broadcast or error
    if prizes.shape != weights.shape:
        try:
            prizes, weights = np.broadcast_arrays(prizes, weights)
        except ValueError:
            # Fallback: return ones if shapes are incompatible, though task guarantees (5,100)
            return np.ones(prizes.shape[1] if prizes.ndim > 1 else prizes.shape[0])

    num_constraints, num_items = prizes.shape
    eps = 1e-9
    
    # Calculate ratios: prize / weight
    # We want to avoid division by zero.
    # If weight is 0, ratio is inf (if prize > 0) or nan (if prize == 0).
    # For harmonic mean, we sum 1/ratio = weight / prize.
    # If prize is 0, weight/prize is inf. Harmonic mean becomes 0.
    # This is correct: items with 0 prize in any constraint should be avoided?
    # Actually, if prize is 0 in ONE constraint but high in others, is it bad?
    # The objective is sum of prizes. If prize[c,i] = 0, it contributes 0 to objective for that constraint's "slot" but consumes weight.
    # So yes, it is "bad" relative to items that have prize there.
    
    # Compute inverse ratios: weight / (prize + eps)
    # Using eps in denominator for prize ensures we don't divide by zero, 
    # but if prize is 0, the term is large, making harmonic mean small.
    
    inv_ratios = weights / (prizes + eps)  # shape (num_constraints, num_items)
    
    # Sum inverse ratios across constraints
    sum_inv_ratios = np.sum(inv_ratios, axis=0)  # shape (num_items,)
    
    # Harmonic Mean = N / Sum(1/R)
    heuristic = num_constraints / (sum_inv_ratios + eps)
    
    # Ensure positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
