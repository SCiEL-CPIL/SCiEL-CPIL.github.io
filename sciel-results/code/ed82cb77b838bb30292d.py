import numpy as np

def heuristics(prizes, weights):
    # Convert to numpy arrays to ensure consistent operations
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Determine the number of items
    # Weights is expected to be (5, 100) or similar (num_constraints, num_items)
    # Prizes is expected to be (5, 100) or (100,)
    
    # Handle weights shape
    # If weights is 1D, assume it's already per-item? Unlikely given "5 constraints".
    # If weights is 2D, we assume shape is (constraints, items).
    if weights.ndim == 1:
        # Fallback: if 1D, treat as single constraint or already reduced?
        # Task says (5, 100). If we get 1D, it's unexpected. 
        # Let's assume if 1D, it's the max weight per item or similar.
        max_w = weights
    else:
        # Ensure we are looking at the item axis.
        # Standard is (constraints, items). 
        # If shape is (100, 5), we might need to transpose? 
        # The task specifies (5, 100). We will assume axis=1 is items.
        # To be safe, if the second dimension is 5 and first is not, transpose?
        # No, task says (5, 100). We trust the contract but add a small check.
        if weights.shape[0] == 5:
            max_w = weights.max(axis=0)
        elif weights.shape[1] == 5:
            max_w = weights.max(axis=1)
        else:
            # Fallback: max across the first axis
            max_w = weights.max(axis=0)
            
    # Handle prizes shape
    # Task says (5, 100). "Total prize" implies summing across constraints.
    if prizes.ndim == 1:
        total_prizes = prizes
    else:
        if prizes.shape[0] == 5:
            total_prizes = prizes.sum(axis=0)
        elif prizes.shape[1] == 5:
            total_prizes = prizes.sum(axis=1)
        else:
            # Fallback: sum across first axis
            total_prizes = prizes.sum(axis=0)
            
    # Ensure shapes match
    # If max_w and total_prizes have different lengths, there is a fundamental mismatch.
    # But we assume the input is valid per contract.
    
    # Calculate heuristic: Total Prize / Max Weight
    # Add epsilon to avoid division by zero
    epsilon = 1e-9
    eta = total_prizes / (max_w + epsilon)
    
    # Ensure finite and positive
    # Replace non-finite values with a small positive number
    eta = np.where(np.isfinite(eta), eta, 1e-9)
    eta = np.maximum(eta, 1e-9)
    
    # Ensure it is a 1D array
    eta = np.atleast_1d(eta)
    
    return eta
