import numpy as np

def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness for placing item j right after item i.
    Larger values are more attractive.
    """
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    # Vectorized pair sums
    sums = demands[:, None] + demands[None, :]
    
    # Check which pairs fit in the bin
    fits = sums <= capacity
    
    # Calculate waste (remaining space)
    waste = capacity - sums
    
    # We want to prioritize pairs with low waste.
    # Heuristic: 1 / (waste + 1)^2
    # +1 prevents division by zero and ensures positive denominator.
    # For pairs that don't fit, we set a very low value (1e-9).
    
    # To safely compute the power without issues from negative waste in the 'else' branch
    # of np.where (which evaluates both), we use a masked approach or safe values.
    
    # Create a safe waste array where non-fitting pairs have a large waste (or 1)
    # so they don't interfere, though they will be overwritten by 1e-9 anyway.
    safe_waste = np.where(fits, waste + 1.0, 1.0)
    
    # Calculate the heuristic values
    h = 1.0 / (safe_waste ** 2)
    
    # Apply the mask: if it doesn't fit, set to 1e-9
    h = np.where(fits, h, 1e-9)
    
    return h
