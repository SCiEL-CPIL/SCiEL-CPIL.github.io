import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Calculate remaining capacity if item is placed
    remainders = bins - item
    
    # Identify valid bins (enough capacity)
    valid = remainders >= 0
    
    # Score: Worst Fit prefers bins with the MOST remaining capacity
    # So we want to maximize 'bins' (or 'remainders' + item, which is bins)
    # For valid bins, score = bins (higher is better)
    # For invalid bins, score = -inf
    
    scores = np.where(valid, bins, -np.inf)
    
    return scores
