import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Identify feasible bins
    feasible = bins >= item
    
    # If no feasible bins, return all -inf (or let caller handle, but safer to return -inf)
    if not np.any(feasible):
        return np.full(len(bins), -np.inf)
    
    # Calculate new remainders for feasible bins
    new_rem = bins[feasible] - item
    
    # Sort new remainders
    new_rem_sorted = np.sort(new_rem)
    
    # Count of other bins with remainder strictly smaller
    # searchsorted with side='left' returns the index of the first element >= x
    # which is the count of elements < x
    counts = np.searchsorted(new_rem_sorted, new_rem, side='left')
    
    # Score is negative count
    scores = -counts
    
    # Construct output array
    result = np.full(len(bins), -np.inf)
    result[feasible] = scores
    
    return result
