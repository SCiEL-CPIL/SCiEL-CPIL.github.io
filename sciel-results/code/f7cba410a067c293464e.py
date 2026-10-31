import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # bins: array of remaining capacities for bins that can fit the item
    # We need to return scores. Higher score is better.
    
    # Calculate remaining capacity if item is placed
    rem = bins - item
    
    # Threshold: 50% of capacity (Capacity is 100)
    # If item is large, prefer Best-Fit (min rem) -> score = -rem
    # If item is small, prefer Worst-Fit (max rem) -> score = rem
    
    if item > 50:
        return -rem
    else:
        return rem
