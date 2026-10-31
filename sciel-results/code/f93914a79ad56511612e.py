import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Calculate remaining capacity for each bin
    remaining = bins - item
    
    # Initialize scores with -infinity for bins that cannot hold the item
    scores = np.full_like(remaining, -np.inf, dtype=float)
    
    # Mask for bins that have enough capacity
    valid_mask = remaining >= 0
    
    if not np.any(valid_mask):
        return scores
    
    # For valid bins, calculate the adaptive weight
    # We want Best Fit (w=1) for large items and Neutral (w=0) for small items.
    # Sigmoid transition centered around a threshold.
    # Capacity is 100. 
    # Let's set the transition center at 15.0 and steepness k=0.4
    # w = 1 / (1 + exp(-k * (item - center)))
    
    center = 15.0
    k = 0.4
    weight = 1.0 / (1.0 + np.exp(-k * (item - center)))
    
    # Apply the weight to the negative Weibull CDF of the remaining capacity.
    # This creates a non-linear Best Fit strategy that penalizes large remainders
    # based on the probability that they could be filled by future items (inverse logic:
    # we prefer small remainders which have low CDF, i.e., low probability of fitting
    # a *large* item? No, we prefer small remainders because F(r) is small, so -F(r) is close to 0.
    # Wait, if we want to MINIMIZE bins, we want to pack tightly.
    # Best Fit minimizes remaining space.
    # Score = -F(r).
    # If r is small, F(r) is small, Score is close to 0 (High).
    # If r is large, F(r) is large, Score is close to -1 (Low).
    # So we pick the bin with the smallest r. This is Best Fit.
    
    # Weibull parameters
    weibull_scale = 20.0
    weibull_shape = 2.0
    
    # Calculate CDF for valid remainders
    # F(r) = 1 - exp(-(r/scale)^shape)
    # Handle r=0 carefully to avoid issues, though exp(0)=1, F(0)=0.
    r_valid = remaining[valid_mask]
    cdf_values = 1.0 - np.exp(-np.power(r_valid / weibull_scale, weibull_shape))
    
    scores[valid_mask] = -weight * cdf_values
    
    return scores
