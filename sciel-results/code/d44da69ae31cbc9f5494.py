import numpy as np

def heuristics(demands, capacity):
    n = len(demands)
    # demands is a float array
    # capacity is float
    
    # Calculate pair sums and gaps
    pair_sum = demands[:, None] + demands[None, :]
    gap = capacity - pair_sum
    
    # Initialize with 1e-9 for all entries
    H = np.full((n, n), 1e-9)
    
    # Mask for valid pairs (gap >= 0)
    mask = gap >= 0
    
    # New Heuristic: Sum-Weighted Inverse Squared Gap
    # Favors pairs that are large (sum) and fit tightly (small gap).
    # The squared gap penalizes loose fits more heavily than linear gap.
    # This encourages "tight packing" of large items, which is crucial for bin packing.
    
    # Avoid division by zero or negative numbers (though mask handles gap>=0)
    # We add 1.0 to the denominator to ensure it's > 0 and to smooth the peak at gap=0
    denom = (gap[mask] ** 2) + 1.0
    numer = pair_sum[mask]
    
    H[mask] = numer / denom
    
    return H
