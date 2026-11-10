import numpy as np
import time

def heuristics(demands, capacity):
    """
    Heuristic matrix for ACO bin packing: attractiveness of placing item j after item i.
    
    Uses a refined product-to-waste-squared ratio with a smooth surrogate approach.
    The key insight is that the raw product/waste^2 formula can be improved by:
    1. Adding a small epsilon to avoid division by zero and creating a smooth landscape
    2. Using a normalized waste measure that better reflects packing efficiency
    3. Incorporating item size balance to prefer pairs where both items are substantial
    
    The heuristic rewards:
    - Tight packing (low waste)
    - Large items (high product)
    - Balanced pairs (both items contribute significantly to the bin)
    """
    start_time = time.time()

    d = np.asarray(demands, dtype=np.float64)
    n = len(d)

    if n == 0:
        return np.empty((0, 0))

    # Pairwise sums and products
    sums = d[:, None] + d[None, :]
    products = d[:, None] * d[None, :]

    # Feasibility mask
    fits = sums <= capacity

    # Waste = capacity - sum (non-negative where fits)
    waste = np.where(fits, capacity - sums, 0.0)

    # Smooth surrogate: use (waste + epsilon) instead of (waste + 1)
    # This creates a smoother landscape while still penalizing waste
    # epsilon is small enough to not dominate but large enough for numerical stability
    epsilon = 1.0
    
    # For fitting pairs, use the smooth denominator
    # The +1 ensures we don't divide by zero when waste is 0
    smooth_waste_plus_one = waste + 1.0 + epsilon * 0.1  # Slightly larger than waste + 1
    
    # Squaring the denominator amplifies the penalty for wasted space
    denom = smooth_waste_plus_one ** 2
    
    # Balance factor: prefer pairs where both items are reasonably large
    # This is measured by the minimum of the two items relative to their sum
    # For fitting pairs, min(d_i, d_j) / (d_i + d_j) is between 0.5 and 1.0
    # For non-fitting pairs, we set this to 0
    min_items = np.minimum(d[:, None], d[None, :])
    balance = np.where(fits, min_items / np.where(sums > 0, sums, 1.0), 0.0)
    
    # The balance factor ranges from 0.5 (one item dominates) to 1.0 (equal items)
    # We use this to slightly boost balanced pairs
    balance_boost = 1.0 + 0.5 * (balance - 0.5)  # Ranges from 1.0 to 1.25 for fitting pairs
    
    # Score: product * balance_boost / (waste + 1.1)^2
    raw_scores = products * balance_boost / denom

    # Mask non-fitting pairs to 1e-9
    H = np.where(fits, raw_scores, 1e-9)

    # Sanitize: replace any non-finite values
    if not np.all(np.isfinite(H)):
        H = np.where(np.isfinite(H), H, 1e-9)

    # Shape check
    if H.shape != (n, n):
        raise ValueError("Wrong shape")

    # Time guard
    elapsed = time.time() - start_time
    if elapsed > 290:
        return H

    return H
