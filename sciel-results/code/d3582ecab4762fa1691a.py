"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.special

_DEADLINE = 140.0  # safety margin before 150s kill
_START_TIME = time.time()

# Weibull distribution parameters
# The task mentions Weibull sequences. Standard assumptions for capacity 100:
# Shape k=2.0, Scale lambda=25.0 often yields a mean around 22 and median around 19-20.
# We use the median as the "target remainder" or "equilibrium gap".
# If a bin has a remainder close to the median item size, it is statistically
# well-positioned to accept the next item without creating excessive fragmentation
# or leaving large unused chunks that could have been used for a larger item.
k = 2.0
lambda_scale = 25.0

# Calculate the median of the Weibull distribution
# Median = lambda * (ln(2))^(1/k)
target_remainder = lambda_scale * (np.log(2)) ** (1.0 / k)

# Pre-compute constants for the control law
# We want to minimize (resulting_remainder - target_remainder)^2
# resulting_remainder = bins - item
# score = - (bins - item - target_remainder)^2
# Expanding: - (bins^2 - 2*bins*item + item^2 - 2*bins*target + 2*item*target - target^2)
# Since item is constant for all bins in a single call, terms depending only on item
# are constant and don't affect the argmax. However, for clarity and numerical stability
# we compute the full squared error.
# Note: We are maximizing the score, so we negate the squared error.

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE:
        valid = bins >= item
        if not np.any(valid):
            return np.full_like(bins, -np.inf)
        # Fallback to Best Fit if deadline is close
        scores = -(bins - item)
        scores[~valid] = -np.inf
        return scores

    # Determine which bins have enough remaining capacity
    valid = bins >= item
    
    if not np.any(valid):
        return np.full_like(bins, -np.inf)
        
    # Calculate the resulting remainder for each valid bin
    # For invalid bins, we set remainder to 0 or a large negative to mask them out later
    # Actually, we only care about valid bins for the score calculation.
    remainders = bins - item
    
    # Control Law: Minimize squared deviation from target remainder
    # deviation = remainders - target_remainder
    # score = -deviation^2
    
    # We apply this only to valid bins. For invalid bins, score is -inf.
    scores = np.full_like(bins, -np.inf)
    
    if np.any(valid):
        dev = remainders[valid] - target_remainder
        scores[valid] = -(dev * dev)
        
    return scores
# EVOLVE-BLOCK-END
