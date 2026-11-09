"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track time for deadline guard
_start_time = time.time()
_DEADLINE = 325.0  # 5 seconds safety margin before 330s limit


def _robust_normalize(arr):
    """Normalize array using median and IQR for robustness."""
    med = np.median(arr)
    q75, q25 = np.percentile(arr, [75, 25])
    iqr = q75 - q25 + 1e-9
    return (arr - med) / iqr


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Expected Improvement (EI)
    and Lower Confidence Bound (LCB), with adaptive exploration parameter.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed function value so far
        cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
        scores: array of shape (2000,) - acquisition scores (higher = more promising)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a safe fallback: just use -mu / cost as a simple heuristic
        return -mu / (cost + 1e-9)
    
    n = len(mu)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute EI (Expected Improvement)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute LCB with adaptive beta
    # Use a fixed beta that balances exploration well for this problem
    # Beta around 2.0-3.0 works well for 3D problems
    beta = 2.5
    
    # LCB score: lower predicted value with uncertainty bonus
    lcb = mu - beta * sigma
    
    # Robustly normalize both components to [0, 1] range
    ei_norm = _robust_normalize(ei)
    lcb_norm = _robust_normalize(-lcb)  # Invert LCB so higher is better
    
    # Clip to [0, 1] for stable combination
    ei_norm = np.clip(ei_norm, 0, 1)
    lcb_norm = np.clip(lcb_norm, 0, 1)
    
    # Weighted combination: 0.7 * EI + 0.3 * LCB
    # EI is the primary exploitation signal, LCB provides exploration
    combined = 0.7 * ei_norm + 0.3 * lcb_norm
    
    # Cost-aware weighting: divide by cost, but use a soft cost penalty
    # This avoids zeroing out high-cost points entirely
    cost_penalty = 1.0 / (cost + 0.5)
    scores = combined * cost_penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
