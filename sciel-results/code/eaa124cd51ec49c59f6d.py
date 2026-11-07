"""Cost-aware acquisition for Bayesian optimization on powell 4d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 295.0  # 5s safety margin before 300s limit

# Precompute constants
# x is in [-4, 5]^4, scaled to [0, 1]: (x - (-4)) / (5 - (-4)) = (x + 4) / 9
# Cost = 1 + mean(x_scaled) where x_scaled = (x + 4) / 9
# We don't have x directly, but cost is given.

# For the acquisition function, we use a Knowledge Gradient (KG) style
# acquisition that estimates the expected improvement in the global optimum,
# scaled by a sublinear cost discount.

def acquisition(mu, sigma, f_best, cost):
    """
    Knowledge Gradient cost-aware acquisition function.
    
    Parameters:
    - mu: (2000,) array of GP mean predictions
    - sigma: (2000,) array of GP standard deviations
    - f_best: float, best observed value
    - cost: (2000,) array of evaluation costs
    
    Returns:
    - (2000,) array of acquisition scores (higher = more desirable)
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(cost)):
        # Return a valid fallback: use negative predicted value (exploit)
        return -mu
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-10)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-10)
    
    # Knowledge Gradient approximation:
    # The KG estimates E[max(f_best - Y, 0)] where Y ~ N(mu, sigma^2)
    # This is exactly the Expected Improvement (EI) formula.
    # However, we use a sublinear cost discount (1/sqrt(cost)) instead of 1/cost
    # to better balance exploration of high-cost regions.
    
    # Expected Improvement (which is the KG for a single candidate vs current best)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Sublinear cost discount: 1/sqrt(cost)
    # This penalizes high-cost evaluations less aggressively than 1/cost,
    # allowing more exploration in expensive regions where uncertainty may be high.
    cost_discount = 1.0 / np.sqrt(cost)
    
    # Add a small exploration bonus to prevent premature convergence
    # The bonus is scaled by the same cost discount
    exploration_bonus = 0.05 * sigma * cost_discount
    
    # Combine
    score = ei * cost_discount + exploration_bonus
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
