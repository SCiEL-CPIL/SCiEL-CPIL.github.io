import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Computes the acquisition function scores for 2000 candidates.
    
    Parameters:
    mu: np.ndarray, shape (2000,), mean of the GP
    sigma: np.ndarray, shape (2000,), standard deviation of the GP
    f_best: float, the best function value observed so far
    cost: np.ndarray, shape (2000,), cost of evaluating each candidate
    
    Returns:
    scores: np.ndarray, shape (2000,), higher is better
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is non-zero to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Ensure cost is non-zero
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: We are minimizing, so Improvement is f_best - f(x).
    # If mu < f_best, z > 0. Phi(z) is close to 1. EI is large.
    # If mu > f_best, z < 0. Phi(z) is small. EI is small.
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Calculate Upper Confidence Bound (UCB)
    # Since we minimize, we want low mu and high sigma.
    # Standard UCB for minimization: -mu + beta * sigma
    # We want high scores, so we define score_ucb = -mu + beta * sigma
    # However, EI is always >= 0. -mu can be negative.
    # To blend them effectively, we should normalize or use a common scale.
    # A common approach is to use the "Optimistic Value" which is roughly -mu + beta*sigma.
    # Let's use beta = 1.0.
    beta = 1.0
    ucb = -mu + beta * sigma_safe
    
    # Normalize EI and UCB to have similar magnitudes before blending
    # This is a structural choice to balance exploration (UCB) and exploitation (EI)
    # Max of EI is roughly f_best - min(mu). Max of UCB is roughly -min(mu) + beta*max(sigma).
    # Simple normalization by max (with epsilon) helps.
    
    ei_max = np.max(ei)
    ucb_max = np.max(ucb)
    
    # If max is 0 or negative (for EI, max should be positive if we haven't found global min),
    # we use 1.0 as denominator to preserve relative ordering.
    if ei_max > 1e-9:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei
        
    if ucb_max > 1e-9:
        ucb_norm = ucb / ucb_max
    else:
        ucb_norm = ucb
    
    # Blend 50% EI and 50% UCB
    # Weight can be tuned, but 0.5 is a robust starting point based on findings
    w_ei = 0.5
    w_ucb = 0.5
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Apply cost-awareness
    # Higher cost should reduce score
    scores = scores / cost_safe
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
