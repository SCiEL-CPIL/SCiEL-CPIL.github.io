"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a combination of:
    1. Expected Improvement (EI) for exploitation
    2. Cost-aware scaling with quadratic penalty for low-cost preference
    3. Uncertainty bonus for exploration in underexplored regions
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions
    sigma : array-like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array, shape (2000,)
        Acquisition scores (higher = more promising)
    """
    start_time = time.time()
    deadline = 290.0  # Safety margin before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Clip cost to avoid division by zero or negative values
    cost = np.maximum(cost, 1e-6)
    
    # Compute z-score for Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    diff = f_best - mu
    z = diff / sigma
    
    # Handle numerical stability for large |z|
    z_clipped = np.clip(z, -10, 10)
    
    # Compute EI components
    phi = norm.pdf(z_clipped)
    Phi = norm.cdf(z_clipped)
    
    # Expected Improvement (positive when mu < f_best)
    ei = diff * Phi + sigma * phi
    
    # Ensure EI is non-negative (it should be by definition, but clip for safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scaling: use quadratic cost penalty to strongly prefer low-cost evaluations
    # This allows more evaluations within the fixed budget, improving chance of finding global min
    cost_factor = 1.0 / (cost ** 2)
    
    # Add a small exploration bonus based on uncertainty
    # This helps escape local optima by exploring high-uncertainty regions
    # Scale the bonus by a factor that balances exploration vs exploitation
    exploration_bonus = 0.1 * sigma
    
    # Combine EI and exploration bonus
    base_score = ei + exploration_bonus
    
    # Apply cost-aware scaling
    score = base_score * cost_factor
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Final safety check
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Return a simple fallback if we're running out of time
        return np.maximum(ei / cost, 0.0)
    
    return score
# EVOLVE-BLOCK-END
