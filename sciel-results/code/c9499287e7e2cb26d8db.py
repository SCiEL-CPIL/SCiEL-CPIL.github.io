"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    and Upper Confidence Bound (UCB) for minimization.
    
    Args:
        mu: Array of shape (N,) with GP mean predictions.
        sigma: Array of shape (N,) with GP standard deviations.
        f_best: The best observed function value so far.
        cost: Array of shape (N,) with evaluation costs for each candidate.
        
    Returns:
        Array of shape (N,) with acquisition scores (higher is better).
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Calculate Expected Improvement (EI) for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Calculate Upper Confidence Bound (UCB) for minimization
    # We want to maximize this score.
    # Low mu (good) and high sigma (uncertain) should yield high score.
    # Score_ucb = -mu + kappa * sigma
    kappa = 3.0  # Exploration constant
    ucb = -mu + kappa * sigma_safe
    
    # 3. Combine EI and UCB
    # Weighting: UCB is often better for exploration in early stages,
    # EI is better for exploitation in later stages.
    # A fixed weight of 0.2 for EI and 0.8 for UCB (scaled) might be good.
    # However, EI and UCB have different scales.
    # Let's normalize or just use a simple sum.
    # To make them comparable, we can weight them.
    # Let's try a hybrid where UCB is dominant but EI provides a nudge.
    
    # Alternative: Use EI / Cost directly but add an exploration term?
    # Let's stick to the structural change: UCB-based.
    
    # Score = (w_ei * EI + w_ucb * UCB) / Cost
    # We need to balance the magnitudes.
    # Typically, |mu| and f_best are similar magnitude.
    # EI is usually smaller than |mu|.
    # Let's use a weighted sum where UCB is the primary driver.
    
    score_numerator = 0.2 * ei + 0.8 * ucb
    
    # Normalize by cost
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = score_numerator / cost_safe
    
    # Handle any NaNs or Infs
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
