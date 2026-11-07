"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Compute a cost-aware acquisition score for 2000 candidates.
    
    Uses a hybrid approach combining:
    1. Cost-aware Expected Improvement (EI)
    2. Cost-aware Probability of Improvement (PI)
    3. A small exploration bonus based on uncertainty
    
    The final score is a weighted combination that balances exploitation
    (finding improvements) with exploration (reducing uncertainty) while
    accounting for evaluation costs.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean predictions from GP
    sigma : array-like, shape (2000,)
        Standard deviation predictions from GP
    f_best : float
        Best observed value so far
    cost : array-like, shape (2000,)
        Evaluation costs for each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is strictly positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Cost-aware EI and PI
    ei_cost = ei / cost
    pi_cost = pi / cost
    
    # Uncertainty bonus: reward points with high sigma relative to the range
    # This encourages exploration in uncertain regions
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-9:
        uncertainty_bonus = (sigma_safe / sigma_max) / cost
    else:
        uncertainty_bonus = np.zeros_like(sigma_safe)
    
    # Combine: primarily use cost-aware EI, with a small weight for PI and uncertainty
    # EI is the most direct measure of expected gain, so give it the highest weight
    # PI provides a more optimistic view, uncertainty bonus encourages exploration
    w_ei = 0.7
    w_pi = 0.2
    w_unc = 0.1
    
    scores = w_ei * ei_cost + w_pi * pi_cost + w_unc * uncertainty_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
