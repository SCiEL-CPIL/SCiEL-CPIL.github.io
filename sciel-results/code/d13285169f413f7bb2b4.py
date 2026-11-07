import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE = 300.0  # seconds


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP standard deviations
        f_best: float, lowest observed value
        cost: (2000,) array of evaluation costs per candidate
    
    Returns:
        (2000,) array of acquisition scores (higher = more promising)
    """
    # Safety margin: return something valid if we're getting close to deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - 5.0:
        # Fall back to a simple cost-aware EI to avoid timeout
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        cost_safe = np.maximum(cost, 1e-9)
        return ei / cost_safe

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of Improvement (for exploration in high-variance regions)
    pi = norm.cdf(z)
    
    # Adaptive weighting: when f_best is large (far from optimum), weight PI more
    # when f_best is small (close to optimum), weight EI more
    # f_best for rastrigin is >= 0, typical range 0 to ~40
    # Scale factor: weight shifts from PI-dominant to EI-dominant as f_best decreases
    scale = f_best / (f_best + 5.0)  # in [0, 1), 1 when f_best large, ~0 when f_best small
    w_pi = 0.3 * scale + 0.1  # PI weight: 0.4 when f_best large, 0.1 when small
    w_ei = 1.0 - w_pi         # EI weight: complementary
    
    # Combined acquisition: weighted mix of EI and PI, divided by cost
    combined = w_ei * ei + w_pi * pi * sigma_safe
    
    # Cost-awareness: divide by cost
    scores = combined / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
