# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Compute acquisition scores for 2000 candidates using Expected Improvement (EI).
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviations.
        f_best: float, lowest observed value.
        cost: (2000,) array of evaluation costs.
    
    Returns:
        (2000,) array of finite scores (higher = evaluate next).
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero and numerical issues
    cost_safe = np.maximum(cost, 1e-9)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement (EI) for minimization
    # Improvement is defined as f_best - f(x). We want to maximize expected improvement.
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    
    # To avoid numerical issues when sigma is very small, we can clamp or handle separately.
    # However, standard EI formula works well if sigma > 0.
    
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid extreme values in normal CDF/PDF which might underflow/overflow or lose precision
    # Phi(z) -> 0 for z << -inf, Phi(z) -> 1 for z >> inf
    # phi(z) -> 0 for |z| >> inf
    # We can just compute directly.
    
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # EI is always >= 0.
    # We want to maximize EI per unit cost.
    score = ei / cost_safe
    
    # Ensure finiteness
    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
