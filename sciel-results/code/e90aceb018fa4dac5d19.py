"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = _start_time + 320.0  # Safety margin before 330s kill


def _check_deadline():
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware multi-scale Expected Improvement (EI) ensemble acquisition function.
    
    This strategy uses a weighted sum of EI values computed with different 'beta' 
    parameters. Standard EI (beta=1) focuses on high-probability improvements. 
    Higher beta values (e.g., 2.0, 3.0) increase the exploration weight, 
    encouraging the search to visit areas with high uncertainty even if the 
    mean prediction is not currently very low. This ensemble approach provides 
    a more robust balance than a single fixed beta, especially in early stages 
    when the GP is uncertain, and late stages when exploitation is key.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
        scores: array of shape (2000,) - acquisition values (higher = better)
    """
    if _check_deadline():
        # Fallback: simple deterministic EI if we're running out of time
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)
    
    # Ensure sigma is strictly positive for numerical stability in division
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Multi-scale EI ensemble
    # We combine EI with different exploration intensities (beta).
    # beta=1.0 is standard EI.
    # beta>1.0 acts like UCB with a different weighting, favoring higher sigma.
    # We use a geometric progression of betas to cover a range of exploration levels.
    
    betas = np.array([0.5, 1.0, 2.0, 4.0])
    weights = np.array([0.1, 0.4, 0.4, 0.1]) # Heuristic weights: emphasize standard EI and moderate exploration
    
    total_scores = np.zeros_like(mu)
    
    for beta, w in zip(betas, weights):
        # For minimization, EI is defined as E[max(0, f_best - f(x))]
        # where f(x) ~ N(mu, sigma^2)
        # However, to create a "multi-scale" effect similar to varying UCB beta,
        # we can adjust the effective uncertainty or use a weighted combination 
        # of the mean and std in the improvement calculation.
        # 
        # A more direct interpretation of "multi-scale EI" in the context of 
        # balancing exploration/exploitation without sampling is to use 
        # the standard EI formula but potentially scale sigma or use 
        # different z-scores. 
        # 
        # Actually, a common robust approach is just standard EI, but the 
        # "multi-scale" inspiration suggests using multiple confidence levels.
        # Let's stick to the standard EI formula which is mathematically 
        # superior to linear UCB in many benchmarks, but ensure we handle 
        # the cost division correctly.
        #
        # Wait, the inspiration says "multi-scale Expected Improvement ensemble".
        # This often refers to using multiple values of the "xi" parameter or 
        # effectively multiple betas in a UCB-like EI formulation.
        #
        # Let's implement EI with a scaled sigma to mimic different exploration levels.
        # If we use sigma_scaled = beta * sigma, we are effectively changing the 
        # confidence level.
        
        sigma_scaled = beta * sigma_safe
        z = (f_best - mu) / sigma_scaled
        # Handle numerical issues in norm.cdf and norm.pdf for very large/small z
        # norm.cdf(z) for z > 8 is ~1, z < -8 is ~0
        # norm.pdf(z) for |z| > 8 is ~0
        
        # To avoid overflow/underflow issues in exp for very large z, we rely on scipy
        # but we should mask extreme values if necessary. For 2000 points, scipy is fast enough.
        
        ei_component = (f_best - mu) * norm.cdf(z) + sigma_scaled * norm.pdf(z)
        
        # Normalize EI by cost
        scores_component = ei_component / cost_safe
        
        total_scores += w * scores_component
    
    # Ensure finite values
    scores = np.where(np.isfinite(total_scores), total_scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
