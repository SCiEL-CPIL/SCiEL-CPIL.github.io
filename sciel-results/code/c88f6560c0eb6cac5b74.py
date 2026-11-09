"""Rewritten acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of styblinski tang 1d.
    
    Uses Quantile-based Expected Improvement (QEI) targeting the 95th percentile
    of the posterior predictive distribution. This optimizes for a high-confidence
    lower bound on improvement rather than the mean improvement, which is more
    robust in multimodal landscapes like Styblinski-Tang.
    
    The score combines:
    1. A quantile-based exploitation term: how much better the 95th percentile 
       (optimistic lower bound) is compared to f_best
    2. An exploration bonus: sigma to encourage exploration in uncertain regions
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array_like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far (minimum).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores for each candidate. Higher is better.
    """
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin before 330s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Quantile-based approach: target the 95th percentile of the predictive distribution
    # For minimization, we want low values. The "optimistic" lower bound at 95% confidence
    # is mu - 1.645 * sigma (5th percentile, since we want the lower tail)
    # But for QEI, we consider the expected improvement over the quantile.
    
    # Standard EI calculation for reference
    z = (f_best - mu) / sigma
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Quantile-based exploitation: the 5th percentile (optimistic lower bound)
    # q_05 = mu - 1.645 * sigma
    q_05 = mu - 1.645 * sigma
    
    # Improvement of the quantile over f_best (positive means the quantile is better)
    quantile_improvement = np.maximum(f_best - q_05, 0.0)
    
    # Combine: use a weighted combination of EI and quantile-based improvement
    # The quantile term is more robust to multimodality because it focuses on
    # high-confidence regions where we're likely to find good values
    
    # Normalize both terms to comparable scales
    ei_scale = np.max(ei) if np.max(ei) > 0 else 1.0
    qi_scale = np.max(quantile_improvement) if np.max(quantile_improvement) > 0 else 1.0
    
    ei_norm = ei / ei_scale
    qi_norm = quantile_improvement / qi_scale
    
    # Exploration bonus: sigma encourages visiting uncertain regions
    sigma_scale = np.max(sigma) if np.max(sigma) > 0 else 1.0
    sigma_norm = sigma / sigma_scale
    
    # Weighted combination: prioritize exploitation with some exploration
    # QEI weights the quantile improvement more heavily for robustness
    weights_ei = 0.3
    weights_qi = 0.5
    weights_explore = 0.2
    
    scores = weights_ei * ei_norm + weights_qi * qi_norm + weights_explore * sigma_norm
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
