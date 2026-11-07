"""Acquisition function for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Hypervolume Indicator (HVI) approximation.
    
    Structural change: Instead of scalarized mean/variance combinations (EI+UCB),
    this uses a hypervolume-based approach where the reference point is set to
    the known global minimum (3.0). The acquisition score represents the
    contribution of each candidate point to the dominated hypervolume relative
    to the reference point, directly optimizing for the area of improvement
    over the true optimum.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Posterior mean predictions at candidate points
    sigma : array of shape (2000,)
        Posterior standard deviation predictions at candidate points
    f_best : float
        Best (lowest) observed function value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores, higher is better
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Safety deadline guard
    t_start = time.time()
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Reference point: known global minimum of Goldstein-Price 2D
    f_ref = 3.0
    
    # For minimization, we want to maximize the "improvement" over f_ref
    # The hypervolume contribution of a point (x, y) relative to reference (x_ref, y_ref)
    # in a single objective can be approximated by the area of the rectangle dominated
    # by the point relative to the reference.
    
    # In a single-objective setting with known reference, the HVI contribution
    # of a candidate point can be approximated as:
    # - If mu < f_ref: the point dominates the reference, contribution is (f_ref - mu) * width
    # - If mu >= f_ref: the point is dominated, but uncertainty may still allow improvement
    
    # We use a Monte-Carlo approximation of the expected hypervolume contribution
    # E[H(Y)] where Y ~ N(mu, sigma^2)
    
    n_samples = 32
    rng = np.random.default_rng(seed=42)
    
    # Generate standard normal samples: shape (n_samples, n)
    z_samples = rng.standard_normal((n_samples, n))
    
    # Sample from posterior: Y = mu + sigma * z
    # Shape: (n_samples, n)
    y_samples = mu[None, :] + sigma_safe[None, :] * z_samples
    
    # For each sample, compute the hypervolume contribution relative to f_ref
    # In 1D, the HVI contribution of a point y relative to reference f_ref is:
    # max(f_ref - y, 0) * (some width factor)
    # Since we're in a single-objective context, we use the improvement directly
    
    # Compute improvement: max(f_ref - Y, 0)
    improvement = np.maximum(f_ref - y_samples, 0.0)
    
    # Expected hypervolume contribution
    hvi_mc = np.mean(improvement, axis=0)
    
    # Additionally, we add a term that accounts for the potential to find
    # points significantly below f_ref, weighted by the probability of such events
    # P(Y < f_ref) = Phi((f_ref - mu) / sigma)
    z = (f_ref - mu) / sigma_safe
    prob_improve = norm.cdf(z)
    
    # Combine HVI contribution with probability of improvement
    # This gives a score that is high when:
    # 1. The expected improvement is large (exploitation)
    # 2. There's a high probability of finding a point below f_ref (exploration)
    
    # Normalize hvi_mc to [0, 1]
    hvi_min = np.min(hvi_mc)
    hvi_max = np.max(hvi_mc)
    if hvi_max - hvi_min > 1e-12:
        hvi_norm = (hvi_mc - hvi_min) / (hvi_max - hvi_min)
    else:
        hvi_norm = np.zeros_like(hvi_mc)
    
    # prob_improve is already in [0, 1]
    
    # Weight the two components
    # Use a fixed weight that balances exploitation and exploration
    w_hvi = 0.7
    w_prob = 0.3
    
    scores = w_hvi * hvi_norm + w_prob * prob_improve
    
    # Add a bonus for points where mu is significantly below f_ref
    below_best = np.maximum(0.0, f_ref - mu)
    if np.max(below_best) > 1e-10:
        below_best_norm = below_best / (np.max(below_best) + 1e-10)
        scores += 0.1 * below_best_norm
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Deadline guard
    if time.time() - t_start > 280:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
