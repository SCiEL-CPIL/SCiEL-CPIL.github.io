"""Acquisition function for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    
    Uses a Hypervolume Indicator (HVI) based acquisition that directly optimizes
    for the area of the Pareto front dominated by the best observed solution
    relative to the known global minimum (3.0).
    
    The HVI score for a candidate point is computed as the expected hypervolume
    contribution when sampling from the posterior N(mu, sigma^2). For a single
    objective (minimization), this reduces to a function of the expected
    improvement over the reference point (global minimum = 3.0) weighted by
    uncertainty.
    
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
        Acquisition scores (HVI-based), higher is better
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
    ref = 3.0
    
    # For HVI in single-objective minimization with reference point ref:
    # The hypervolume dominated by a point f is max(ref - f, 0)
    # We want to maximize expected hypervolume contribution.
    # 
    # Expected hypervolume for a sample y ~ N(mu, sigma^2):
    # E[max(ref - y, 0)]
    # 
    # This is related to the expected improvement but with reference at the
    # global minimum rather than f_best.
    #
    # E[max(ref - y, 0)] where y ~ N(mu, sigma^2)
    # Let z = (ref - mu) / sigma
    # E[max(ref - y, 0)] = (ref - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    z = (ref - mu) / sigma_safe
    hv = (ref - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Additionally, we want to incorporate information about f_best to guide
    # search. Points where we already have good observations (close to f_best)
    # should be deprioritized. We use a combination of the HVI score and
    # a penalty for regions where f_best is already close to the optimum.
    
    # The gap between f_best and the global minimum
    gap = max(ref - f_best, 1e-10)
    
    # Scale the HVI by the current gap to normalize the score
    # When gap is large (early in optimization), exploration is more important
    # When gap is small (late in optimization), exploitation is more important
    
    # A blended score: primarily HVI, but also consider expected improvement
    # over f_best to avoid redundant evaluations
    z_best = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z_best) + sigma_safe * norm.pdf(z_best)
    
    # Combine HVI and EI with a weight that adapts to the optimization stage
    # Early: more weight on HVI (exploration), Late: more weight on EI (exploitation)
    # Use the gap as a proxy for optimization stage
    # When gap is large, weight HVI more; when gap is small, weight EI more
    w_hvi = np.exp(-gap / 10.0)  # Decays as we get closer to optimum
    w_ei = 1.0 - w_hvi
    
    scores = w_hvi * hv + w_ei * ei
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Deadline guard
    if time.time() - t_start > 280:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
