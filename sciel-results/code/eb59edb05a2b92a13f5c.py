"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 325.0  # Safety margin before the 330s kill limit

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    Uses a Pareto-based multi-objective strategy combining normalized
    Expected Improvement (EI) and normalized predictive uncertainty.
    A dynamic weight increases with the number of evaluations (estimated
    from the spread of mu) to shift from exploration to exploitation.
    
    Args:
        mu: Array of predicted means (shape (2000,)).
        sigma: Array of predicted standard deviations (shape (2000,)).
        f_best: The best observed value so far.
        
    Returns:
        Array of scores (shape (2000,)), higher is better.
    """
    # Check for timeout
    if time.time() - _start_time > _TIMEOUT:
        return np.zeros_like(mu)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-8)
    sigma = np.nan_to_num(sigma, nan=1e-8, posinf=1e8, neginf=1e-8)
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e8, neginf=-1e8)
    
    # Estimate number of evaluations from the posterior structure.
    # Early in optimization, mu has high variance and sigma is large.
    # As more points are added, mu becomes more structured and sigma decreases.
    # We use a heuristic: the ratio of median sigma to the range of mu
    # decreases as we collect more data.
    
    mu_range = np.ptp(mu) if np.ptp(mu) > 1e-10 else 1e-8
    median_sigma = np.median(sigma)
    
    # Heuristic estimate of evaluation count based on uncertainty reduction
    # When few points: sigma is large relative to mu spread
    # When many points: sigma is small relative to mu spread
    uncertainty_ratio = median_sigma / mu_range
    
    # Map uncertainty_ratio to an estimated evaluation count (5 to 30)
    # High ratio -> few evaluations, low ratio -> many evaluations
    est_evals = np.clip(30 - 25 * np.clip(uncertainty_ratio, 0, 1), 5, 30)
    
    # Dynamic weight: starts high (exploration) and decreases (exploitation)
    # Using a smooth transition: w = 1 / (1 + est_evals / 10)
    # At est_evals=5: w = 1/1.5 = 0.67
    # At est_evals=10: w = 1/2 = 0.5
    # At est_evals=20: w = 1/3 = 0.33
    # At est_evals=30: w = 1/4 = 0.25
    w = 1.0 / (1.0 + est_evals / 10.0)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)  # Numerical stability
    
    EI = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    EI = np.maximum(EI, 0)  # EI should be non-negative
    
    # Normalize EI and uncertainty to [0, 1] for the Pareto combination
    ei_min, ei_max = np.min(EI), np.max(EI)
    ei_range = ei_max - ei_min if ei_max - ei_min > 1e-10 else 1e-8
    EI_norm = (EI - ei_min) / ei_range
    
    sig_min, sig_max = np.min(sigma), np.max(sigma)
    sig_range = sig_max - sig_min if sig_max - sig_min > 1e-10 else 1e-8
    sigma_norm = (sigma - sig_min) / sig_range
    
    # Pareto-based score: weighted sum of normalized objectives
    # Higher w means more weight on exploration (uncertainty)
    # Lower w means more weight on exploitation (EI)
    scores = w * EI_norm + (1 - w) * sigma_norm
    
    # Clean up non-finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e8, neginf=-1e8)
    
    return scores
# EVOLVE-BLOCK-END
