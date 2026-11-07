"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Ackley 2D.
    Uses a hybrid of Expected Improvement (EI) and Lower Confidence Bound (LCB).
    Score = alpha * EI + (1 - alpha) * LCB_score. Higher score is preferred.
    """
    # Deadline guard: ensure we return well before 330s wall-clock limit
    t_start = time.time()
    deadline = t_start + 280.0  # 50s safety margin

    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)

    # Fallback simple LCB if time is tight or inputs are degenerate
    def fallback_lcb():
        kappa = 1.0
        scores = -mu + kappa * sigma
        return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

    if time.time() > deadline:
        return fallback_lcb()

    n = len(mu)
    if n == 0:
        return np.array([])

    # 1. Compute Expected Improvement (EI) for minimization
    # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x)) / sigma(x)) + sigma(x) * phi((f_best - mu(x)) / sigma(x))
    # We handle sigma == 0 separately to avoid division by zero.
    
    ei_scores = np.zeros(n)
    mask = sigma > 1e-10
    
    if np.any(mask):
        improvement = f_best - mu[mask]
        # Z-score
        z = improvement / sigma[mask]
        
        # Standard Normal CDF and PDF
        # Using scipy.stats.norm for accuracy
        cdf = norm.cdf(z)
        pdf = norm.pdf(z)
        
        ei_scores[mask] = improvement * cdf + sigma[mask] * pdf
    
    # For sigma == 0, EI is simply max(0, f_best - mu)
    if np.any(~mask):
        ei_scores[~mask] = np.maximum(0, f_best - mu[~mask])
    
    # 2. Compute LCB score
    # LCB(x) = mu(x) - kappa * sigma(x)
    # We want to maximize the "negated" LCB, so score_lcb = -mu + kappa * sigma
    kappa = 1.0
    lcb_scores = -mu + kappa * sigma

    # Check deadline before combining
    if time.time() > deadline:
        return fallback_lcb()

    # 3. Combine EI and LCB scores
    # Based on findings, higher alpha (more EI) and lower beta/kappa (less exploration) worked well.
    # Let's use a weighted sum. 
    # To make them comparable, we can normalize them or just weight them directly.
    # Since EI is in objective units and LCB is also in objective units (roughly), direct combination is plausible.
    # However, EI can be zero if mu > f_best, while LCB can be positive.
    # Let's try alpha = 0.9 for EI and 0.1 for LCB to favor exploitation as suggested by the lesson.
    
    alpha = 0.9
    scores = alpha * ei_scores + (1 - alpha) * lcb_scores

    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

    return scores
# EVOLVE-BLOCK-END
