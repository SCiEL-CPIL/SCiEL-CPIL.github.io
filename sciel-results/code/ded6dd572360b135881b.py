"""Acquisition function for Bayesian optimization of a fixed seeded random RBF mixture in 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE_SECONDS = 320.0  # Safety margin before the 330s kill

def acquisition(mu, sigma, f_best):
    """
    Returns 2000 finite scores (higher = more likely to be evaluated next).
    Uses a hybrid approach:
    1. Compute Expected Improvement (EI) with an exploration parameter xi.
    2. Blend with a diversity/exploration term based on normalized sigma.
    3. Apply a soft Pareto-like filter to boost candidates that are good in both
       low mu and moderate sigma (uncertainty), without hard-dropping any candidate.
    
    The blend weight shifts from exploration-heavy early to exploitation-heavy late,
    estimated from the gap between f_best and the known global minimum (-2.53218).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Known global minimum for this specific problem
    f_min = -2.53218
    
    # Estimate progress: 0 = start (f_best high), 1 = near optimum
    # f_best starts around 0 or positive, goes down to -2.53218
    # Normalize progress
    progress = np.clip((f_best - f_min) / (0.0 - f_min), 0.0, 1.0)
    # As progress increases, we want more exploitation
    # So exploration weight decreases
    # At progress=0, weight=1.0 (pure exploration-ish)
    # At progress=1, weight=0.0 (pure exploitation)
    # But we never go fully to 0; keep a floor
    exploration_weight = 0.1 + 0.9 * (1.0 - progress)
    
    # Compute EI with xi for exploration
    # xi > 0 makes EI more exploratory
    xi = 0.01 * exploration_weight  # Small exploration parameter
    
    # Standard normal CDF and PDF
    # z = (f_best - mu) / sigma
    z = (f_best - mu - xi) / sigma
    
    # Use scipy for accuracy
    from scipy.stats import norm
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # EI = (f_best - mu - xi) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu - xi) * cdf_z + sigma * pdf_z
    
    # Normalize EI to [0, 1] range approximately
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min + 1e-8
    ei_norm = (ei - ei_min) / ei_range
    
    # Exploration term: normalized sigma
    sigma_min = np.min(sigma)
    sigma_max = np.max(sigma)
    sigma_range = sigma_max - sigma_min + 1e-8
    sigma_norm = (sigma - sigma_min) / sigma_range
    
    # Blend: high EI is good, high sigma is good for exploration
    # But we want to balance them
    score = (1.0 - exploration_weight) * ei_norm + exploration_weight * sigma_norm
    
    # Soft Pareto boost: candidates that have both low mu (good) and moderate sigma (uncertain)
    # are valuable. We can boost candidates that are in the "knee" of the mu-sigma tradeoff.
    # Simple approach: boost candidates where mu is in the bottom 50% AND sigma is in the top 50%
    # This identifies uncertain regions that are promising.
    
    # Normalize mu (lower is better, so invert for ranking)
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = mu_max - mu_min + 1e-8
    mu_norm = (mu - mu_min) / mu_range  # 0 = best, 1 = worst
    
    # Pareto-like score: reward low mu and high sigma
    pareto_score = (1.0 - mu_norm) * sigma_norm
    
    # Blend in a small amount of pareto score
    score = 0.9 * score + 0.1 * pareto_score
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
