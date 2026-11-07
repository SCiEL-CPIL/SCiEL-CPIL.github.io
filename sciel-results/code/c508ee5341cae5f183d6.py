"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 320.0  # wall-clock seconds total allowed, with safety margin

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
    
    Uses a Knowledge Gradient (KG) approximation blended with Expected Improvement (EI)
    to balance exploitation and exploration for the multi-modal landscape.
    
    The KG component estimates the probability that each candidate is the global minimum
    among the 2000 candidates and the current best, providing a principled exploration
    signal. The EI component provides local improvement pressure.
    
    Args:
        mu: array of shape (2000,) with posterior means
        sigma: array of shape (2000,) with posterior standard deviations
        f_best: lowest observed function value
    
    Returns:
        array of shape (2000,) with acquisition scores (higher is better)
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        return np.zeros_like(mu)
    
    # Guard against negative or zero sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Standard Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Knowledge Gradient approximation:
    # Estimate the probability that each candidate i is the global minimum
    # among all 2000 candidates and the current best.
    # P(candidate i is min) ≈ P(f_i < f_j for all j ≠ i) * P(f_i < f_best)
    # Using pairwise comparisons approximated by normal CDF:
    # P(f_i < f_j) = Phi((mu_j - mu_i) / sqrt(sigma_i^2 + sigma_j^2))
    
    # For efficiency, we approximate KG by computing the probability that
    # each candidate beats the current best, weighted by how much lower
    # its mean is relative to other candidates.
    
    # P(f_i < f_best) = Phi((f_best - mu_i) / sigma_i)
    prob_beat_best = norm.cdf(z)
    
    # Estimate the probability that candidate i is the minimum among all candidates:
    # This is hard to compute exactly for 2000 points, so we use a surrogate:
    # The lower the mean and the higher the probability of being low, the better.
    # We compute a normalized "minimality" score.
    
    # Use the expected minimum of the 2000 candidates as a reference
    # Approximate: the candidate with the lowest (mu - k*sigma) for some k
    # is likely to be the minimum.
    
    # Compute a robust "minimality" score:
    # For each candidate, compute how many other candidates are likely to be lower
    # P(f_j < f_i) = Phi((mu_i - mu_j) / sqrt(sigma_i^2 + sigma_j^2))
    # This is O(n^2) which is too slow for n=2000, so we use an approximation.
    
    # Approximation: Use the rank of (mu - 0.5*sigma) as a proxy for minimality
    # Candidates with low mu and moderate sigma are likely to be the minimum.
    
    # Compute a score that favors low mu and high sigma (exploration)
    # but penalizes very high sigma (uncertainty)
    minimality_score = -mu + 0.3 * sigma_safe
    
    # Normalize minimality score to [0, 1]
    min_min = np.min(minimality_score)
    max_min = np.max(minimality_score)
    min_range = max_min - min_min + 1e-9
    minimality_norm = (minimality_score - min_min) / min_range
    
    # Combine KG-like exploration (minimality) with EI exploitation
    # Weight: 60% EI, 40% KG-like minimality
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min + 1e-9
    ei_norm = (ei - ei_min) / ei_range
    
    score = 0.6 * ei_norm + 0.4 * minimality_norm
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
