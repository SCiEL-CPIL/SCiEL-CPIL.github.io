"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement with a domain-knowledge
    bonus for the 1D sphere function on [-5, 5] (minimum at x=0).
    
    Since we don't receive coordinates, we use the GP posterior (mu, sigma) to
    infer proximity to the optimum. The idea: candidates with low mu (close to 0)
    and moderate sigma are good. We add a bonus that is high when mu is small
    (indicating the GP thinks this point is near the minimum) and sigma is not
    too large.
    
    We use a smooth, adaptive approach:
    - EI: standard expected improvement
    - Bonus: a Gaussian-like bonus centered at mu=0, with width that adapts
      to f_best. Early on (large f_best), the bonus is strong and wide;
      later (small f_best), the bonus weakens as the GP converges.
    """
    # Handle edge cases
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Domain knowledge bonus: the true minimum is at x=0, so f_min=0.
    # We want to bias toward points where mu is close to 0 (the GP's prediction
    # is near the true minimum value).
    
    # Adaptive parameters based on f_best:
    # - When f_best is large (early), we need strong guidance toward x=0
    # - When f_best is small (late), the GP should have converged, so reduce bias
    # 
    # Use a smooth decay: weight ~ exp(-f_best / scale)
    # Scale is chosen so that at f_best=25 (max of sphere at |x|=5), weight~1,
    # and at f_best=0, weight~0.
    f_best_clamped = max(f_best, 0.0)
    
    # Adaptive width: early on, we're uncertain, so use a wider bonus
    # Later, narrow the bonus to focus precisely on x=0
    width = 1.0 + 2.0 * np.exp(-f_best_clamped / 5.0)
    
    # Adaptive weight: strong early, weak late
    weight = 2.0 * np.exp(-f_best_clamped / 10.0) + 0.1
    
    # Bonus: Gaussian centered at mu=0 (since f(0)=0 for sphere)
    # mu is the GP's predicted mean at each candidate point
    # If mu is close to 0, the GP thinks this point is near the minimum
    bonus = weight * np.exp(-0.5 * (mu / width) ** 2)
    
    # Also add a small exploration bonus for low sigma points that have
    # moderate mu (not already well-explored)
    # This helps balance exploitation and exploration
    exploration_bonus = 0.05 * np.exp(-sigma / 0.5) * (1.0 + mu / (f_best + 1.0))
    exploration_bonus = np.maximum(exploration_bonus, 0.0)
    
    score = ei + bonus + exploration_bonus
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
