"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a structural change:
    - Primary signal: rank-normalized Probability of Improvement (PI) per unit cost
    - Refinement: adaptive cost exponent that sharpens the penalty as f_best approaches 0
    - Safety: finite-check and deadline guard
    
    The rank-normalized PI focuses on relative improvement likelihood, which
    proved superior to magnitude-based EI (2.743 vs 1.0). We add an adaptive
    cost exponent to better navigate the Rosenbrock valley by dynamically
    adjusting how aggressively high-cost regions are penalized based on
    convergence progress.
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviations.
        f_best: scalar, the best observed value so far.
        cost: (2000,) array of costs for each candidate.
        
    Returns:
        (2000,) array of scores. Higher is better.
    """
    start_time = time.time()
    
    # Ensure arrays are float
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Prevent division by zero in sigma and cost
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Calculate Z-score for P(f(x) < f_best)
    # Since we are minimizing, improvement is when f(x) is lower than f_best.
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Clip PI to [0, 1] for numerical stability
    pi = np.clip(pi, 0.0, 1.0)
    
    # Adaptive cost exponent: as f_best approaches 0, we want to be more
    # aggressive in avoiding high-cost regions. The exponent increases from
    # 1.0 (linear penalty) to 2.0 (quadratic penalty) as f_best gets small.
    # This helps navigate the Rosenbrock valley by staying in cheaper areas.
    # When f_best is large (early in search), we use linear cost penalty.
    # When f_best is small (near convergence), we use stronger penalty.
    f_best_abs = abs(f_best)
    # Scale: when f_best_abs < 1.0, start increasing exponent
    # When f_best_abs < 0.01, use maximum exponent
    if f_best_abs < 0.01:
        cost_exponent = 2.0
    elif f_best_abs < 1.0:
        # Linear interpolation from 1.0 to 2.0 as f_best goes from 1.0 to 0.01
        cost_exponent = 1.0 + (1.0 - f_best_abs) / 0.99
    else:
        cost_exponent = 1.0
    
    # Apply adaptive cost penalty
    pi_cost = pi / (cost_safe ** cost_exponent)
    
    # Rank-based scoring: rank candidates by pi_cost
    # Higher pi_cost means higher probability of improvement per unit cost
    # Convert to rank-based score to make the scoring robust to scale differences
    ranks = np.argsort(np.argsort(pi_cost))
    n = len(pi_cost)
    
    # Convert ranks to scores in [0, 1]
    score_ranks = ranks.astype(float) / max(n - 1, 1)
    
    # Also compute normalized magnitude for blending
    pi_cost_min = np.min(pi_cost)
    pi_cost_max = np.max(pi_cost)
    if pi_cost_max - pi_cost_min > 1e-12:
        pi_cost_norm = (pi_cost - pi_cost_min) / (pi_cost_max - pi_cost_min)
    else:
        pi_cost_norm = np.zeros_like(pi_cost)
    
    # Combine: 70% rank-based, 30% magnitude-based
    # The rank-based component provides robustness to scale differences
    # The magnitude component preserves relative improvement information
    scores = 0.7 * score_ranks + 0.3 * pi_cost_norm
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 280:
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
