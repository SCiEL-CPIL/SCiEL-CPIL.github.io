# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using aggressive Pareto-frontier filtering with pure UCB.
    
    Strategy:
    1. Compute EI for minimization as a secondary signal.
    2. Apply a sharp Pareto filter: only keep candidates where mu is within
       a small tolerance above f_best (using sigma-adaptive threshold).
    3. On the filtered subset, use pure unnormalized UCB score (-mu + beta*sigma)
       to aggressively exploit high-uncertainty regions with plausible improvement.
    4. Candidates outside the Pareto front get a heavily reduced score.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: ensure finite and positive sigma
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # f_best is the lowest observed value
    f_best = float(f_best)
    if not np.isfinite(f_best):
        f_best = 0.0
    
    # Compute Expected Improvement for minimization
    # EI = E[max(f_best - f(x), 0)] where f(x) ~ N(mu, sigma)
    z = (f_best - mu) / sigma
    z = np.clip(z, -10.0, 10.0)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive beta: higher uncertainty early, more exploitation late
    f_best_clamped = np.clip(f_best, 0.0, 2.0)
    beta_ucb = 1.5 + 2.5 * f_best_clamped
    
    # Pure UCB score for minimization: -mu + beta * sigma
    score_ucb = -mu + beta_ucb * sigma
    
    # Sharp Pareto filter: keep candidates where mu is close to f_best
    # Use sigma-adaptive threshold: candidates with mu > f_best + k*sigma are filtered
    # Smaller k means more aggressive filtering
    k_filter = 0.5 + 0.5 * f_best_clamped  # 0.5 to 1.0, more permissive early
    threshold = f_best + k_filter * sigma
    
    # Create mask: True for candidates in the Pareto front (plausible improvement)
    in_front = mu <= threshold
    
    # Apply pure UCB on filtered subset, heavily penalize others
    # Use a large negative penalty for out-of-front candidates
    penalty = -1e4
    score = np.where(in_front, score_ucb, penalty)
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -1e6)
    
    return score
# EVOLVE-BLOCK-END
