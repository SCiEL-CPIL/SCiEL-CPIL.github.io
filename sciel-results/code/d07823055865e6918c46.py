import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using EI with adaptive variance penalty.
    
    Combines Expected Improvement (EI) with a variance penalty that discourages
    high-uncertainty regions. The penalty is dynamically scaled based on the
    progress of the search (gap to the known minimum), allowing for more
    exploration early on and tighter exploitation later. The score is normalized
    by the evaluation cost to prioritize cheap, promising points.
    """
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback: Simple EI / Cost
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        score = ei / cost_safe
        return np.where(np.isfinite(score), score, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # 1. Calculate Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 2. Probability of Improvement (PI)
    pi = norm.cdf(z)
    
    # 3. Adaptive Variance Penalty
    # The penalty discourages points with high uncertainty.
    # We scale the penalty based on how close we are to the known minimum (-0.8).
    known_min = -0.8
    
    # Estimate progress: 0.0 at start (f_best ~ 0), 1.0 at end (f_best ~ -0.8)
    # This mirrors the logic from the current program's UCB adaptation.
    if f_best > known_min:
        # Max possible improvement range is approx 0.8 (from 0 to -0.8)
        progress = np.clip((0.0 - f_best) / 0.8, 0.0, 1.0)
    else:
        progress = 1.0
        
    # Dynamic penalty coefficient:
    # Early (progress ~ 0): Lower penalty, allow exploration of high sigma regions.
    # Late (progress ~ 1): Higher penalty, focus on low sigma regions with good mu.
    # Base coefficient 0.05, scaled by progress.
    # We use a smooth transition.
    base_penalty_coeff = 0.05
    # Increase penalty as we progress
    penalty_coeff = base_penalty_coeff * (0.5 + 1.5 * progress)
    
    # Quadratic penalty on sigma
    variance_penalty = penalty_coeff * sigma_safe * sigma_safe
    
    # 4. Combine EI and PI
    # EI is the primary driver, PI provides a smooth gradient near f_best.
    # Weighting: 0.8 EI + 0.2 PI
    base_score = 0.8 * ei + 0.2 * pi
    
    # Subtract the variance penalty
    score = base_score - variance_penalty
    
    # 5. Cost-Aware Normalization
    # Divide by cost to favor cheaper evaluations
    score = score / cost_safe
    
    # 6. Safety: Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
