"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Ackley 2D.
    
    Uses a combination of:
    1. Expected Improvement per unit cost (baseline)
    2. A bonus for low predicted uncertainty (sigma) to encourage exploitation
    3. A penalty for high cost to encourage exploration in cheap regions
    
    mu, sigma, f_best, cost all have shape (2000,)
    Returns 2000 finite scores (higher = evaluate next)
    """
    # Deadline guard: track elapsed time and return early if needed
    t0 = time.time()
    DEADLINE = 280.0  # safety margin before 300s limit
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Clamp sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware EI: EI per unit cost
    # cost is in range [1, 2] based on the problem description
    cost_safe = np.maximum(cost, 1e-9)
    ei_per_cost = ei / cost_safe
    
    # Add a small exploration bonus for low sigma (certainty)
    # This encourages the algorithm to exploit regions where the GP is confident
    # and the prediction is good
    sigma_bonus = 0.05 * np.exp(-sigma_safe * 5.0)
    
    # Penalize high cost more aggressively
    cost_penalty = 0.1 * (cost_safe - 1.0)
    
    # Combine: EI per cost + sigma bonus - cost penalty
    scores = ei_per_cost + sigma_bonus - cost_penalty
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # If we're running low on time, return immediately
    if time.time() - t0 > DEADLINE:
        return scores
    
    # Optional: apply a soft-max-like scaling to make scores more discriminative
    # but keep them finite and positive-ish
    # Shift scores so the minimum is 0
    min_score = np.min(scores)
    scores = scores - min_score
    
    # Scale to a reasonable range
    max_score = np.max(scores)
    if max_score > 0:
        scores = scores / max_score
    
    return scores
# EVOLVE-BLOCK-END
