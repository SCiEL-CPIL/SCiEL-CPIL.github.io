"""Acquisition function for Bayesian optimization of Branin 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for time tracking
_start_time = None

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    
    Structural change: Kriging Believer with UCB region identification.
    
    The core idea is to decouple exploration from exploitation:
    1. Use UCB scores to identify which *regions* of the candidate set are 
       most promising (high uncertainty or low predicted mean).
    2. Within those promising regions, select based on the GP mean (exploitation).
    
    This is implemented by computing a soft "gate" based on UCB rank, then
    multiplying the exploitation signal (-mu) by this gate. This structurally
    differs from standard UCB because the final selection is driven by the
    predicted mean in the most promising uncertain regions, rather than a 
    linear combination of mean and uncertainty.
    
    A small EI component is added for local refinement near the current best.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: return a safe fallback if we're running out of time
    elapsed = time.time() - _start_time
    if elapsed > 325:
        return np.zeros_like(mu)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Step 1: Compute UCB scores for exploration guidance
    # UCB for minimization: -mu + beta * sigma
    # Higher UCB score = more promising for exploration
    # Use a slightly higher beta to encourage exploration in multimodal landscapes
    beta = 1.5
    ucb_scores = -mu + beta * sigma_safe
    
    # Step 2: Normalize UCB scores to [0, 1] for gating
    ucb_min = np.min(ucb_scores)
    ucb_max = np.max(ucb_scores)
    ucb_range = ucb_max - ucb_min
    
    if ucb_range < 1e-10:
        # If all UCB scores are similar, use uniform gating
        ucb_norm = np.ones_like(ucb_scores) / len(ucb_scores)
    else:
        ucb_norm = (ucb_scores - ucb_min) / ucb_range
    
    # Step 3: Create a smooth gating function based on UCB rank
    # Use a sigmoid-like function that gives higher weight to top UCB regions
    # but still allows some exploration of lower-ranked regions
    # temperature controls how sharp the gating is
    # Lower temperature = sharper focus on top UCB regions
    temperature = 0.2
    gate = 1.0 / (1.0 + np.exp(-ucb_norm / temperature))
    
    # Normalize gate to have mean 1 for scale consistency
    gate_mean = np.mean(gate)
    if gate_mean > 1e-10:
        gate = gate / gate_mean
    
    # Step 4: Compute the exploitation signal
    # For minimization, lower mu is better, so -mu is the exploitation score
    exploitation = -mu
    
    # Step 5: Combine exploitation with gating
    # The final score is the exploitation signal weighted by the UCB gate
    # This focuses exploitation on regions that are both promising (low mu)
    # and exploratory (high UCB)
    score = exploitation * gate
    
    # Add a small EI component for local refinement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Scale EI to be comparable and add with small weight
    ei_weight = 0.1
    score = score + ei_weight * ei
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
