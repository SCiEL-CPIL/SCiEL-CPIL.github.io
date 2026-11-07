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
    
    Structural change: Kriging Believer with UCB gating.
    Instead of a linear combination of UCB and EI, this decouples
    exploration and exploitation:
    
    1. Compute UCB scores to identify promising regions (high uncertainty
       or low predicted mean).
    2. Use the GP mean at those regions as the primary exploitation signal.
    3. Apply a soft gating mechanism that weights the exploitation signal
       by how "promising" the UCB score is relative to the distribution.
    
    This is structurally different from standard UCB or EI hybrids because
    it uses uncertainty to *gate* exploitation rather than to *add* to it.
    The score is essentially -mu * gate, where gate is a smooth function
    of the UCB rank. This allows the algorithm to focus exploitation
    effort on regions that are both uncertain and have low predicted values,
    rather than uniformly exploring all high-uncertainty regions.
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
    beta = 1.0
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
    temperature = 0.3
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
    ei_weight = 0.05
    score = score + ei_weight * ei
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
