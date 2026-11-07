"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidates using Adaptive UCB.
    
    This is a structural change from the previous Thompson Sampling + EI hybrid.
    It uses a deterministic Lower Confidence Bound (LCB) approach, which has
    shown stronger performance on this specific problem (Beale 2D) in previous
    lineages (score ~2.23 vs ~1.95).
    
    The exploration parameter beta is adapted dynamically based on the current
    best observed value (f_best). Since the minimum of Beale is 0, a high f_best
    indicates we are in the exploration phase, requiring larger beta to escape
    local minima and map the valley. As f_best approaches 0, beta decreases to
    favor exploitation of the GP mean.
    """
    t_start = time.time()
    
    # Safety checks for inputs
    sigma = np.maximum(sigma, 1e-8)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    n = len(mu)
    
    # Deadline guard check
    if time.time() - t_start > 250:
        return -mu
    
    # Adaptive Exploration Parameter (Beta)
    # For Beale function, min value is 0.
    # f_best is the current minimum observed.
    # We want high exploration when f_best is large (far from optimum).
    # We want low exploration when f_best is small (close to optimum).
    
    # Base beta for UCB. 
    # Standard UCB often uses sqrt(2 log t). Here we use a function of f_best.
    # log1p ensures stability when f_best is small.
    # Scaling factor 1.5 allows for aggressive exploration early on.
    # We clip f_best to avoid extreme values if the GP is broken, though it shouldn't be.
    f_best_safe = np.clip(f_best, 0.0, 100.0)
    
    # Beta starts around 1.5 * log(1) = 0? No, log1p(0)=0.
    # We need a floor for beta to ensure some exploration even at the start.
    # Let's use: beta = base + scale * log1p(f_best)
    # If f_best is ~10, log1p(10) ~ 2.4.
    # Let's try beta = 1.0 + 0.8 * np.log1p(f_best_safe)
    
    beta = 1.0 + 0.8 * np.log1p(f_best_safe)
    
    # Optional: Normalize sigma to be relative to mu's scale?
    # If mu is around 10 and sigma is 0.1, beta*sigma is small.
    # If mu is around 0.1 and sigma is 0.1, beta*sigma is significant.
    # This is desirable.
    
    # Calculate UCB Score: We want to MINIMIZE mu - beta*sigma.
    # So the acquisition score (to MAXIMIZE) is: -mu + beta*sigma
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Deadline guard
    if time.time() - t_start > 250:
        return -mu
    
    return scores
# EVOLVE-BLOCK-END
