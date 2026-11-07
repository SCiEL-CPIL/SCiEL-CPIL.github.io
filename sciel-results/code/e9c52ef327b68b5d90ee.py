"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement with
    adaptive exploration, normalized by evaluation cost.
    """
    start_time = time.time()
    
    # Clamp sigma to avoid numerical issues
    sigma = np.clip(sigma, 1e-9, None)
    
    # Cost is in [1, 2] range (1 + mean of coords scaled to [0,1])
    # Use cost normalization to prefer cheaper evaluations
    
    # Adaptive exploration parameter based on progress
    # f_best starts high and decreases toward 0 (the minimum)
    # Use log-scale to get a smooth decay
    # When f_best is far from 0, explore more; when close, exploit more
    progress = np.clip(1.0 - f_best / max(f_best, 1e-6), 0.0, 1.0)
    
    # Base beta: start aggressive, decay as we improve
    # f_best ranges from ~10+ initially down to ~0
    # Use a soft normalization
    f_best_safe = max(f_best, 1e-6)
    # Beta decays from ~4.0 (early) to ~1.0 (late)
    beta = 1.0 + 3.0 * np.exp(-f_best_safe * 0.5)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    # Use scipy for accuracy
    from scipy.stats import norm
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma * phi
    ei = np.maximum(ei, 0.0)
    
    # Exploration bonus: reward high uncertainty regions
    # Scale by beta and normalize by typical sigma magnitude
    # Use sigma itself as the exploration signal
    exploration = beta * sigma
    
    # Combined acquisition: EI + exploration, normalized by cost
    # Weight: blend EI and exploration
    # Early on: more weight to exploration
    # Late on: more weight to EI
    w_ei = 0.5 + 0.5 * np.exp(-f_best_safe * 0.3)
    w_exp = 1.0 - w_ei
    
    combined = w_ei * ei + w_exp * exploration
    
    # Normalize by cost to prefer cheaper evaluations
    score = combined / cost
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e10, neginf=0.0)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 250.0:
        return score
    
    return score
# EVOLVE-BLOCK-END
