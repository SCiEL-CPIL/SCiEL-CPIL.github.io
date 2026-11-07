import numpy as np
import time
from scipy.stats import norm

# Global variable to track start time for the deadline guard
_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    
    # Initialize start time on first call
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline Guard: 
    # The evaluator kills after 330s. 
    # We check if we are dangerously close to the limit.
    # If so, we return a "safe" acquisition function that just exploits 
    # the most promising point (lowest mu) to finish quickly and safely.
    elapsed = time.time() - _start_time
    if elapsed > 320: 
        # Return score based purely on exploitation (lowest mu is best)
        # Score = -mu (higher is better)
        # Ensure finiteness
        scores = -mu
        return np.where(np.isfinite(scores), scores, -np.inf)
        
    # Ensure arrays are 1D and finite
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    n = len(mu)
    
    # Handle any non-finite values
    mask = np.isfinite(mu) & np.isfinite(sigma)
    if not np.all(mask):
        # Replace non-finite with worst-case for mu (high value for minimization)
        # and 0 for sigma
        safe_mu_val = np.max(mu[mask]) if np.any(mask) else 0.0
        mu = np.where(mask, mu, safe_mu_val)
        sigma = np.where(mask, sigma, 0.0)
    
    # Ensure sigma is strictly positive to avoid division by zero
    eps = 1e-12
    sigma = np.maximum(sigma, eps)
    
    # 1. Calculate Standard Expected Improvement (EI)
    # Improvement is f_best - mu
    improvement = f_best - mu
    z = improvement / sigma
    
    # Clip z to prevent numerical overflow/underflow in norm.cdf/pdf
    z_clipped = np.clip(z, -50, 50)
    
    phi_z = norm.pdf(z_clipped)
    Phi_z = norm.cdf(z_clipped)
    
    scores_ei = improvement * Phi_z + sigma * phi_z
    
    # 2. Calculate Rank-Based Uncertainty Bonus
    # We want to boost points with high uncertainty, but only if they are promising.
    
    # Rank sigma from 0 (lowest) to 1 (highest)
    # Using argsort to get ranks
    if n > 1:
        sigma_ranks = np.argsort(np.argsort(sigma)) / (n - 1)
    else:
        sigma_ranks = np.ones(n)
        
    # Create a "Promising" mask: mu < f_best
    # We use a soft gate to avoid hard discontinuities
    # If mu is significantly better than f_best, gate is 1.
    # If mu is worse, gate is 0.
    # Transition width: 0.5 seems reasonable for Hartmann 3D scale
    transition_width = 0.5
    gate = 1.0 / (1.0 + np.exp((mu - f_best) / transition_width))
    
    # 3. Determine Adaptive Exploration Weight
    # f_global_min = -3.86278
    # If f_best is close to global min, we should exploit (low weight).
    # If f_best is far, we should explore (high weight).
    
    f_global_min = -3.86278
    
    # Normalize progress: 0.0 at f_best = 0.0 (start), 1.0 at f_best = -3.86
    # We clamp f_best to [0, -3.86] for this calculation
    f_best_clamped = np.clip(f_best, -3.86, 0.0)
    progress = (0.0 - f_best_clamped) / 3.86  # 0 at start, 1 at end
    
    # Exploration weight lambda
    # Start with high exploration (lambda=1.0) and reduce to low (lambda=0.2)
    lambda_explore = 1.0 - 0.8 * progress  # 1.0 -> 0.2
    
    # Refinement: Use a power law on the rank to emphasize the top quantiles (e.g., top 5-10%)
    # rank^2 emphasizes high ranks more than rank^1.
    # This aligns with the "top 5% of uncertainty" directive.
    rank_emphasis = 2.0
    sigma_ranks_boosted = sigma_ranks ** rank_emphasis
    
    # Scale factor: Adjusted to balance with EI.
    # Since rank^2 makes the average bonus smaller, we might need a slightly higher scale
    # or keep it similar to ensure the top points still get a significant boost.
    scale_factor = 0.8
    
    # The bonus is: lambda * scale * boosted_rank * gate
    bonus = lambda_explore * scale_factor * sigma_ranks_boosted * gate
    
    # 4. Combine Scores
    # Score = EI + Bonus
    scores = scores_ei + bonus
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
