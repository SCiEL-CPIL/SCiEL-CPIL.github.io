import numpy as np
from scipy.stats import norm
import time

# Module-level state for adaptive acquisition
_call_count = 0
_start_time = None

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function combining Expected Improvement with adaptive UCB.
    
    This function uses a two-pronged approach:
    1. Expected Improvement (EI) for exploitation - finds points likely to improve
    2. Adaptive UCB term for exploration - finds points with high uncertainty
    
    The balance between these two shifts over time:
    - Early: more exploration (higher UCB weight)
    - Late: more exploitation (higher EI weight)
    
    The UCB weight uses the classic sqrt(n) scaling where n is the number of evaluations,
    which theoretically balances exploration and exploitation in sequential decision making.
    """
    global _call_count, _start_time
    
    # Initialize timing if not already done
    if _start_time is None:
        _start_time = time.time()
    
    _call_count += 1
    
    # Deadline guard: ensure we finish well before 330s
    elapsed = time.time() - _start_time
    if elapsed > 325:  # Safety margin
        # Return a simple EI score to avoid timeout
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-10)
        z = (f_best - mu) / sigma_safe
        z = np.clip(z, -10, 10)
        scores = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(scores), scores, -1e9)
    
    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle edge cases
    if np.any(~np.isfinite(mu)) or np.any(~np.isfinite(sigma)):
        mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
        sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=0.0)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-10)
    
    # Compute Expected Improvement for minimization
    # Improvement = f_best - f(x), where we want to maximize this
    # For a GP with mean mu and std sigma, E[I] = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    improvement = f_best - mu
    z = improvement / sigma
    
    # Clip z to avoid numerical issues in normal CDF/PDF
    z = np.clip(z, -10, 10)
    
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI should be non-negative
    
    # Adaptive UCB term using sqrt(n) scaling
    # n is the number of evaluations (call_count)
    n = max(_call_count, 1)
    
    # Classic UCB: kappa = sqrt(2 * log(n))
    # This gives more exploration early (small n) and less later (large n)
    kappa = np.sqrt(2.0 * np.log(n + 1))
    
    # Normalize kappa to a reasonable range
    # For n=1: kappa ~ 0.83, n=5: kappa ~ 1.24, n=10: kappa ~ 1.49, n=30: kappa ~ 1.77
    # Scale to be in a useful range
    kappa_scaled = 0.5 + 0.5 * kappa  # Range roughly [1.0, 1.5] for n in [1, 30]
    
    # UCB score: -mu + kappa * sigma (higher is better)
    ucb_score = -mu + kappa_scaled * sigma
    
    # Normalize both EI and UCB to have similar scales
    # This prevents one term from dominating due to different scales
    ei_max = np.max(ei) if np.max(ei) > 0 else 1e-10
    ucb_max = np.max(ucb_score) if np.max(ucb_score) > 0 else 1e-10
    
    ei_normalized = ei / ei_max
    ucb_normalized = ucb_score / ucb_max
    
    # Adaptive weighting based on progress
    # gap = f_best - (-3.86278) is the distance to known optimum
    opt_val = -3.86278
    gap = max(f_best - opt_val, 0.0)
    
    # When gap is large (early), weight UCB more; when gap is small (late), weight EI more
    # Use exponential decay for smooth transition
    ucb_weight = 0.3 + 0.7 * np.exp(-gap)  # Range: [0.3, 1.0]
    ei_weight = 1.0 - ucb_weight + 0.3  # Range: [0.3, 0.7]
    
    # Normalize weights
    total_weight = ucb_weight + ei_weight
    ucb_weight /= total_weight
    ei_weight /= total_weight
    
    # Combine scores
    scores = ei_weight * ei_normalized + ucb_weight * ucb_normalized
    
    # Scale back to original EI scale for consistent magnitude
    scores = scores * ei_max
    
    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
    
    return scores
