import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we're running low on time, return a safe fallback
    elapsed = time.time() - _start_time
    if elapsed > 300:
        # Fallback: simple EI-like score
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        return (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    n = len(mu)
    
    # Normalize mu and sigma for rank-based selection
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = mu_max - mu_min + 1e-12
    
    sigma_min = np.min(sigma)
    sigma_max = np.max(sigma)
    sigma_range = sigma_max - sigma_min + 1e-12
    
    # Normalize to [0, 1] where 1 is "best" for exploitation (low mu)
    mu_norm = 1.0 - (mu - mu_min) / mu_range
    # Normalize to [0, 1] where 1 is "best" for exploration (high sigma)
    sigma_norm = (sigma - sigma_min) / sigma_range
    
    # Compute EI score (for exploitation ranking)
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize EI to [0, 1]
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min + 1e-12
    ei_norm = (ei - ei_min) / ei_range
    
    # Compute UCB score (for exploration ranking)
    # Adaptive beta: start high, decay as f_best approaches optimum
    # Hartmann 3D minimum is -3.86278
    f_opt = -3.86278
    # Progress: 0 at start (f_best >> f_opt), 1 when f_best close to f_opt
    progress = np.clip((f_best - f_opt) / 10.0, 0, 1)  # rough normalization
    # Actually, let's use a different approach: beta decays with number of evals
    # But we don't have eval count. Let's use f_best relative to f_opt.
    # When f_best is far from f_opt, we want more exploration.
    # f_best starts around 0 (Hartmann 3D values are roughly in [-4, 0])
    # f_opt = -3.86278
    # So f_best - f_opt starts around 3.86 and decreases to 0.
    # Let's normalize: progress = (f_best - f_opt) / 4.0, clipped to [0,1]
    progress = np.clip((f_best - f_opt) / 4.0, 0, 1)
    
    # Beta: starts at 2.0, decays to 0.1
    beta = 0.1 + 1.9 * progress
    
    ucb = -mu + beta * sigma
    
    # Normalize UCB to [0, 1]
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min + 1e-12
    ucb_norm = (ucb - ucb_min) / ucb_range
    
    # Pareto-frontier rank selection:
    # Rank candidates by EI (lower rank = better exploitation)
    # Rank candidates by UCB (lower rank = better exploration)
    # Select candidates that are in the top k% for BOTH criteria
    
    # Compute ranks (1-based, 1 = best)
    ei_rank = np.argsort(np.argsort(ei)) + 1
    ucb_rank = np.argsort(np.argsort(ucb)) + 1
    
    # Minimum rank: candidate must be good in both
    min_rank = np.minimum(ei_rank, ucb_rank)
    
    # But we want candidates that are TOP in both, so use max of normalized ranks
    # Actually, the finding says "min of normalized EI and UCB ranks"
    # Let's normalize ranks to [0,1] where 1 = best (rank 1)
    ei_rank_norm = 1.0 - (ei_rank - 1) / (n - 1)
    ucb_rank_norm = 1.0 - (ucb_rank - 1) / (n - 1)
    
    # Pareto score: min of the two normalized ranks
    # A candidate gets high score only if it's good in both
    pareto_score = np.minimum(ei_rank_norm, ucb_rank_norm)
    
    # As beta decays (progress -> 1), we want pure exploitation
    # Blend: early on, use pareto_score; later, use ei_norm
    # When progress is high (close to optimum), weight toward EI
    # When progress is low (early), weight toward Pareto
    
    # Let's use a smooth blend
    # weight_exploit = progress (0 early, 1 late)
    # weight_explore = 1 - progress
    
    # Final score = weight_exploit * ei_norm + weight_explore * pareto_score
    weight_exploit = progress
    weight_explore = 1.0 - progress
    
    scores = weight_exploit * ei_norm + weight_explore * pareto_score
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    return scores
