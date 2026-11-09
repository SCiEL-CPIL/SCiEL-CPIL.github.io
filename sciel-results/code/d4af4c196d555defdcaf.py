import numpy as np
from scipy.stats import norm
import time

_start_time = None

def _safe_divide(a, b, eps=1e-9):
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=(np.abs(b) > eps))

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard
    if time.time() - _start_time > 320:
        return np.zeros_like(mu, dtype=np.float64)

    sigma = np.maximum(sigma, 1e-9)
    diff = f_best - mu
    z = _safe_divide(diff, sigma)
    z = np.clip(z, -10, 10)
    
    # Structural change: Use adaptive UCB instead of EI/PI blend
    # UCB is optimistic: it rewards high uncertainty (exploration) directly
    # beta adapts to the current mean sigma - more exploration when uncertain
    mean_sigma = float(np.mean(sigma))
    
    # Beta scales with uncertainty: higher sigma -> more exploration
    # Clip to reasonable range
    beta = 0.5 + 2.0 * np.clip(mean_sigma / (mean_sigma + 0.01), 0, 1)
    
    # UCB: mu + beta * sigma (we want to minimize, so higher UCB = more promising)
    # But we need to convert to an "improvement" score
    # Expected best value at x is approximately mu - beta*sigma (optimistic)
    # Improvement over f_best is f_best - (mu - beta*sigma) = diff + beta*sigma
    ucb_improvement = diff + beta * sigma
    ucb_improvement = np.maximum(ucb_improvement, 0.0)
    
    # Also compute PI as a secondary signal for very uncertain regions
    pi = norm.cdf(z)
    
    # Combine: UCB dominates when there's potential improvement,
    # PI adds weight to uncertain regions even if mu is not great
    score_base = ucb_improvement + 0.1 * pi * sigma
    
    # Cost-aware: divide by cost (linear, not power) to be less aggressive
    score = _safe_divide(score_base, cost, eps=1e-9)
    
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    return score
