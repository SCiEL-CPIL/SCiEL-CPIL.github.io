"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 300  # seconds, with 30s safety margin from 330s limit


def acquisition(mu, sigma, f_best, cost):
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback: return uniform random scores to avoid timeout
        return np.ones(len(mu)) * 0.5
    
    # Probability of Improvement with soft cost penalty
    # PI favors finding ANY improvement, which helps escape local minima on Rosenbrock
    z = (f_best - mu) / (sigma + 1e-9)
    pi = norm.cdf(z)
    
    # Soft cost penalty: higher cost means less exploration
    # cost is cumulative total cost (starts ~5, grows to 30)
    # Normalize cost to [0, 1] range for the penalty
    cost_norm = np.clip(cost / 30.0, 0.0, 1.0)
    
    # Blend: as cost increases, we become more conservative
    # Early: prioritize PI (exploration), Late: more weight on expected improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0)
    
    # Weight between PI and EI based on progress
    # When cost is low, use mostly PI; when high, blend in EI
    alpha = 0.3 + 0.7 * cost_norm  # 0.3 to 1.0
    
    score = alpha * pi + (1 - alpha) * ei / (cost + 1e-9)
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
