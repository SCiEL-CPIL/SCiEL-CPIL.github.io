import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_TIME_LIMIT = 295.0  # Safety margin for 300s limit

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback: Greedy exploitation
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is non-negative and cost is positive
    sigma = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # f_best is a scalar (lowest observed value)
    # Rosenbrock range in [-2, 2]^2 is [0, ~130]
    
    # Strategy: Hybrid of Expected Improvement (EI) and Thompson Sampling (TS)
    # EI provides deterministic exploitation signal (probability of improvement weighted by magnitude).
    # TS provides stochastic exploration, scaled by uncertainty.
    
    # 1. Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Compute Thompson Sampling sample
    # We use a dynamic kappa to control exploration intensity.
    # Early on (high f_best), we want more exploration (higher kappa -> wider sampling).
    # Later (low f_best), we want more exploitation (lower kappa -> tighter sampling).
    
    # Linear scaling with cap:
    # f_best=100 -> kappa = 1 + 100/30 = 4.33
    # f_best=10 -> kappa = 1 + 10/30 = 1.33
    # f_best=0 -> kappa = 1.0
    kappa = 1.0 + f_best / 30.0
    kappa = np.clip(kappa, 1.0, 5.0)
    
    std = kappa * sigma
    y_ts = np.random.normal(mu, std)
    
    # 3. Combine EI and TS
    # EI is a "regret" measure (positive when mu < f_best). Higher is better.
    # TS sample y_ts: lower is better. So -y_ts is the "score" from TS. Higher is better.
    
    # Both are in objective units, so they are comparable.
    # We normalize both by cost to make them cost-aware.
    
    ei_score = ei / cost_safe
    ts_score = -y_ts / cost_safe
    
    # Dynamic blending:
    # If we are early (high f_best), rely more on TS for exploration.
    # If we are late (low f_best), rely more on EI for exploitation.
    
    # Linear interpolation of alpha (weight for EI):
    # f_best > 10: alpha = 0.3 (more TS)
    # f_best < 1: alpha = 0.8 (more EI)
    # f_best = 0: alpha = 0.8
    # f_best = 10: alpha = 0.3
    
    # Let's refine the transition.
    # We want to shift from exploration to exploitation as f_best drops.
    # A smooth sigmoid or linear ramp works.
    # Let's use a linear ramp from f_best=15 (alpha=0.2) to f_best=0 (alpha=0.9).
    
    # alpha = 0.9 - 0.7 * (min(f_best, 15) / 15.0)
    # f_best=0 -> alpha = 0.9
    # f_best=15 -> alpha = 0.2
    # f_best>15 -> alpha = 0.2
    
    alpha = 0.9 - 0.7 * (min(max(f_best, 0.0), 15.0) / 15.0)
    
    scores = alpha * ei_score + (1.0 - alpha) * ts_score
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
