import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Styblinski-Tang 2D BO.
    Uses Uncertainty-Weighted Improvement (UWI) with a local baseline.
    """
    t0 = time.time()
    
    # Safety: ensure inputs are clean
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-12), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    known_min = -78.3323
    
    # Progress estimate: 0 at start, 1 near optimum
    # f_best starts around 50-100 for random points, ends near -78.3
    progress = np.clip(1.0 - (f_best - known_min) / 250.0, 0.0, 1.0)
    
    # Deadline guard check early
    if time.time() > _DEADLINE - 5.0:
        # Fallback to simple UCB
        beta_fallback = 2.5
        score = (-mu + beta_fallback * sigma) / cost
        return np.where(np.isfinite(score), score, 0.0)

    # Structural Change: Local Baseline UWI
    # Instead of comparing to global f_best, we estimate a "local best" 
    # using the GP posterior mean minus a confidence width.
    # This makes the acquisition sensitive to local minima structure.
    
    # Adaptive beta for local baseline and exploration
    # Early: high exploration, wide local baseline
    # Late: low exploration, tight local baseline
    beta_local = 1.5 * (1.0 - progress) + 0.5
    beta_explore = 2.5 * (1.0 - progress) + 1.0
    
    # Local baseline: mu - beta_local * sigma
    # This represents "how good could this point be locally?"
    local_baseline = mu - beta_local * sigma
    
    # We want to find points that are:
    # 1. Likely to be better than the local baseline (improvement)
    # 2. Have high uncertainty (exploration)
    # 3. Are cheap
    
    # Compute z-score relative to local baseline
    # P(f(x) < local_baseline) is not directly useful for minimization in the same way.
    # Instead, let's look at the "Expected Improvement" over the local baseline.
    # EI_local = E[max(local_baseline - f(x), 0)]
    # Since f(x) ~ N(mu, sigma^2):
    # EI_local = (local_baseline - mu) * Phi((local_baseline - mu)/sigma) + sigma * phi((local_baseline - mu)/sigma)
    
    z_local = (local_baseline - mu) / (sigma + 1e-12)
    # Note: local_baseline = mu - beta_local * sigma, so z_local = -beta_local
    # This is constant! This means EI_local is proportional to sigma.
    # EI_local = beta_local * sigma * Phi(-beta_local) + sigma * phi(-beta_local)
    # = sigma * [beta_local * Phi(-beta_local) + phi(-beta_local)]
    # This is just a scaled uncertainty.
    
    # To make it more interesting, let's combine a global improvement component
    # with a local exploration component.
    
    # Global EI (standard)
    z_global = (f_best - mu) / (sigma + 1e-12)
    ei_global = (f_best - mu) * norm.cdf(z_global) + sigma * norm.pdf(z_global)
    
    # Local Exploration Bonus: Reward high sigma where mu is already low
    # This targets "boundaries of explored basins"
    # If mu is low (good) and sigma is high (uncertain), this is a great spot.
    # Score = -mu * sigma? No, let's use a weighted sum.
    
    # Let's try a "Weighted Uncertainty Improvement"
    # Score = w1 * EI_global + w2 * (sigma * exp(-mu/scale)) / cost
    # The term (sigma * exp(-mu/scale)) rewards high uncertainty in low-value regions.
    
    # Scale for exponential decay of mu
    # mu ranges from ~100 to -78. 
    # We want exp(-mu/scale) to be significant when mu is low.
    # If mu = -78, we want high weight. If mu = 100, low weight.
    # Let's normalize mu to [0, 1] roughly for the exponent.
    # mu_norm = (mu - mu_max) / (mu_min - mu_max) -> 0 to 1
    # But we don't know mu_max/min globally easily without extra cost.
    # Let's just use a fixed scale.
    # exp(-mu / 50) : mu=0 -> 1, mu=-50 -> e^1=2.7, mu=50 -> e^-1=0.37
    # This seems reasonable.
    
    exploration_term = sigma * np.exp(-mu / 50.0)
    
    # Weights:
    # Early: mix of global EI and local exploration
    # Late: mostly global EI (or even pure exploitation)
    
    w_global = 0.5 + 0.5 * progress
    w_explore = 1.0 - 0.5 * progress
    
    # Combine
    raw_score = w_global * ei_global + w_explore * exploration_term
    
    # Cost adjustment
    # Penalize high cost. 
    # cost = 1 + mean(x_scaled). Range [1, 2].
    # Power penalty: cost^1.2
    score = raw_score / (cost ** 1.2)
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
