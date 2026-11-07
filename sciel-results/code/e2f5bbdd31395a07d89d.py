import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 320.0

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)

    # Hybrid acquisition: blend Expected Improvement (EI) with Thompson Sampling (TS)
    # EI provides a deterministic, cost-aware exploitation signal.
    # TS provides stochastic exploration.
    
    # 1. Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Compute Thompson Sampling sample
    # We use a dynamic kappa to control exploration intensity.
    # Early on (high f_best), we want more exploration.
    # Later (low f_best), we want more exploitation.
    # The current program used kappa = 1 + f_best/50.
    # Inspiration 2 used sqrt scaling. Let's stick to a moderate linear scaling 
    # but cap it to prevent extreme noise.
    # f_best range is roughly [0, 130].
    # If f_best is 100, we want significant exploration.
    # If f_best is 0, we want pure exploitation (kappa=1).
    
    # Let's use a slightly more aggressive exploration than the current program
    # to ensure we don't get stuck in local minima of the rosenbrock function,
    # especially given the shifted minima.
    kappa = 1.0 + f_best / 30.0
    kappa = np.clip(kappa, 1.0, 5.0)
    
    std = kappa * sigma
    y_ts = np.random.normal(mu, std)
    
    # 3. Combine EI and TS
    # We want to maximize a score.
    # EI is positive when mu < f_best.
    # TS sample y_ts: lower is better. So -y_ts is the "score" from TS.
    
    # Normalize EI and -y_ts to comparable scales?
    # EI can be large if f_best is large.
    # -y_ts can be large negative if mu is large.
    
    # A simple convex combination:
    # score = alpha * (EI / cost) + (1 - alpha) * (-y_ts / cost)
    
    # However, EI is already a "regret" measure. 
    # Let's try a weighted sum.
    # When f_best is high, EI values are large. TS values are also large.
    # Let's use alpha = 0.5 for a balanced approach.
    # Or, we can use EI as the primary driver and TS as a perturbation.
    
    # Actually, a common hybrid is:
    # score = EI + lambda * (mu - y_ts)  # not quite
    
    # Let's stick to the cost-aware normalization for both.
    # EI_per_cost = EI / cost
    # TS_per_cost = -y_ts / cost
    
    # The problem with mixing them directly is scale.
    # EI is in objective units. y_ts is in objective units.
    # So they are comparable.
    
    # Let's use a dynamic alpha.
    # If we are early (high f_best), rely more on TS for exploration.
    # If we are late (low f_best), rely more on EI for exploitation.
    
    # f_best > 10: alpha = 0.3 (more TS)
    # f_best < 1: alpha = 0.8 (more EI)
    
    # Linear interpolation
    # alpha = 0.3 + 0.5 * (1 - min(f_best/10, 1))
    # f_best=10 -> alpha=0.3
    # f_best=0 -> alpha=0.8
    
    alpha = 0.3 + 0.5 * (1.0 - min(max(f_best / 10.0, 0.0), 1.0))
    
    ei_score = ei / cost
    ts_score = -y_ts / cost
    
    scores = alpha * ei_score + (1.0 - alpha) * ts_score
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
