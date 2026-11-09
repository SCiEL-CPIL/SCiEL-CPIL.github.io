"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: 330s wall-clock limit, return safely before it.
    t0 = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero or near-zero sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Bayesian Regret Minimization:
    # Approximate E[min(f(x), f_best)] where f(x) ~ N(mu, sigma^2)
    # E[min(f, f_best)] = f_best - E[max(f_best - f, 0)]
    # For f ~ N(mu, sigma^2):
    #   E[max(f_best - f, 0)] = (f_best - mu) * Phi(z) + sigma * phi(z)
    #   where z = (f_best - mu) / sigma
    # So E[min(f, f_best)] = f_best - [(f_best - mu) * Phi(z) + sigma * phi(z)]
    #
    # The regret is E[min(f(x), f_best)] - f_min, but since f_min is unknown,
    # we minimize E[min(f(x), f_best)] directly.
    # Score = -E[min(f(x), f_best)] / cost (higher score = better)
    
    z = (f_best - mu) / sigma_safe
    
    # Expected improvement (this is E[max(f_best - f, 0)])
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # E[min(f, f_best)] = f_best - ei
    # We want to MINIMIZE this, so score = -(f_best - ei) / cost = (ei - f_best) / cost
    # But this can be negative. Let's use: score = ei / cost
    # This is equivalent to maximizing expected improvement per unit cost
    
    # Actually, let's think more carefully:
    # Simple regret = f_best_observed - f_min
    # We want to minimize this.
    # E[simple regret after evaluating x] = E[f_best_observed] - f_min
    # If we evaluate x, new f_best = min(f_best, f(x))
    # E[min(f_best, f(x))] = f_best - E[max(f_best - f(x), 0)] = f_best - ei
    # So minimizing E[min(f_best, f(x))] is equivalent to maximizing ei
    # Score = ei / cost
    
    scores = ei / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time: if we've used more than 325s, return immediately
    if time.time() - t0 > 325:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
