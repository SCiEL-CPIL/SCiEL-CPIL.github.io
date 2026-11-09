"""Cost-aware acquisition function for Bayesian optimization (powell 4d)"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_T0 = [None]

def acquisition(mu, sigma, f_best, cost):
    if _T0[0] is None:
        _T0[0] = time.time()
    
    elapsed = time.time() - _T0[0]
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Deadline guard: if we're close to the limit, just return a simple score
    if elapsed > 315.0:
        score = sigma_safe / cost_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    # Structural Change: Cost-normalized Probability of Improvement (POI) 
    # combined with a deterministic regret lower bound term.
    # This replaces the non-linear EI with a simpler, more robust score that 
    # directly optimizes expected reduction in simple regret per unit cost.
    
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid extreme values
    z_clipped = np.clip(z, -10, 10)
    
    # Probability of Improvement
    poi = norm.cdf(z_clipped)
    
    # Deterministic regret lower bound: (f_best - mu + sigma * 0.5)
    # This term encourages points where the mean is low (good) and 
    # uncertainty is moderate (not too high, not too low).
    # The 0.5 factor balances exploitation (low mu) and exploration (moderate sigma).
    regret_bound = f_best - mu + sigma_safe * 0.5
    
    # Combine POI and regret bound
    # POI is in [0,1], regret_bound can be negative or positive.
    # We want to prefer points with high POI and high regret_bound (i.e., likely to beat f_best by a lot).
    # However, regret_bound can be negative if mu > f_best + 0.5*sigma.
    # Let's normalize or scale appropriately.
    
    # A simpler combination: weighted sum of POI and a scaled regret bound.
    # The regret bound has units of the objective function, POI is dimensionless.
    # We can scale the regret bound by a characteristic scale of the problem.
    # Since we don't know the scale, we can use f_best as a proxy if it's non-zero,
    # or just use a fixed weight.
    
    # Alternative: Use the formula directly as suggested in the directive:
    # score = (f_best - mu + sigma * 0.5) / cost
    # But this doesn't include the probability aspect.
    
    # Let's try a hybrid: POI * (regret_bound / scale)
    # Or simply: POI + 0.1 * (regret_bound / (abs(f_best) + 1e-6))
    
    # Let's stick to the directive's suggestion but refine it.
    # The directive suggests: (f_best - mu + sigma * 0.5) / cost
    # This is a linear approximation of EI. It is simpler and faster.
    # Let's use this as the base and add a small POI term to ensure we don't ignore probability.
    
    base_score = (f_best - mu + sigma_safe * 0.5) / cost_safe
    
    # Add a small POI term to break ties and encourage high-probability improvements
    poi_term = 0.01 * poi / cost_safe
    
    score = base_score + poi_term
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
