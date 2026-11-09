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
        return sigma_safe / cost_safe
    
    # Structural Change: Hybrid POI + EI with Cost Awareness
    # Instead of pure EI, we blend Probability of Improvement (POI) and EI.
    # POI is more robust to scale and emphasizes the likelihood of improvement.
    # EI emphasizes the magnitude of improvement.
    # We also add a variance-based exploration term (PDF of Z) to encourage
    # exploration in regions where the mean is close to f_best but uncertainty is high.
    
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid extreme values in CDF/PDF calculations if necessary, 
    # though norm.cdf/pdf handles large values well (returning 0 or 1).
    # However, for numerical stability in some edge cases:
    z_clipped = np.clip(z, -10, 10)
    
    poi = norm.cdf(z_clipped)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0)
    
    # Normalize EI to be comparable to POI (which is in [0,1])
    # Scale EI by a factor related to the current f_best or just use a fixed scale
    # Since f_best can be close to 0, we use a small epsilon for scaling if we divide.
    # Alternatively, just weight them.
    # Let's use a weighted sum: 0.5 * POI + 0.5 * (EI / (abs(f_best) + 1e-6))
    # This makes EI scale-invariant relative to the current best.
    
    scale = np.abs(f_best) + 1e-6
    ei_normalized = ei / scale
    
    # Exploration term: PDF of Z is high when z is close to 0 (mu ~ f_best).
    # This encourages looking at points where we are "on the edge" of improvement.
    exploration_term = norm.pdf(z_clipped)
    
    # Combine
    # Weight POI and Normalized EI equally, and add a smaller exploration bonus
    raw_score = 0.5 * poi + 0.5 * ei_normalized + 0.1 * exploration_term
    
    # Cost-aware: Divide by cost to prefer cheaper evaluations
    score = raw_score / cost_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
