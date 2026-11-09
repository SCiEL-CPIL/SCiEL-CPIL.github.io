"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we've used most of the 300s budget, return quickly
    # (the evaluator kills at 330s, so leave margin for overhead)
    # Note: we can't easily track elapsed time across calls without global state,
    # but we can make the computation efficient.
    
    # Use PoI (Probability of Improvement) blended with EI, normalized by cost
    # PoI is more exploratory, reducing risk of local optima in multimodal Shekel 4d
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # EI component
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # PoI component: P(f(x) < f_best) = Phi(z)
    poi = norm.cdf(z)
    
    # Blend: weight PoI more when sigma is large (exploration) and EI when sigma is small
    # Use a simple blend: 0.5 * EI + 0.5 * PoI * (f_best - mu) to keep units consistent
    # Actually, let's use a cleaner approach: EI with a temperature parameter
    # Or simply: use the measured finding that PoI is more exploratory and scored 1.229
    
    # Use PoI-based acquisition: P(improvement) / cost
    # But to maintain units and comparability, use:
    # score = (EI + lambda * PoI * max(0, f_best - mu)) / cost
    # Simpler: just use PoI / cost, which is what the measured improvement used
    
    # Actually, let's use a blend that's been shown to work:
    # The key insight from the finding: PoI is more exploratory, reducing local optima risk
    # Use: (EI * 0.7 + PoI * (f_best - mu) * 0.3) / cost
    # But (f_best - mu) can be negative, so let's be careful
    
    # Safest approach based on the measured result: use PoI normalized by cost
    # PoI = P(f(x) < f_best) which is always in [0,1]
    # This is more exploratory and the measured program scored 1.229
    
    score = poi / cost
    return score
# EVOLVE-BLOCK-END
