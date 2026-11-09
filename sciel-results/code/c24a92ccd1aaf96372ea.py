"""Cost-aware acquisition for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Wall-clock guard: the evaluator kills the program at 330s.
# Keep a safety margin so we never exceed the limit.
_T0 = time.time()
_DEADLINE = _T0 + 290.0  # 40s safety margin before the 330s kill
_EPS = 1e-9

def _time_left():
    return _DEADLINE - time.time()

def acquisition(mu, sigma, f_best, cost):
    # Expected Improvement (EI) based acquisition with cost-aware exploration.
    # 
    # Strategy:
    # 1. Compute Expected Improvement (EI). This is superior to PI because it 
    #    accounts for the *magnitude* of the improvement, not just the probability.
    # 2. The task asks for "cost-aware" acquisition. The cost is defined as 
    #    1 + mean(x). Since we don't have x, we use the `cost` array provided.
    # 3. Instead of dividing EI by cost (which is a simple efficiency ratio and 
    #    can be unstable), we use a "Cost-Aware Exploration" strategy.
    #    - Early in the search (low total cost), we might want to explore more.
    #    - However, the `cost` input here is the cost of *each candidate*.
    #    - A robust cost-aware strategy is to subtract a penalty proportional to 
    #      the cost, or to scale the exploration term.
    #    
    #    Let's look at the inspirations. Inspiration 1 (score 1.87) blended 
    #    normalized EI and PI. 
    #    
    #    Let's try a pure EI with a small exploration bonus that is *discounted* 
    #    by cost. 
    #    
    #    EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    #    where z = (f_best - mu) / sigma
    #    
    #    We want to penalize high-cost candidates. 
    #    Score = EI - lambda * cost?
    #    Or Score = EI / (1 + alpha * (cost - min_cost))?
    #    
    #    The current program uses PI / cost. 
    #    Let's try EI with a cost penalty.
    #    
    #    Note: mu and sigma are in objective units.
    #    
    #    To make it robust:
    #    1. Compute EI.
    #    2. Add a small exploration bonus based on sigma (uncertainty).
    #    3. Apply a cost penalty.
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Guard: if we are about to run out of wall-clock, return a safe finite
    # result immediately.
    if _time_left() < 0.5:
        # Fallback: cheap points are better if we are running out of time?
        # Or just return zeros? 
        # Let's return a safe score that prefers low cost.
        return np.where(cost > _EPS, 1.0 / (cost + _EPS), 0.0)

    # Calculate z
    z = (f_best - mu) / (sigma + _EPS)
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If mu > f_best, z is negative. Phi(z) is small, phi(z) is small.
    # If mu < f_best, z is positive. Phi(z) is large, phi(z) is large.
    
    # Use scipy.stats.norm for Phi and phi
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure EI is non-negative (numerical errors can cause tiny negatives)
    ei = np.maximum(ei, 0.0)
    
    # Exploration bonus
    # We want to explore when sigma is high.
    # Normalize sigma to [0, 1] roughly?
    max_sigma = float(np.max(sigma)) if sigma.size else 1.0
    if not np.isfinite(max_sigma) or max_sigma <= _EPS:
        max_sigma = _EPS
        
    exploration_bonus = 0.1 * (sigma / max_sigma)
    
    # Cost-awareness
    # The cost is 1 + mean(x). 
    # We want to penalize higher costs.
    # A simple penalty: subtract a small fraction of the cost.
    # However, EI is in objective units (hartmann 3d min is ~-3.86, range ~10).
    # Cost is in [1, 2].
    # A penalty of 0.1 * cost might be too small or too large depending on the scale of EI.
    # Let's normalize EI? No, EI magnitude matters.
    
    # Let's try: Score = EI - 0.05 * (cost - min_cost)
    # Or simply: Score = EI / (1 + 0.1 * cost)
    
    # Let's stick to a multiplicative penalty which is scale-invariant-ish.
    # Score = EI * (1 / (1 + 0.2 * cost))
    
    # Let's test the "Efficiency" ratio: EI / cost.
    # This directly maximizes improvement per unit cost.
    
    safe_cost = np.where(cost > _EPS, cost, 1.0)
    
    # Combine EI and exploration
    base_score = ei + exploration_bonus
    
    # Apply cost penalty
    score = base_score / safe_cost
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
