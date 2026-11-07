"""Cost-aware Thompson Sampling acquisition for Hartmann 3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_start_time = time.time()
DEADLINE = 330.0
SAFETY_MARGIN = 5.0  # seconds

def _check_deadline():
    return (time.time() - _start_time) > (DEADLINE - SAFETY_MARGIN)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Strategy:
    1. Compute the probability that a new sample x improves upon f_best, 
       weighted by the inverse of the cost.
    2. To make it a proper acquisition score (higher is better), we use 
       a "soft" improvement score: 
       score = P(improve) * (f_best - mu) / cost_eff
       However, standard TS samples from the posterior. Since we only have 
       mu and sigma, we can approximate the expected value of the 
       "improvement per cost" in a way that favors high uncertainty 
       (exploration) when cost is low.
       
    Alternative Robust Approach:
    Use a cost-aware Upper Confidence Bound (UCB) which is deterministic 
    and often more stable than TS in BO loops without re-sampling overhead.
    
    Let's stick to a refined Cost-Aware UCB (CA-UCB) as it is deterministic 
    and computationally cheap, but with a different scaling than the current 
    EI/sqrt(cost).
    
    The current program uses EI / sqrt(cost).
    The new strategy will use:
    score = (f_best - mu) + beta * sigma / cost
    
    This is a linear combination of exploitation (f_best - mu) and 
    exploration (sigma), where exploration is penalized by cost.
    This is fundamentally different from EI which is a non-linear 
    probabilistic integral.
    """
    if _check_deadline():
        # Fallback: simple EI / cost to ensure we return something fast
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)

    # Clip sigma to avoid division issues and extreme values
    sigma = np.maximum(sigma, 1e-8)
    
    # Cost-awareness:
    # We want to explore (high sigma) when cost is low.
    # We want to exploit (low mu) always, but maybe less if cost is very high?
    # Actually, exploitation should be independent of cost in UCB, 
    # but the *value* of a point is its improvement.
    
    # Let's define the score as:
    # Exploitation term: (f_best - mu)
    # Exploration term: beta * sigma
    # Penalty: 1 / cost
    
    # Wait, if we just divide by cost, we might ignore high improvement 
    # in high cost regions if the improvement is small.
    
    # Let's try: score = (f_best - mu + beta * sigma) / cost
    # This is "Expected Improvement per unit cost" approximation using UCB.
    
    # Adaptive beta: 
    # Start with high exploration, decrease as budget is spent?
    # We don't have total budget info directly, but we have f_best.
    # However, we can estimate progress by how close f_best is to the known min (-3.86).
    # But we don't know the instance specific min.
    
    # Let's use a fixed beta that balances the scales.
    # mu and sigma are in objective units. Hartmann 3D range is roughly [-4, 0].
    # cost is in [1, 2].
    # sigma is typically small (0.1 - 0.5).
    # f_best - mu is typically small.
    
    # If we use beta = 1.0, exploration is weighted equally to exploitation.
    # If we use beta = 0.5, exploitation is favored.
    
    # Let's try a slightly more sophisticated cost-awareness:
    # Instead of dividing by cost, we subtract a penalty proportional to cost * sigma?
    # No, we want to *avoid* high cost for the same uncertainty.
    
    # Let's use: score = (f_best - mu) + beta * sigma * (1.0 / (cost + 1e-9))
    
    # This ensures that for a given uncertainty (sigma), the exploration bonus
    # is higher for cheaper points.
    
    # What value for beta?
    # Typical EI values are in the range of 0.01 to 0.1.
    # (f_best - mu) is similar.
    # sigma is similar.
    # cost is ~1.5.
    # So 1/cost is ~0.6.
    # If beta=1, exploration term is ~0.6 * sigma.
    # This seems reasonable.
    
    # To differentiate from the seed (EI/sqrt(cost)), this is a linear UCB 
    # with inverse cost weighting on the exploration term only.
    # The seed penalizes BOTH exploitation and exploration by cost.
    # This new strategy says: "Exploitation is valuable regardless of cost 
    # (if you found a good point, take it, even if it's expensive), but 
    # Exploration should be cheap."
    
    # This is a crucial conceptual shift: 
    # Seed: "Everything is expensive, so divide everything by cost."
    # New: "Exploitation is a reward, Exploration is a gamble. 
    #        Gamble more on cheap points."
    
    beta = 1.5  # Slightly favor exploration to avoid local minima in 3D
    
    exploration_term = beta * sigma / (cost + 1e-9)
    exploitation_term = f_best - mu
    
    scores = exploitation_term + exploration_term
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
