# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_DEADLINE = None
_DEADLINE_MARGIN = 10.0  # seconds before the hard limit (300s)

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 290.0  # 300s limit, 10s safety margin
    return _DEADLINE

def _time_remaining():
    return max(0.0, _get_deadline() - time.time())

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Lower Confidence Bound (LCB) per unit cost.
    
    Strategy:
    1. Calculate LCB: mu - kappa * sigma.
       This directly targets the most optimistic lower bound of the posterior.
    2. Normalize by cost to prefer cheaper evaluations that offer high potential improvement.
    3. Ensure numerical stability and finite outputs.
    
    Rationale:
    Unlike EI, which can be conservative and diluted by the probability of improvement term,
    LCB with a tuned kappa provides a stronger, more direct exploration signal, especially
    in early stages where sigma is high. Normalizing by cost ensures we value information
    density over absolute improvement if the improvement is expensive to obtain.
    """
    # Check time remaining to avoid timeout
    if _time_remaining() < 0.1:
        # Return a safe, finite score if time is almost up
        return np.zeros(len(mu))
    
    # Ensure sigma is positive to avoid issues, though LCB doesn't divide by sigma
    sigma = np.maximum(sigma, 0.0)
    
    # Tuning constant for exploration
    # Based on knowledge from other solutions, kappa=1.5 showed strong performance
    # for this specific task structure.
    kappa = 1.5
    
    # Calculate LCB: mu - kappa * sigma
    # We want to minimize, so lower LCB is better.
    # However, the task requires returning scores where HIGHER = evaluate next.
    # So we need to invert the LCB logic or negate it.
    # LCB_value = mu - kappa * sigma
    # We want to maximize -LCB_value / cost
    # Which is equivalent to maximizing (kappa * sigma - mu) / cost
    
    # Let's stick to the standard form:
    # lcb = mu - kappa * sigma
    # score = -lcb / cost = (kappa * sigma - mu) / cost
    
    # Alternatively, just compute (kappa * sigma - mu) / cost directly.
    # This rewards high uncertainty (sigma) and low mean (mu), penalized by cost.
    
    numerator = kappa * sigma - mu
    denominator = cost + 1e-9
    
    score = numerator / denominator
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
