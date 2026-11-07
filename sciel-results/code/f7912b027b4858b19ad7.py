import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using cost-normalized Lower Confidence Bound (LCB)
    with dynamic exploration based on remaining budget.
    
    Args:
        mu: (2000,) array of GP means.
        sigma: (2000,) array of GP standard deviations.
        f_best: float, the best observed value so far.
        cost: (2000,) array of evaluation costs.
        
    Returns:
        (2000,) array of scores, higher is better.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Estimate remaining budget: total budget is 30, initial 5 points cost ~2.5 each = ~12.5
    # So remaining is roughly 30 - 12.5 = 17.5, but we don't know exact spent.
    # Use a heuristic: if f_best is close to min (-0.8), we're exploiting; if far, exploring.
    
    # Dynamic exploration parameter xi:
    # When f_best is far from the theoretical minimum (-0.8), explore more.
    # When close, exploit more.
    gap_to_min = f_best - (-0.8)
    # Normalize gap: if gap is large (e.g., > 0.5), explore more; if small, exploit.
    # xi controls how much we trust the mean vs uncertainty.
    # For minimization LCB: score = -(mu - xi * sigma) / cost
    # Higher xi = more exploration.
    
    # Adaptive xi: start high when far from optimum, decrease as we get closer.
    # gap_to_min ranges from ~0 (at optimum) to ~1.2 (worst case in [-0.8, 0.4] range)
    # Map gap to xi in [0.5, 2.0]
    xi = 0.5 + 1.5 * np.clip(gap_to_min / 1.0, 0.0, 1.0)
    
    # LCB for minimization: we want low mu and high sigma (uncertainty)
    # score = -(mu - xi * sigma) / cost
    # This is negative of the LCB value, normalized by cost.
    lcb_value = mu - xi * sigma_safe
    scores = -lcb_value / cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
