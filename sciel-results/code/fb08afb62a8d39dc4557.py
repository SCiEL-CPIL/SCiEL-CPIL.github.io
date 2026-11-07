# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_deadline = None
_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    """
    Compute a cost-aware acquisition score using a Kriging Believer variant of EI.
    
    Strategy:
    1. Compute Expected Improvement (EI) for each candidate.
    2. Find the top EI value and its corresponding z-score (z* = (f_best - mu*)/sigma*).
    3. Compute a 'believer' score: mu + sigma * z* for all candidates.
       This selects points that are most likely to be good *given* the improvement potential.
    4. Combine with cost-awareness: Score = (Believer Score) / (cost ** alpha).
       The adaptive alpha increases cost avoidance as the budget depletes.
    
    This is a structural change from simple PI or TS: it uses the EI landscape to
    guide a deterministic selection (believer) that is more focused than random sampling.
    """
    global _deadline, _start_time, _call_count
    
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + 325.0  # 5s safety margin before 330s limit

    # Increment call count
    _call_count += 1

    # Check deadline
    if time.time() > _deadline:
        # Emergency fallback: return simple EI/cost
        eps = 1e-10
        sigma_safe = np.maximum(sigma, eps)
        cost_safe = np.maximum(cost, eps)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        scores = ei / cost_safe
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # 1. Compute Z-scores and EI
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 2. Find the best EI and its corresponding z-score
    # We use the top EI candidate to define the "believer" threshold.
    # This is the z* value that defines the quantile of the posterior for the best point.
    idx_best = np.argmax(ei)
    z_star = z[idx_best]
    
    # 3. Compute Kriging Believer score for all points
    # score_believer = mu + sigma * z_star
    # This effectively asks: "If I were to pick the point that achieves the top EI's z-score, where would I be?"
    # Points with high mu and/or high sigma (if z_star > 0) will score high.
    # Since z_star is typically positive (we want improvement), this favors high mu.
    # But it also scales with sigma, so it balances exploration.
    believer_score = mu + sigma_safe * z_star
    
    # 4. Estimate remaining budget for adaptive cost exponent
    # Initial 5 points cost ~7.5 (uniform in [0,1]^6, mean cost = 1 + 0.5 = 1.5 each)
    # Each subsequent point costs ~1.5 on average
    initial_cost_est = 7.5
    avg_cost_per_step = 1.5
    total_cost_est = initial_cost_est + _call_count * avg_cost_per_step
    remaining_budget = max(0.0, 30.0 - total_cost_est)
    
    # Adaptive cost exponent:
    # When budget is plentiful (remaining > 10), alpha = 1.0
    # When budget is tight (remaining < 5), alpha = 2.0
    # Linear interpolation in between
    if remaining_budget > 10.0:
        alpha = 1.0
    elif remaining_budget < 5.0:
        alpha = 2.0
    else:
        alpha = 1.0 + (10.0 - remaining_budget) / 5.0
    
    # 5. Final Score = Believer Score / (cost ** alpha)
    # We want higher scores to be selected.
    # Believer score can be negative (if mu is very low), which is good for minimization.
    # However, dividing by cost (which is positive) means negative scores stay negative.
    # The evaluator picks the argmax.
    # If all scores are negative, the one closest to 0 (least negative) is picked.
    # This is correct behavior: we want the highest (least negative) predicted value.
    
    scores = believer_score / (cost_safe ** alpha)
    
    # Handle numerical issues
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores

# EVOLVE-BLOCK-END
