import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 10.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a hybrid EI + UCB approach with adaptive blending.
    
    The blend weight is determined by the exploration-exploitation tradeoff:
    - Early on (high uncertainty relative to best), UCB dominates for exploration.
    - Later (low uncertainty relative to best), EI dominates for exploitation.
    
    The UCB component uses a time-varying beta that decreases as we get closer
    to the deadline, encouraging exploitation in later steps.
    
    Args:
        mu: Mean of the GP posterior at candidates (shape 2000,).
        sigma: Std dev of the GP posterior at candidates (shape 2000,).
        f_best: Current best function value (minimum).
        
    Returns:
        scores: Array of shape 2000, higher is better.
    """
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        # Fallback to pure EI if too close to deadline
        z = (f_best - mu) / (sigma + 1e-9)
        scores = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    n = len(mu)
    
    # Ensure sigma is positive to avoid errors
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute EI (Expected Improvement)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute UCB (Upper Confidence Bound)
    # UCB = mu + beta * sigma, where beta controls exploration
    # Use a time-varying beta: start high for exploration, decrease for exploitation
    # The fraction of time remaining can guide beta
    time_remaining_fraction = max(0.0, (_DEADLINE - elapsed) / _DEADLINE)
    
    # Beta decreases from ~3.0 early to ~0.5 late
    # This encourages more exploration early and more exploitation later
    beta = 0.5 + 2.5 * time_remaining_fraction
    
    ucb = mu + beta * sigma_safe
    
    # Normalize EI and UCB to comparable scales for blending
    # EI is already in objective units (improvement over f_best)
    # UCB is in objective units (predicted value + exploration bonus)
    # We want higher scores to mean "evaluate next"
    # For minimization: lower predicted value is better
    # EI: higher = more expected improvement = better
    # UCB: lower mu + higher sigma = more exploration potential
    # But UCB as defined above gives higher values for more uncertain/lower mean
    # We need to invert UCB for minimization: score_ucb = -ucb
    # Actually, let's think about this more carefully:
    # - EI directly measures expected improvement (higher is better)
    # - For UCB in minimization, we want to explore where mu is low and sigma is high
    #   So score_ucb = -mu + beta * sigma (higher when mu is low and sigma is high)
    
    score_ucb = -mu + beta * sigma_safe
    
    # Structural Change: Direct weighted sum without standardization.
    # EI is in objective units (improvement).
    # UCB score is in objective units (potential value).
    # We scale UCB to be comparable to EI by dividing by a characteristic scale.
    # Or simply use a fixed weight and a tuned beta.
    
    # Use a fixed beta that balances exploration and exploitation for ST-1D.
    # Beta ~ 2.0 is a common starting point for UCB in BO.
    beta_fixed = 2.0
    
    # Recalculate UCB with fixed beta
    score_ucb_fixed = -mu + beta_fixed * sigma_safe
    
    # Combine EI and UCB directly.
    # EI values are typically small (improvement).
    # UCB values can be large (absolute value).
    # To make them comparable, we can scale UCB by the range of the function or f_best.
    # A simple heuristic: scale UCB by 1/|f_best| if f_best is large, or just use a weight.
    
    # Let's use a weight of 0.5 for EI and 0.5 for UCB, but scale UCB down.
    # Scaling factor: 1.0 / (abs(f_best) + 1.0) might be too aggressive.
    # Let's try: scores = ei + 0.1 * score_ucb_fixed
    # Or: scores = 0.5 * ei + 0.5 * score_ucb_fixed / (abs(f_best) + 1.0)
    
    # Actually, EI is already "improvement". UCB is "value".
    # If we want to minimize, we want high EI and low UCB value (high score_ucb_fixed).
    
    # Let's try a simple linear combination with a small weight on UCB to allow exploration without dominating.
    # The previous normalization might have over-amplified noise.
    
    # Use a time-decaying weight for UCB to ensure exploitation later.
    time_remaining_fraction = max(0.0, (_DEADLINE - elapsed) / _DEADLINE)
    w_ucb = 0.5 * time_remaining_fraction  # Start at 0.5, go to 0
    w_ei = 1.0
    
    # Scale UCB to be in "improvement" units?
    # Approximate improvement potential: f_best - mu + beta * sigma
    # This is actually the UCB of the *improvement*? No.
    
    # Let's just use:
    # scores = ei + w_ucb * sigma_safe
    # This adds a pure exploration bonus (variance) to the EI.
    # This is a "EI + Exploration" structure.
    
    scores = ei + w_ucb * sigma_safe
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
