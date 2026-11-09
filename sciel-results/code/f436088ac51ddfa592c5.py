"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track call count and previous best for adaptive beta
_call_count = 0
_prev_f_best = None


def acquisition(mu, sigma, f_best):
    """
    Lower Confidence Bound (LCB) acquisition function with adaptive beta.
    LCB = mu - beta * sigma, where beta starts high for exploration and decays
    to focus on exploitation as we converge.
    """
    global _call_count, _prev_f_best
    
    _call_count += 1
    
    # Guard against zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Adaptive beta: start with high exploration, decay over time
    # For 30 evaluations total, we want beta to decay from ~3.0 to ~0.5
    # Use a smooth decay based on call count
    # The first 5 are initial points, so we start adapting from call 1 (which is the 6th evaluation)
    # But _call_count starts at 1 for the first acquisition call (which is for the 6th point)
    # Let's use a decay that depends on how much progress we've made
    
    # Simple approach: beta decays exponentially with call count
    # beta = beta_min + (beta_max - beta_min) * exp(-decay_rate * call_count)
    beta_max = 3.0
    beta_min = 0.5
    decay_rate = 0.15
    beta = beta_min + (beta_max - beta_min) * np.exp(-decay_rate * _call_count)
    
    # Lower Confidence Bound: lower is better
    lcb = mu - beta * sigma_safe
    
    # We want higher scores to mean "evaluate next"
    # So we negate LCB: higher negated LCB means lower LCB means more promising
    scores = -lcb
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Update previous f_best for potential future use
    _prev_f_best = f_best
    
    return scores
# EVOLVE-BLOCK-END
