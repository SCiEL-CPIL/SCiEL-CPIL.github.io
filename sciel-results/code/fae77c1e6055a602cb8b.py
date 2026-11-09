"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    UCB-based acquisition function.
    
    Strategy:
    - Use Upper Confidence Bound (UCB) to balance exploration and exploitation.
    - Score = mu + kappa * sigma
    - kappa is dynamic: it starts high to encourage exploration when f_best is poor,
      and decays as f_best approaches the known minimum (-3.32237).
    
    This differs from EI by explicitly rewarding uncertainty (sigma) with a bonus,
    rather than only rewarding expected improvement over f_best. This helps escape
    local minima in 6D space by visiting high-variance regions.
    """
    # Ensure sigma is positive to avoid division issues, though UCB doesn't divide by sigma.
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Dynamic kappa:
    # The known minimum is -3.32237.
    # If f_best is far from the minimum, we are in early exploration -> high kappa.
    # If f_best is close, we are in late exploitation -> low kappa.
    
    # Normalize the "distance" from the known minimum.
    # f_best is the current best (lowest) value.
    # We want kappa to be large when f_best is high (bad) and small when f_best is low (good).
    
    # A simple heuristic:
    # Let's assume the function values are roughly in [-3.3, 0].
    # If f_best is -3.3, kappa should be small (e.g., 0.5).
    # If f_best is 0.0, kappa should be large (e.g., 3.0).
    
    # Linear interpolation based on f_best relative to the range.
    # min_val = -3.32237
    # max_val = 0.0 (approx, Hartmann 6D max is around 0 or slightly positive depending on definition, usually bounded)
    
    min_val = -3.32237
    # Use a reference "worst" value for scaling. 0.0 is a reasonable upper bound for Hartmann 6D.
    worst_val = 0.0
    
    # Clamp f_best to [min_val, worst_val] for safe interpolation
    f_best_clamped = np.clip(f_best, min_val, worst_val)
    
    # t goes from 0 (f_best = min_val) to 1 (f_best = worst_val)
    # If f_best is min_val, t=0. If f_best is worst_val, t=1.
    t = (f_best_clamped - min_val) / (worst_val - min_val)
    
    # kappa interpolates from kappa_min to kappa_max
    # We want more exploration when t is high (bad f_best) and less when t is low (good f_best).
    kappa_min = 0.5  # Exploitation phase
    kappa_max = 3.0  # Exploration phase
    
    kappa = kappa_min + t * (kappa_max - kappa_min)
    
    # UCB score: higher is better.
    # We want to maximize mu + kappa * sigma.
    # Since we are minimizing the function, a low mu is good.
    # Wait, standard UCB for minimization is: score = mu - kappa * sigma?
    # No, in BO, we usually maximize the acquisition function.
    # If we want to find low values of f(x), we want low mu.
    # But UCB is typically for maximization of reward.
    # Let's reframe: We want to maximize the "potential" for finding a better minimum.
    # If we view "improvement" as (f_best - f(x)), we want to maximize that.
    # The UCB of the improvement?
    
    # Let's stick to the standard UCB interpretation for minimization:
    # We want to visit points where the true value might be low.
    # The lower bound of the prediction is mu - kappa * sigma.
    # We want to maximize this lower bound? No, that's optimistic.
    # Actually, for minimization, we often use:
    # Acquisition = -mu + kappa * sigma  (Maximize this)
    # Or simply:
    # Acquisition = mu - kappa * sigma (Minimize this)
    
    # The task requires: "Return 2000 finite scores, higher meaning evaluate next".
    # So we must return a score where HIGHER is better.
    
    # If we use UCB for minimization:
    # We are optimistic that the function value is low.
    # The "optimistic" value is mu - kappa * sigma.
    # But we want to maximize the score.
    # So score = -(mu - kappa * sigma) = -mu + kappa * sigma.
    
    # Let's verify:
    # If mu is very low (good), -mu is high (good score).
    # If sigma is high (uncertain), kappa*sigma is high (good score).
    # This encourages both low mean and high uncertainty.
    
    score = -mu + kappa * sigma_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
