"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290  # 40s margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using a dynamic GP-LCB with adaptive exploration.
    
    The exploration coefficient beta is dynamically adjusted based on:
    1. The current best value f_best (larger beta when f_best is larger,
       since we're further from the optimum)
    2. The remaining budget (larger beta when more budget remains)
    3. The spread of sigma values (adaptive to local uncertainty)
    
    Parameters
    ----------
    mu : array of shape (2000,)
        GP mean predictions in objective units.
    sigma : array of shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed value so far.
    cost : array of shape (2000,)
        Evaluation costs for each candidate.
    
    Returns
    -------
    scores : array of shape (2000,)
        Acquisition values (higher = evaluate next).
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.zeros(2000, dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against division by zero and negative values
    cost_safe = np.maximum(cost, 1e-9)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute basic quantities
    f_best_val = float(f_best)
    
    # Dynamic exploration coefficient beta:
    # - Scale by sqrt(log-like term) based on f_best
    # - When f_best is large, we need more exploration
    # - When f_best is small (close to 0), we can be more exploitative
    # - Also scale by remaining budget fraction
    
    # Estimate remaining budget: total budget is 30, initial cost ~5 + 5*1.25 ≈ 11.25
    # So remaining is roughly 30 - current_cost, but we don't have current_cost directly.
    # We can estimate from the mean cost of evaluated points, but simpler: use f_best as proxy.
    
    # Base beta: use a log-scale adjustment
    # When f_best is large (e.g., 100+), beta should be large (more exploration)
    # When f_best is small (e.g., < 1), beta should be smaller (more exploitation)
    
    # Use: beta = sqrt(2 * log(1 + f_best) + 2 * log(1 + n_evals))
    # But we don't have n_evals. Use f_best as a proxy for progress.
    
    # Simple adaptive beta:
    # beta = c * sqrt(1 + f_best) / (1 + 0.1 * f_best)
    # This peaks at moderate f_best and decreases as f_best -> 0
    
    # Alternative: use the quantile of sigma to adapt
    sigma_median = np.median(sigma_safe)
    sigma_max = np.max(sigma_safe)
    
    # Dynamic beta based on f_best and sigma spread
    # beta should be larger when:
    # 1. f_best is large (far from optimum)
    # 2. sigma is large (high uncertainty)
    
    # Use a formula that balances exploitation and exploration:
    # LCB = mu - beta * sigma
    # We want beta to adapt: larger when f_best is large, smaller when f_best is small
    
    # beta = sqrt(2 * (1 + f_best)) * (sigma_median / (sigma_max + 1e-9))
    # This makes beta proportional to the relative uncertainty
    
    # Let's use a simpler, more robust approach:
    # beta = 2.0 * (1.0 + 0.5 * np.log1p(max(f_best_val, 0.0)))
    # This gives beta ~2 when f_best=0, and increases slowly as f_best grows
    
    # But we also want cost-awareness: divide by cost
    
    # Compute LCB
    # Use a dynamic beta that accounts for the current state
    # beta scales with the "distance" from optimum and uncertainty
    
    # Strategy: 
    # 1. Compute raw LCB with a base beta
    # 2. Normalize by cost
    # 3. Add a small exploration bonus for high-sigma regions
    
    # Base beta: start with a moderate value, adjust based on f_best
    # When f_best is large, we're far from the minimum, so explore more
    # When f_best is small, exploit more
    
    # Use: beta = 1.5 + 0.5 * min(f_best / 10, 1.0)
    # This caps the exploration boost
    
    f_best_clipped = min(max(f_best_val, 0.0), 50.0)
    beta = 1.5 + 0.5 * (f_best_clipped / 10.0)
    
    # Also incorporate sigma-based adaptation
    # If sigma is very large relative to the scale of mu, increase beta
    mu_scale = np.std(mu) + 1e-9
    sigma_scale = np.std(sigma_safe) + 1e-9
    
    # Relative uncertainty
    rel_uncertainty = sigma_scale / (mu_scale + 1e-9)
    
    # Adjust beta based on relative uncertainty
    beta *= (1.0 + 0.3 * min(rel_uncertainty, 2.0))
    
    # Compute GP-LCB: mu - beta * sigma (for minimization, lower is better)
    # But we want higher scores to be evaluated next, so we negate:
    # score = -(mu - beta * sigma) = -mu + beta * sigma
    # Or equivalently: score = beta * sigma - mu
    
    lcb_scores = -mu + beta * sigma_safe
    
    # Cost-aware: divide by cost
    # But also add a small term to encourage exploring high-cost regions
    # if they have high expected improvement
    
    # Normalize lcb_scores to [0, 1] range for better cost scaling
    lcb_min = np.min(lcb_scores)
    lcb_max = np.max(lcb_scores)
    lcb_range = lcb_max - lcb_min + 1e-9
    
    lcb_normalized = (lcb_scores - lcb_min) / lcb_range
    
    # Combine: primary score is cost-normalized LCB
    scores = lcb_normalized / cost_safe
    
    # Add a small exploration bonus for high sigma regions
    # This helps escape local minima in the Rosenbrock valley
    sigma_normalized = sigma_safe / (np.max(sigma_safe) + 1e-9)
    exploration_bonus = 0.1 * sigma_normalized / cost_safe
    
    scores = scores + exploration_bonus
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
