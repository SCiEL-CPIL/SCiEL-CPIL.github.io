"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 280.0  # Leave 50s for cleanup/other overhead


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on hartmann 3d.
    
    Strategy: Adaptive UCB with budget-aware exploration coefficient.
    Uses UCB for better exploration-exploitation balance, with beta adapting
    to remaining budget (estimated from f_best magnitude and cost statistics).
    """
    # Deadline guard
    if (time.time() - _START_TIME) > _TIME_LIMIT:
        # Return a safe, fast score: just exploit (negative mu for minimization)
        return -mu.copy()
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Estimate number of evaluations so far from cost statistics
    # Initial cost: 5 uniform random points, each costs 1 + mean(x) where x in [0,1]^3
    # So initial cost ~ 5 * (1 + 0.5) = 7.5
    # Current total cost can be estimated: we know cost of candidates, but not total spent.
    # Use a heuristic: assume total cost is roughly proportional to how well f_best has improved.
    # Better: use the mean cost of candidates as a proxy for average evaluation cost,
    # and estimate n_evals from budget remaining.
    
    # Total budget is 30. Initial cost ~7.5. Remaining ~22.5.
    # Average cost per eval ~ 1 + 0.5 = 1.5 (for uniform points in [0,1]^3)
    # But cost varies: cost = 1 + mean(x) where x scaled to [0,1]
    # For hartmann 3d on [0,1]^3, cost = 1 + (x1+x2+x3)/3
    
    # Estimate total evaluations so far:
    # We don't have direct access to total cost spent, but we can estimate.
    # The evaluator runs while total cost < 30.
    # Let's estimate n_evals from the spread of mu and sigma as a proxy for GP maturity.
    
    # Simpler approach: use a fixed but adaptive beta based on cost statistics
    mean_cost = np.mean(cost)
    std_cost = np.std(cost)
    
    # Beta for UCB: start with moderate exploration, decrease as budget depletes
    # Estimate remaining budget fraction:
    # Assume average cost per eval is ~ mean_cost (from candidate costs)
    # Total budget = 30, initial ~7.5, so ~22.5 remaining
    # But we don't know exactly how much has been spent.
    
    # Use a heuristic: beta decreases with the number of evaluations
    # Estimate n_evals: if total cost is C, and avg cost is ~1.5, then n ~ C/1.5
    # We can estimate C from the fact that the loop continues while C < 30.
    # Use f_best as a proxy: better f_best suggests more evaluations.
    
    # For simplicity, use a fixed beta that balances exploration well for this problem.
    # The hartmann 3d has a known global minimum of -3.86278.
    # f_best starts around -1 to -2 and improves to ~-3.86.
    
    # Adaptive beta: higher when uncertainty is high (early), lower when confident (late)
    # Use sigma statistics to gauge GP uncertainty
    mean_sigma = np.mean(sigma)
    
    # Beta: scale with mean uncertainty relative to improvement scale
    # When mean_sigma is high relative to |f_best|, explore more
    improvement_scale = max(abs(f_best), 1.0)
    uncertainty_ratio = mean_sigma / improvement_scale
    
    # Base beta with adaptation
    base_beta = 1.5
    # Increase beta when uncertainty is high (early phase)
    beta = base_beta * (1.0 + 0.5 * uncertainty_ratio)
    
    # UCB for minimization: score = -mu + beta * sigma
    # Higher score = more promising
    ucb_score = -mu + beta * sigma
    
    # Cost-aware adjustment: penalize high-cost points
    # Use a multiplicative penalty based on cost relative to mean
    if mean_cost > 1e-12:
        cost_ratio = cost / mean_cost
        # Penalty: reduce score for expensive points
        # Use a smooth penalty that's 1 at mean cost
        penalty_factor = 1.0 / (1.0 + 0.3 * np.maximum(cost_ratio - 1.0, 0.0))
        score = ucb_score * penalty_factor
    else:
        score = ucb_score.copy()
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
