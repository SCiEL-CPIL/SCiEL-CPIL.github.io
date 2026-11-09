import numpy as np
from scipy.stats import norm
import time

# ---- Deadline guard ----
_START_TIME = time.time()
TIME_LIMIT = 330.0
SAFETY_MARGIN = 20.0

# ---- Constants ----
DIM = 4
RANGE_LOW = -4.0
RANGE_HIGH = 5.0
SPAN = RANGE_HIGH - RANGE_LOW
COST_BUDGET = 30.0
N_INIT = 5
N_CAND = 2000

def _compute_scores(mu, sigma, cost, f_best):
    """
    Computes cost-aware acquisition scores.
    Uses Expected Improvement (EI) normalized by cost, plus a small UCB term.
    """
    eps = 1e-8
    # Ensure sigma is positive and finite
    safe_sigma = np.where(sigma < eps, eps, sigma)
    safe_sigma = np.where(np.isfinite(safe_sigma), safe_sigma, eps)
    
    # Ensure cost is positive and finite
    safe_cost = np.where(cost < eps, eps, cost)
    safe_cost = np.where(np.isfinite(safe_cost), safe_cost, eps)
    
    # Ensure mu is finite
    mu_safe = np.where(np.isfinite(mu), mu, f_best)
    
    # Standardized distance to current best
    z = (f_best - mu_safe) / safe_sigma
    z = np.clip(z, -10.0, 10.0)
    
    # PDF and CDF of standard normal
    phi_z = norm.pdf(z)
    cdf_z = norm.cdf(z)
    
    # Expected Improvement
    # EI = (f_best - mu) * P(f < f_best) + sigma * pdf(z)
    # Note: We want to minimize f. So improvement is f_best - f.
    ei = (f_best - mu_safe) * cdf_z + safe_sigma * phi_z
    ei = np.maximum(ei, 0.0) # EI should be non-negative
    
    # UCB term for exploration
    # We use a small weight for UCB to ensure we don't get stuck in local minima
    # The weight decays as f_best improves (simple heuristic)
    # f_best starts high and goes to 0.
    # Let's use a fixed small weight for now, or adapt based on z.
    # High z (mu << f_best) means high confidence improvement, low need for exploration.
    # Low z (mu ~ f_best) means uncertain, need exploration.
    # Actually, UCB is -mu + kappa*sigma.
    # Let's blend EI and UCB.
    
    # A common robust approach is simply EI / cost.
    # Let's add a small exploration bonus proportional to sigma / cost.
    # This encourages sampling where we are uncertain, but penalized by cost.
    
    exploration_weight = 0.1
    score = (ei + exploration_weight * safe_sigma) / safe_cost
    
    # Handle any remaining numerical issues
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

def acquisition(mu, sigma, f_best, cost):
    """
    Public entry point.
    """
    # Hard deadline guard
    if time.time() - _START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback: just prefer low cost
        cost_arr = np.asarray(cost, dtype=np.float64)
        return 1.0 / np.where(cost_arr < 1e-9, 1e-9, cost_arr)
    
    mu_arr = np.asarray(mu, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    cost_arr = np.asarray(cost, dtype=np.float64)
    
    return _compute_scores(mu_arr, sigma_arr, cost_arr, float(f_best))
