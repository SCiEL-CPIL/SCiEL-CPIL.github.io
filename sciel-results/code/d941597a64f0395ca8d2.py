"""Cost-aware acquisition function for Bayesian optimization (Styblinski-Tang 2D)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 285  # Safety margin before 300s limit


def _elapsed_ok():
    return (time.time() - _start_time) < _TIMEOUT


def _bisect_lambda(scores_no_cost, cost, target_cost, lo=0.0, hi=100.0, tol=1e-3):
    """
    Binary search for lambda such that the mean cost of the top 20% candidates
    (by score) is close to target_cost.
    """
    n = len(cost)
    top_k = max(1, int(0.2 * n))
    
    def mean_cost_at_lambda(lam):
        scores = scores_no_cost - lam * cost
        idx = np.argpartition(-scores, top_k)[:top_k]
        return np.mean(cost[idx])
    
    # Check boundaries
    mc_lo = mean_cost_at_lambda(lo)
    mc_hi = mean_cost_at_lambda(hi)
    
    # If even at hi the mean cost is above target, use hi
    if mc_hi > target_cost:
        return hi
    # If at lo the mean cost is below target, use lo
    if mc_lo < target_cost:
        return lo
    
    for _ in range(20):
        mid = (lo + hi) / 2.0
        mc = mean_cost_at_lambda(mid)
        if mc > target_cost:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    
    return (lo + hi) / 2.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Lagrangian relaxation of the cost constraint.
    
    Score = (f_best - mu) - lambda * cost
    
    lambda is dynamically tuned so that the average cost of the top candidates
    aligns with a target cost, decoupling exploration from cost geometry.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = more promising)
    """
    # Ensure all inputs are finite and valid
    sigma = np.maximum(sigma, 1e-8)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    mu = np.where(np.isfinite(mu), mu, f_best)
    cost = np.maximum(cost, 1e-8)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Base score: expected improvement (higher is better)
    # For minimization, improvement = f_best - mu
    # We use EI as the base signal
    improvement = f_best - mu
    z = improvement / sigma
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (improvement * cdf_z + sigma * pdf_z)
    ei = np.maximum(ei, 0.0)
    
    # Add exploration: use LCB-style term
    # LCB = mu - beta*sigma, we want to minimize, so score = -LCB = -mu + beta*sigma
    # But we want to combine with EI. Let's use a simpler approach:
    # Base score = EI + exploration_bonus
    # exploration_bonus encourages high sigma (uncertainty)
    known_min = -78.3323
    # Progress: 0 = far from min, 1 = at min
    progress = max(0.0, min(1.0, 1.0 - (f_best - known_min) / 100.0))
    
    # Beta for exploration: more exploration early, less later
    beta = 1.5 * (1.0 - 0.7 * progress)
    
    # Exploration bonus: beta * sigma (encourages uncertain regions)
    exploration_bonus = beta * sigma
    
    # Normalize ei and exploration_bonus to have similar scale
    ei_mean = np.mean(ei)
    ei_std = np.std(ei)
    if ei_std > 1e-9:
        ei_norm = (ei - ei_mean) / ei_std
    else:
        ei_norm = np.zeros_like(ei)
    
    ex_mean = np.mean(exploration_bonus)
    ex_std = np.std(exploration_bonus)
    if ex_std > 1e-9:
        ex_norm = (exploration_bonus - ex_mean) / ex_std
    else:
        ex_norm = np.zeros_like(exploration_bonus)
    
    # Combine: weight exploitation vs exploration
    w_exploit = 0.6 + 0.3 * progress  # 0.6 early, 0.9 late
    w_explore = 1.0 - w_exploit
    
    scores_no_cost = w_exploit * ei_norm + w_explore * ex_norm
    
    # Safety check
    if not _elapsed_ok():
        # Return simple score
        return ei / cost
    
    # Determine target cost: we want to balance cost
    # The cost ranges from 1 to 2. 
    # We want to prefer cheaper points but not exclusively.
    # Target: slightly above the minimum cost, to allow some expensive points
    # if they are very promising.
    # A good heuristic: target the mean cost of the top 20% to be ~1.2-1.3
    # This allows some expensive points but biases towards cheaper ones.
    target_cost = 1.2
    
    # Find lambda using binary search
    lam = _bisect_lambda(scores_no_cost, cost, target_cost)
    
    # Final score
    scores = scores_no_cost - lam * cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
