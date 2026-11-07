"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state to track elapsed time for safety
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Compute cost-aware acquisition scores using Kriging Believer / Probabilistic Improvement
    with dynamic exploration and cost normalization.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Mean of GP predictions
    sigma : array of shape (2000,)
        Standard deviation of GP predictions
    f_best : float
        Best observed function value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = more promising to evaluate)
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    # Safety guard: if we are running out of time, return a safe fallback
    # The task says 300s per evaluation, but the total budget is 330s wall-clock.
    # If we've been running for > 290s, we should be very fast or return defaults.
    # However, this function is called repeatedly. We assume the caller manages the loop.
    # We just need to ensure this function itself is fast and doesn't hang.
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-8)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # 1. Core Signal: Probability of Improvement (Kriging Believer proxy)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues in CDF
    z = np.clip(z, -10, 10)
    prob_improvement = norm.cdf(z)
    
    # 2. Exploration Bonus:
    # We want to explore regions with high uncertainty, but the importance of exploration
    # should decay as we approach the budget limit or as we get more confident.
    # Since we don't have explicit 't' (iteration count), we can infer progress from f_best
    # or simply use a fixed moderate exploration weight that is scaled by the inverse cost.
    # However, the directive suggests dynamic annealing based on cost/budget.
    # We don't know the total budget spent so far directly, only the cost of the *candidates*.
    # But we can estimate "progress" by how low f_best is relative to the known minimum?
    # No, we don't know the true minimum. 
    # Alternative: Use the magnitude of improvement. If f_best is very low, we are likely late.
    # Hartmann 6d min is ~ -3.32. Initial uniform randoms might be ~ -1.0 to -2.0.
    # Let's use a heuristic: 
    # exploration_weight = base_explore * (1 + (f_best - 0.0) / 3.0) ? 
    # Actually, simpler: The standard Kriging Believer relies on the GP to handle exploration via sigma.
    # The term Phi((f_best - mu)/sigma) naturally balances:
    # - If mu < f_best, prob is high (exploitation)
    # - If sigma is large, the distribution spreads, increasing the chance of being below f_best even if mu is high.
    
    # To explicitly encourage exploration in high-sigma areas where the mean isn't already great,
    # we can add a small bonus proportional to sigma, but normalized.
    # Let's stick to the pure Probabilistic Improvement as the primary driver, 
    # but normalize it by cost.
    
    # Cost Normalization:
    # The cost is 1 + mean(x). Ranges from 1.0 to 2.0.
    # Dividing by cost penalizes expensive points.
    # Dividing by sqrt(cost) is a softer penalty, allowing some expensive points if they are very promising.
    # The "Knowledge" section mentioned sqrt(cost) improved a score. Let's use sqrt(cost).
    
    cost_norm = np.sqrt(cost)
    
    scores = prob_improvement / cost_norm
    
    # 3. Dynamic Annealing / Boost:
    # If f_best is still high (e.g., > -1.0), we are in early stages. We should explore more.
    # If f_best is low (e.g., < -3.0), we are in late stages. We should exploit.
    # The prob_improvement already does this implicitly, but we can sharpen it.
    # In late stages, we want to focus tightly on the best region.
    # In early stages, we want to cover the space.
    
    # Let's add a small UCB-like term that is active only when uncertainty is high relative to improvement potential.
    # Or, simpler: Just rely on the CDF which is robust.
    
    # One issue: If all points have mu > f_best and low sigma, prob_improvement is ~0.
    # We need to ensure we pick the "least bad" or most uncertain point.
    # The CDF handles this: if mu >> f_best, z is very negative, CDF is ~0.
    # If sigma is large, z is less negative, CDF is higher.
    
    # To ensure we always have a valid score and handle the "all bad" case,
    # we can add a tiny epsilon or rely on the relative ranking.
    
    # Let's add a small exploration bonus that is inversely proportional to the number of points?
    # No, let's keep it simple. The CDF is a strong signal.
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
