# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a Quantile-Based Lower Confidence Bound (LCB) divided by cost.
    
    The score is:
        score(x) = LCB(x) / cost(x)
    
    where LCB(x) = mu(x) - beta * sigma(x) is the lower confidence bound
    at the 10th percentile (beta = norm.ppf(0.10) ≈ -1.28, so we use
    mu + 1.28*sigma for minimization... wait, for minimization LCB is
    mu - beta*sigma with positive beta for exploration).
    
    For minimization, LCB = mu - kappa * sigma (explores where sigma is high).
    We use kappa corresponding to the 10th percentile of the standard normal,
    i.e., kappa = -norm.ppf(0.10) ≈ 1.28.
    
    Dividing by cost gives a cost-aware score: higher scores are preferred
    (lower predicted value, more exploration, lower cost).
    """
    t_start = time.time()
    TIME_LIMIT = 300.0  # Safety margin before 330s kill

    # Convert inputs to numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    # Safety: handle non-finite values
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    mu_safe = np.where(np.isfinite(mu), mu, f_best + 1.0)
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    # Quantile-based LCB for minimization:
    # LCB = mu - kappa * sigma, where kappa > 0 encourages exploration
    # Using the 10th percentile: kappa = -norm.ppf(0.10) ≈ 1.2816
    kappa = -norm.ppf(0.10)  # ≈ 1.2816

    lcb = mu_safe - kappa * sigma_safe

    # Cost-aware score: divide by cost (higher score = better)
    # Since cost is in [1, 2], dividing by cost penalizes expensive evaluations
    scores = lcb / cost_safe

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Time check
    if time.time() - t_start > TIME_LIMIT:
        return scores

    return scores
# EVOLVE-BLOCK-END
