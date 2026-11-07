import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 290.0  # safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Knowledge-gradient-inspired cost-aware acquisition function.
    
    Estimates the probability that each candidate is the global minimizer
    among all candidates, then discounts by evaluation cost. This is a
    fundamentally different approach from EI/UCB: it considers the full
    distribution of predicted values across all candidates simultaneously.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions in objective units.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed function value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clamp sigma to avoid division by zero and numerical instability
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Clamp cost to avoid division by zero
    cost_safe = np.maximum(cost, 1e-10)
    
    n = len(mu)
    
    # Knowledge gradient approximation:
    # For each candidate i, estimate P(x_i is the global minimizer)
    # = P(f(x_i) < min_{j != i} f(x_j))
    # 
    # Approximate by: P(f(x_i) < f_best) weighted by uncertainty
    # A better approximation: for each i, compute the probability that
    # mu_i - k*sigma_i is lower than all other mu_j - k*sigma_j
    #
    # We use a two-part score:
    # 1. Exploitation: how likely is this point to be near the minimum?
    # 2. Exploration: how much would we learn from evaluating here?
    
    # Part 1: Probability of being the minimum (knowledge gradient proxy)
    # For minimization, P(x_i is min) ~ P(mu_i - alpha*sigma_i < min_j(mu_j - alpha*sigma_j))
    # We approximate this by ranking candidates by their optimistic value
    
    alpha = 1.0  # optimism parameter
    
    # Optimistic values (lower is better)
    optimistic = mu - alpha * sigma_safe
    
    # Rank: 0 = best (lowest optimistic value), n-1 = worst
    # Use argsort of optimistic values
    order = np.argsort(optimistic)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    
    # Convert rank to a probability-like score: P(being min) ~ 1/(rank + 1)
    # But this is too crude. Better: use a softmax over negative optimistic values
    # with temperature based on the spread
    
    # Temperature: scale of the spread of optimistic values
    opt_spread = np.std(optimistic)
    if opt_spread < 1e-10:
        opt_spread = 1.0
    
    # Temperature for softmax: larger temperature = more uniform
    temperature = opt_spread * 0.5
    
    # Softmax over negative optimistic values (so lower optimistic = higher weight)
    neg_opt = -optimistic
    # Shift for numerical stability
    neg_opt_shifted = neg_opt - np.max(neg_opt)
    weights = np.exp(neg_opt_shifted / temperature)
    total_weight = np.sum(weights)
    if total_weight > 0:
        prob_min = weights / total_weight
    else:
        prob_min = np.ones(n) / n
    
    # Part 2: Information gain proxy
    # Points with high sigma provide more information
    # Normalize sigma to [0, 1] range
    sigma_min = np.min(sigma_safe)
    sigma_max = np.max(sigma_safe)
    if sigma_max - sigma_min > 1e-10:
        info_gain = (sigma_safe - sigma_min) / (sigma_max - sigma_min)
    else:
        info_gain = np.zeros(n)
    
    # Combine: knowledge gradient score
    # Weight between exploitation and exploration
    # Early on (high uncertainty), weight exploration more
    # Later (low uncertainty), weight exploitation more
    
    # Use the ratio of median sigma to the range of mu as an uncertainty indicator
    mu_range = np.ptp(mu)
    if mu_range < 1e-10:
        mu_range = 1.0
    
    uncertainty_ratio = np.median(sigma_safe) / mu_range
    # Clip to [0, 1]
    uncertainty_ratio = np.clip(uncertainty_ratio, 0.0, 1.0)
    
    # Exploration weight: higher when uncertainty is high
    expl_weight = 0.3 + 0.7 * uncertainty_ratio
    exploit_weight = 1.0 - expl_weight
    
    # Normalize prob_min to [0, 1]
    prob_min_norm = prob_min
    # It's already in [0, 1] from softmax
    
    # Combined score before cost
    combined = exploit_weight * prob_min_norm + expl_weight * info_gain
    
    # Cost-normalized score
    scores = combined / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard: if this took too long, return a simpler heuristic
    elapsed = time.time() - start_time
    if elapsed > 250.0:
        # Fallback: simple expected improvement per cost
        delta = f_best - mu
        z = delta / sigma_safe
        EI = delta * norm.cdf(z) + sigma_safe * norm.pdf(z)
        EI = np.maximum(EI, 0.0)
        simple_scores = EI / cost_safe
        simple_scores = np.where(np.isfinite(simple_scores), simple_scores, 0.0)
        return simple_scores
    
    return scores
