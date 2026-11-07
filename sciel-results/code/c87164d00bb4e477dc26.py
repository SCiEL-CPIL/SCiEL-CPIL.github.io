"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_start_time = time.time()
DEADLINE = 330.0
SAFETY_MARGIN = 2.0  # seconds


def _check_deadline():
    """Return True if we should abort early."""
    return (time.time() - _start_time) > (DEADLINE - SAFETY_MARGIN)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining:
    1. Expected Improvement (EI) for exploitation
    2. Cost-aware UCB for exploration
    3. Adaptive weighting based on remaining budget
    """
    if _check_deadline():
        # Fallback: simple EI/cost to avoid timeout
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / np.maximum(cost, 1e-9)

    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Number of points seen so far (proxy for budget usage)
    # We don't have explicit iteration count, but we can estimate from cost
    # Total budget is 30, initial cost from 5 uniform points ~ 5 * (1 + 0.5) = 7.5
    # Each additional point costs ~1.5 on average
    # So we've used roughly (total_cost - 7.5) / 1.5 additional points
    # But we don't have total_cost directly. Use the mean cost as a proxy for progress.
    
    # Better: use the cost array to estimate how much budget has been spent
    # The cost parameter represents the cost of evaluating each candidate point.
    # We can use the fraction of candidates with low cost as a heuristic.
    
    # Compute EI
    delta = f_best - mu
    z = delta / sigma_safe
    ei = delta * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI: divide by cost
    ei_cost = ei / cost_safe
    
    # UCB component: explore regions with high uncertainty
    # The exploration coefficient should decrease as budget is exhausted
    # Use a simple heuristic: beta based on the spread of costs
    cost_mean = np.mean(cost)
    cost_std = np.std(cost) + 1e-9
    
    # Beta: larger when we have more budget left (lower mean cost suggests we're early)
    # Actually, cost is the cost of evaluating each point, not cumulative.
    # Let's use a different approach: use the percentile of costs to gauge exploration
    cost_percentile = np.percentile(cost, 50)
    
    # Adaptive beta: start high for exploration, decrease for exploitation
    # Use the ratio of low-cost to high-cost points as a proxy
    low_cost_fraction = np.mean(cost < cost_percentile)
    
    # Beta between 0.5 and 2.0
    beta = 0.5 + 1.5 * low_cost_fraction
    
    # UCB: mu + beta * sigma, normalized by cost
    ucb = (mu + beta * sigma_safe) / cost_safe
    
    # Note: for minimization, high UCB means high predicted value + uncertainty
    # We want to find points where the true value is low, so we should use
    # lower confidence bound: mu - beta * sigma
    lcb = (mu - beta * sigma_safe) / cost_safe
    
    # Combine: use EI for exploitation and LCB for exploration
    # Weight: as we get closer to the minimum, rely more on EI
    # Use the minimum EI as a gauge of how close we are
    ei_max = np.max(ei) + 1e-9
    ei_normalized = ei / ei_max
    
    # Weight: early on, explore more (higher UCB weight); later, exploit more
    # Use the fraction of points that have high EI
    high_ei_fraction = np.mean(ei_normalized > 0.5)
    
    # w_ei: weight for EI, w_ucb: weight for LCB
    w_ei = 0.3 + 0.7 * high_ei_fraction
    w_ucb = 1.0 - w_ei
    
    # Final score: combination of EI/cost and LCB/cost
    # For minimization, we want low values, so LCB/cost should be high when
    # the lower bound is low and cost is low
    scores = w_ei * ei_cost + w_ucb * lcb
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
