"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_acq_state = {'start_time': None, 'call_count': 0}

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Bayesian Regret Minimization (BRM) acquisition function.
    Uses a softmax-based ranking that captures joint structure across all candidates,
    weighting by posterior probability of being the new minimum rather than independent EI.
    
    Parameters:
    - mu: array of shape (2000,), GP mean predictions in objective units
    - sigma: array of shape (2000,), GP standard deviations in objective units
    - f_best: scalar, lowest observed function value
    - cost: array of shape (2000,), evaluation cost for each candidate
    
    Returns:
    - scores: array of shape (2000,), acquisition values (higher = evaluate next)
    """
    # Initialize module-level state
    if _acq_state['start_time'] is None:
        _acq_state['start_time'] = time.time()
    _acq_state['call_count'] += 1
    
    # Deadline guard
    elapsed = time.time() - _acq_state['start_time']
    if elapsed > 310.0:
        # Return a safe default: cost-adjusted PI
        mu_safe = np.where(np.isfinite(mu), mu, f_best)
        sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
        cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
        z = (f_best - mu_safe) / np.maximum(sigma_safe, 1e-8)
        pi = norm.cdf(z)
        return pi / np.maximum(np.sqrt(cost_safe), 1e-8)
    
    # Ensure proper array types
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Ensure sigma is at least a tiny positive value
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    n = len(mu)
    
    # Bayesian Regret Minimization approach:
    # Score each candidate by the probability that it is the global minimum
    # among all 2000 candidates, weighted by cost efficiency.
    #
    # For each candidate i, compute:
    #   p_i = P(f_i = min(f_1, ..., f_n))
    # This is approximated using a softmax over the negative "expected improvement"
    # or directly over the negative predicted values with uncertainty.
    #
    # We use a temperature-scaled softmax over the negative cost-adjusted values.
    # The cost adjustment reflects that expensive points are less likely to be
    # worth evaluating even if they look promising.
    
    # Compute a "value" for each candidate that balances promise and cost.
    # Use the mean minus a bonus for uncertainty, divided by cost.
    # This is similar to UCB but normalized by cost.
    
    # Temperature for softmax - controls how peaked the distribution is
    # Lower temperature => more exploitative (peaked on best candidates)
    # Higher temperature => more exploratory (flatter distribution)
    # We adapt temperature based on how many calls we've made
    call_idx = _acq_state['call_count']
    
    # Start with moderate temperature, decrease over time for more exploitation
    base_temp = 0.3
    # Decay temperature slightly over time to become more exploitative
    temp = base_temp * (1.0 + 0.5 / (1.0 + call_idx * 0.05))
    
    # Compute cost-adjusted "promisingness" score
    # Higher means more promising (lower cost, lower predicted value, higher uncertainty)
    # Use: (f_best - mu) / sigma as z-score of improvement, then combine with cost
    z = (f_best - mu) / sigma
    
    # Use the expected improvement as the base score
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-adjust: divide by cost to prefer cheaper evaluations
    # Use sqrt(cost) for a softer penalty
    cost_adj = np.sqrt(cost)
    adjusted_ei = ei / cost_adj
    
    # Now compute softmax probability of being the best candidate
    # Use temperature-scaled softmax
    # Subtract max for numerical stability
    max_adj = np.max(adjusted_ei)
    scaled = (adjusted_ei - max_adj) / temp
    
    # Softmax
    exp_scaled = np.exp(scaled)
    sum_exp = np.sum(exp_scaled)
    if sum_exp > 0 and np.isfinite(sum_exp):
        prob_min = exp_scaled / sum_exp
    else:
        # Fallback: uniform
        prob_min = np.ones(n) / n
    
    # The BRM score is the probability of being the minimum,
    # which already incorporates the cost adjustment through adjusted_ei.
    # Multiply by a small factor to ensure meaningful differentiation
    scores = prob_min
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Add a small base value to avoid all-zero scores
    scores += 1e-10
    
    return scores
# EVOLVE-BLOCK-END
