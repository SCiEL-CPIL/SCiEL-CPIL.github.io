"""Rewritten program: Thompson Sampling acquisition for cost-aware Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level cache for deterministic behavior within a single evaluation loop
_ts_state = {
    'call_count': 0,
    'last_f_best': None,
    'last_cost_sum': None,
    'last_seed': None,
}


def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling acquisition function.
    
    Draws a sample from the GP posterior at each candidate point and selects
    the candidate that minimizes (sample / cost), treating the objective as
    a stochastic reward-to-go ratio.
    
    Parameters:
        mu: (2000,) array of GP posterior means
        sigma: (2000,) array of GP posterior standard deviations
        f_best: scalar, lowest observed value
        cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
        (2000,) array of scores (higher = evaluate next)
    """
    # Deadline guard: track elapsed time since module load or first call
    if 'start_time' not in _ts_state:
        _ts_state['start_time'] = time.time()
    
    elapsed = time.time() - _ts_state['start_time']
    # Safety margin: return early if we're too close to the 330s limit
    if elapsed > 325:
        # Fallback: return simple EI-like scores to be safe
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        from scipy.stats import norm
        scores = ((f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)) / np.maximum(cost, 1e-9)
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Increment call counter for deterministic seeding within the loop
    _ts_state['call_count'] += 1
    call_idx = _ts_state['call_count']
    
    # Use a deterministic seed based on the call index and f_best to ensure
    # reproducibility within a single run while providing different samples
    # across calls
    seed = (call_idx * 1000003 + int(np.round(f_best * 1e6))) % (2**32)
    
    # Check if we're in the same evaluation context (same f_best and similar cost sum)
    # If so, we want to maintain consistency; otherwise, new random draw
    cost_sum = float(np.sum(cost))
    if (_ts_state['last_f_best'] is not None and 
        abs(f_best - _ts_state['last_f_best']) < 1e-12 and
        abs(cost_sum - _ts_state['last_cost_sum']) < 1e-9):
        # Same context, use same seed for consistency (shouldn't happen in normal flow)
        seed = _ts_state['last_seed']
    else:
        _ts_state['last_f_best'] = f_best
        _ts_state['last_cost_sum'] = cost_sum
        _ts_state['last_seed'] = seed
    
    rng = np.random.default_rng(seed)
    
    # Draw samples from the GP posterior: sample = mu - sigma * z
    # where z ~ N(0, 1)
    # We use mu - sigma * z so that the sample is a draw from N(mu, sigma^2)
    z_samples = rng.standard_normal(len(mu))
    samples = mu - sigma * z_samples
    
    # Ensure samples are finite
    samples = np.where(np.isfinite(samples), samples, f_best)
    
    # Cost-aware Thompson Sampling: minimize sample / cost
    # Higher score = better, so we return negative of (sample / cost)
    cost_safe = np.maximum(cost, 1e-9)
    scores = -(samples / cost_safe)
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Add a small exploration bonus for points with high uncertainty
    # This helps avoid getting stuck in local minima of the sample
    sigma_safe = np.maximum(sigma, 1e-9)
    # Normalize sigma to [0, 1] range for the bonus
    sigma_max = np.max(sigma_safe) if len(sigma_safe) > 0 else 1.0
    if sigma_max > 1e-12:
        exploration_bonus = 0.1 * (sigma_safe / sigma_max)
    else:
        exploration_bonus = np.zeros_like(sigma_safe)
    
    scores = scores + exploration_bonus
    
    # Final safety check
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
