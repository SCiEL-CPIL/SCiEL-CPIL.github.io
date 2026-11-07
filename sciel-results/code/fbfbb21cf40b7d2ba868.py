"""Cost-aware acquisition for Shekel 4d BO with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: track start time for the acquisition call
_acq_start_time = None
_DEADLINE_MARGIN = 0.5  # seconds before the 330s limit to return safely


def _now():
    return time.time()


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Shekel 4d.
    
    Parameters:
        mu: (2000,) array of GP mean predictions in objective units
        sigma: (2000,) array of GP standard deviations in objective units
        f_best: scalar, lowest observed value
        cost: (2000,) array of evaluation costs (1 + mean(x_scaled_to_01))
    
    Returns:
        (2000,) array of finite scores; higher = more desirable to evaluate next
    """
    global _acq_start_time
    if _acq_start_time is None:
        _acq_start_time = _now()
    
    # Safety: if we're too close to the deadline, return a simple score
    elapsed = _now() - _acq_start_time
    if elapsed > 329.0:
        # Return a simple exploitation-based score as fallback
        sigma_safe = np.maximum(sigma, 1e-9)
        scores = (f_best - mu) / cost
        return np.where(np.isfinite(scores), scores, 0.0)
    
    n = len(mu)
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against non-finite or zero/negative cost
    cost_safe = np.where(np.isfinite(cost) & (cost > 1e-12), cost, 1.0)
    
    # Guard sigma to be positive
    sigma_safe = np.maximum(np.where(np.isfinite(sigma), sigma, 1e-9), 1e-9)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Ensure EI is non-negative (clip tiny negative values from numerical error)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement (POI)
    poi = norm.cdf(z)
    
    # UCB with adaptive beta based on data regime
    # Early on (low total cost), explore more; later, exploit
    # We don't have direct access to iteration count, but we can infer from f_best
    # and the spread of mu/sigma. Use a fixed reasonable beta for UCB.
    # A common choice: beta that balances exploration.
    # Let's use a moderate UCB term.
    beta_ucb = 0.5  # exploration coefficient
    
    ucb = mu + beta_ucb * sigma_safe  # for minimization, we want low mu + uncertainty
    
    # For minimization:
    # - EI rewards points where mu < f_best with high probability
    # - UCB for minimization: we want points with low (mu - beta*sigma)
    #   Actually, for minimization, UCB = mu - beta * sigma (lower is better)
    #   But since we're returning "higher = evaluate next", we need to invert.
    #   Let's define a "desirability" where higher is better.
    
    # Redefine: we want to maximize a score where higher means "more worth evaluating"
    # For minimization:
    #   EI is already "expected improvement" (positive when mu < f_best)
    #   POI is probability that f(x) < f_best
    #   For UCB in minimization context, we can use: -(mu - beta * sigma) = -mu + beta * sigma
    #   But this can be very negative. Let's normalize.
    
    # Alternative: use a blend of normalized signals
    # Normalize EI to [0, 1] roughly
    ei_max = np.max(ei) if n > 0 else 1.0
    if ei_max < 1e-12:
        ei_max = 1.0
    ei_norm = ei / ei_max
    
    # POI is already in [0, 1]
    
    # For UCB-like term in minimization: lower mu + higher sigma is more exploratory
    # We want to reward points that are either promising (low mu) or uncertain (high sigma)
    # Use: -mu / (|mu| + sigma + eps) as a normalized "potential"
    mu_scale = np.abs(mu) + sigma_safe + 1e-9
    ucb_score = -mu / mu_scale  # higher when mu is more negative relative to scale
    
    # Blend: strong exploitation (POI + EI) with moderate exploration (UCB)
    # Weights: 0.5 * POI + 0.3 * EI_norm + 0.2 * ucb_score
    # All divided by cost for cost-awareness
    
    score_unnormalized = 0.5 * poi + 0.3 * ei_norm + 0.2 * ucb_score
    
    # Cost-aware: divide by cost
    scores = score_unnormalized / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure scores are in a reasonable range
    # Clip to avoid extreme values
    scores = np.clip(scores, -1e6, 1e6)
    
    return scores


# Reset the start time tracker when the module is reloaded or when a new "session" starts
# We use a simple heuristic: if the elapsed time since the last call is large, reset
_last_call_time = None


def _maybe_reset_timer():
    global _acq_start_time, _last_call_time
    current = _now()
    if _last_call_time is not None:
        gap = current - _last_call_time
        # If more than 5 seconds since last call, assume new session
        if gap > 5.0:
            _acq_start_time = current
    _last_call_time = current


# Wrap the acquisition to include timer reset logic
_orig_acquisition = acquisition


def acquisition(mu, sigma, f_best, cost):
    _maybe_reset_timer()
    return _orig_acquisition(mu, sigma, f_best, cost)
# EVOLVE-BLOCK-END
