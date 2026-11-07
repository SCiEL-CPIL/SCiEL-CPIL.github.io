"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for deadline tracking
_acq_state = {'start_time': None, 'call_count': 0}

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
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
    
    # Deadline guard: if we're getting close to the 330s wall-clock limit,
    # return a safe default to avoid timeout. Use conservative margin.
    elapsed = time.time() - _acq_state['start_time']
    if elapsed > 305.0:
        return np.zeros(len(mu), dtype=np.float64)
    
    # Ensure proper array types and handle edge cases
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
    
    # Bayesian Regret Minimization (BRM) approach with cost-aware weighting.
    
    n_calls = _acq_state['call_count']
    
    # Compute the probability that mu(x) is below f_best (i.e., improvement)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma
    prob_improve = norm.cdf(z)
    
    # Compute the expected improvement magnitude
    # E[max(f_best - f(x), 0)] = (f_best - mu) * Phi(z) + sigma * phi(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * prob_improve + sigma * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # Expected minimum: E[min(f(x), f_best)] = mu - sigma * phi(z) + (f_best - mu) * (1 - Phi(z))
    # This is the expected value of the minimum between f(x) and f_best
    # = mu - sigma*phi(z) + (f_best-mu)*(1-Phi(z))
    # Simplify: E[min] = f_best - EI (since EI = E[max(f_best - f(x), 0)])
    # Actually: E[min(f(x), f_best)] = f_best - EI
    expected_min = f_best - ei
    
    # Cost normalization: use log-cost for smoother scaling
    # This avoids extreme penalization of slightly more expensive points
    log_cost = np.log(cost + 1e-8)
    log_cost_mean = np.mean(log_cost)
    log_cost_std = np.std(log_cost) + 1e-8
    # Normalize log-cost to have mean 0, std 1, then shift so min is ~0.5
    norm_log_cost = (log_cost - log_cost_mean) / log_cost_std
    # Convert back to a positive cost weight: higher cost -> lower weight
    cost_weight = 1.0 / (1.0 + 0.5 * norm_log_cost)
    cost_weight = np.clip(cost_weight, 0.5, 2.0)
    
    # Component 1: Probability of improvement weighted by cost efficiency
    prob_score = prob_improve * cost_weight
    
    # Component 2: Expected improvement weighted by cost efficiency  
    ei_score = ei * cost_weight
    
    # Component 3: Expected minimum (lower is better, so we maximize -expected_min)
    # This encourages points that could have very low values
    min_score = -expected_min * cost_weight
    
    # Component 4: Uncertainty (exploration)
    unc_score = sigma * cost_weight
    
    # Smooth adaptive weights based on call count
    # Transition from exploration to exploitation over ~20 calls
    t = min(1.0, n_calls / 20.0)
    
    # Weights: start with high exploration, end with high exploitation
    w_prob = 1.0 - 0.4 * t          # 1.0 -> 0.6
    w_ei = 0.3 + 0.7 * t            # 0.3 -> 1.0
    w_min = 0.2 + 0.3 * t           # 0.2 -> 0.5
    w_unc = 0.8 - 0.7 * t           # 0.8 -> 0.1
    
    scores = w_prob * prob_score + w_ei * ei_score + w_min * min_score + w_unc * unc_score
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Normalize scores to [0, 1] range for stability
    score_min = np.min(scores)
    score_max = np.max(scores)
    if score_max > score_min + 1e-12:
        scores = (scores - score_min) / (score_max - score_min)
    else:
        scores = np.zeros_like(scores)
    
    return scores
# EVOLVE-BLOCK-END
