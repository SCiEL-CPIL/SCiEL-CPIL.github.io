# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 330.0
_SAFETY_MARGIN = 5.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of GP posterior in objective units.
    sigma : ndarray, shape (2000,)
        Standard deviation of GP posterior in objective units.
    f_best : float
        Best observed value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next).
    """
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT - _SAFETY_MARGIN:
        mu = np.asarray(mu, dtype=float)
        cost = np.asarray(cost, dtype=float)
        mu = np.where(np.isfinite(mu), mu, f_best)
        cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
        return -mu / (cost + 1e-9)
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma >= 0), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # --- Expected Improvement (EI) for minimization ---
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # --- Upper Confidence Bound (UCB) for minimization ---
    # We want to minimize, so "upper confidence bound" in the sense of
    # optimistic lower bound: mu - kappa * sigma (we hope it's low)
    # The UCB score for minimization: -mu + kappa * sigma
    # Choose kappa based on exploration needs
    # Early on: more exploration; later: more exploitation
    # Use f_best as a proxy: if f_best is still high, explore more
    # Rastrigin minimum is 0, typical values can be up to ~100+
    # Normalize f_best to get exploration pressure
    # If f_best is large relative to expected scale, we need more exploration
    
    # Adaptive kappa: start high, decrease as f_best improves
    # f_best starts around ~10-50 for random points on rastrigin
    # Use a soft schedule
    kappa = 2.0 * np.exp(-f_best / 20.0) + 0.5
    # Clip kappa to reasonable range
    kappa = np.clip(kappa, 0.5, 3.0)
    
    ucb_score = -mu + kappa * sigma_safe
    
    # --- Probability of Improvement (PI) for minimization ---
    # P(mu < f_best) = Phi((f_best - mu) / sigma)
    pi_score = norm.cdf((f_best - mu) / sigma_safe)
    
    # --- Cost-normalized scores ---
    # Normalize each component by cost to get "improvement per unit cost"
    # Use a soft cost normalization to avoid extreme values
    cost_norm = cost  # cost is in [1, 2]
    
    ei_cost = ei / cost_norm
    ucb_cost = ucb_score / cost_norm
    pi_cost = pi_score / cost_norm
    
    # --- Hybrid: weighted combination ---
    # Weights adapt based on how close f_best is to the minimum (0)
    # When f_best is large: weight UCB more (explore)
    # When f_best is small: weight EI more (exploit)
    
    # f_best for rastrigin: minimum is 0, random init gives ~10-50
    # Use f_best directly as exploration pressure
    # w_ei increases as f_best decreases
    w_ei = 0.6 + 0.4 * np.exp(-f_best / 10.0)
    w_ucb = 1.0 - w_ei
    w_pi = 0.1  # small weight for PI as tiebreaker
    
    # Normalize weights
    w_sum = w_ei + w_ucb + w_pi
    w_ei /= w_sum
    w_ucb /= w_sum
    w_pi /= w_sum
    
    scores = w_ei * ei_cost + w_ucb * ucb_cost + w_pi * pi_cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
