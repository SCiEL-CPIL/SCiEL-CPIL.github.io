import numpy as np
from scipy.stats import norm
import time

# Track call count and time for safety
_call_count = 0
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Shekel 4d.
    
    Combines a UCB-EI blend (structural idea from the best-performing inspiration)
    with a stochastic cost model (SEI refinement from the current program).
    
    The base acquisition is a weighted blend of UCB and EI:
      - UCB: f_best - mu - kappa * sigma (exploitation + exploration)
      - EI:  E[max(f_best - f(x), 0)] (probabilistic improvement)
    
    The cost is modeled as a log-normal random variable, and the acquisition
    is the expected improvement per unit expected inverse cost:
      score = (w_ucb * ucb + w_ei * ei) * E[C^{-1}]
    
    E[C^{-1}] for C ~ LogNormal(mu_c, sigma_c) is exp(-mu_c + 0.5*sigma_c^2).
    With a small cost uncertainty (10% relative std), this is approximately
    1/cost but slightly penalizes higher-cost points more, capturing the
    convexity of 1/C.
    
    A small exploration bonus weighted by E[C^{-1}] encourages exploring
    cheap, uncertain regions.
    """
    global _call_count, _start_time
    
    # Initialize timing on first call
    if _start_time is None:
        _start_time = time.time()
    _call_count += 1
    
    # Deadline guard: if we've been running too long, return a safe fallback
    elapsed = time.time() - _start_time
    if elapsed > 280:  # Safety margin before 330s limit
        return 1.0 / cost
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # z-score for improvement: positive when mu < f_best
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # UCB component: f_best - mu - kappa * sigma
    # kappa controls exploration; 0.5 is moderate
    kappa = 0.5
    ucb = (f_best - mu) - kappa * sigma_safe
    ucb = np.maximum(ucb, 0.0)  # Clip to avoid negative UCB dominating
    
    # Blend UCB and EI: 0.6 * UCB + 0.4 * EI
    # UCB provides stronger exploration signal; EI provides probabilistic dampening
    w_ucb = 0.6
    w_ei = 0.4
    base_score = w_ucb * ucb + w_ei * ei
    
    # Model cost as log-normal with 10% relative standard deviation
    cost_mean = cost
    cost_std = 0.1 * cost_mean
    
    # Log-normal parameters
    var = cost_std ** 2
    sigma_c_sq = np.log(1.0 + var / (cost_mean ** 2))
    sigma_c = np.sqrt(sigma_c_sq)
    mu_c = np.log(cost_mean) - 0.5 * sigma_c_sq
    
    # E[C^{-1}] = exp(-mu_c + 0.5 * sigma_c^2)
    # This is slightly less than 1/cost_mean due to convexity of 1/x
    e_inv_c = np.exp(-mu_c + 0.5 * sigma_c_sq)
    
    # SEI-like score: base acquisition * E[C^{-1}]
    score = base_score * e_inv_c
    
    # Add a small exploration bonus for high-uncertainty regions
    # Scaled by relative sigma and weighted by E[C^{-1}] to prefer cheap regions
    sigma_scale = np.abs(f_best) + 1.0
    rel_sigma = sigma_safe / sigma_scale
    
    exploration_bonus = 0.05 * rel_sigma * e_inv_c
    
    score = score + exploration_bonus
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
